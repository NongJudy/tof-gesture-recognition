/**
 * my_tof.c
 * โปรแกรมอ่านข้อมูลจากเซ็นเซอร์ ToF (VL53L8CX) เข้าสู่หน่วยความจำ MCU
 * แล้วส่งออกทาง UART ในรูปแบบ CSV
 */

#include "my_tof.h"
#include <stdio.h>                    // สำหรับ printf (retarget ไป UART)
#include "53l8a1_ranging_sensor.h"    // API ของบอร์ด X-NUCLEO-53L8A1
#include "stm32f4xx_nucleo.h"         // BSP_COM_Init (เปิด Virtual COM Port)
#include "app_tof_pin_conf.h"         // ชื่อขา PWR_EN / LPn ของเซ็นเซอร์

/* ===== ฟังก์ชันจาก my_platform.c (โค้ดที่เราเขียนเอง) ===== */
extern uint8_t  my_platform_i2c_probe(uint16_t address);
extern void     my_platform_dwt_init(void);
extern uint32_t my_platform_cycles(void);
extern uint32_t my_platform_cycles_to_us(uint32_t cycles);
extern void     my_platform_stats_reset(void);
extern volatile uint32_t g_rd_calls, g_rd_bytes, g_rd_cycles;
extern volatile uint32_t g_rd_max_bytes, g_rd_max_cycles;

/* ===== โหมดการทำงาน =====
   1 = โหมดวัดเวลา  พิมพ์เฉพาะบรรทัด T (ปิดการส่งข้อมูล 64 ค่า
       เพื่อไม่ให้ UART ไปกวนค่าเวลาที่วัด)
   0 = โหมดปกติ     พิมพ์บรรทัด F ตามเดิม                        */
#define MY_TOF_TIMING_MODE   1

/* ===== ค่าตั้งต้นของเซ็นเซอร์ ===== */
#define MY_TIMING_BUDGET   (30U)   // เวลาเก็บแสงต่อการวัด (ms) ต้องอยู่ระหว่าง 5-100 ms

/* ===== หน่วยความจำของเรา (อยู่ใน RAM ของ STM32) ===== */
static uint16_t g_distance_mm[MY_TOF_ZONES];  // ระยะทาง 64 ค่า (mm)
static uint8_t  g_status[MY_TOF_ZONES];       // สถานะ 64 ค่า
static uint32_t g_frame_count = 0;            // นับจำนวนเฟรมที่อ่านมา

/* ===== ค่าที่วัดได้ของเฟรมล่าสุด ===== */
static uint32_t m_rd_calls  = 0;   // เรียก RdMulti กี่ครั้ง
static uint32_t m_rd_bytes  = 0;   // ไบต์รวมที่อ่าน
static uint32_t m_rd_us     = 0;   // เวลารวม (us)
static uint32_t m_max_bytes = 0;   // ไบต์ของก้อนใหญ่สุด
static uint32_t m_max_us    = 0;   // เวลาของก้อนใหญ่สุดนั้น (us)
static uint32_t m_uart_us   = 0;   // เวลาที่ UART ใช้ (us)

/* ตัวแปรรับข้อมูลดิบจาก driver */
static RANGING_SENSOR_Result_t Result;


/* =====================================================================
 *  my_tof_init : เตรียมระบบให้พร้อมวัด
 * ===================================================================== */
uint8_t my_tof_init(void)
{
    int32_t status;
    RANGING_SENSOR_ProfileConfig_t Profile;

    /* --- 0. เปิด UART และรีเซ็ตเซ็นเซอร์ (จำเป็นก่อนใช้งาน) --- */
    BSP_COM_Init(COM1);          // เปิด Virtual COM Port ให้ printf ส่งออกได้
    HAL_Delay(100);              // รอ UART ตั้งตัว ไม่งั้นตัวอักษรแรก ๆ หาย

    my_platform_dwt_init();      // เปิดตัวนับ cycle ของ Cortex-M4

    /* รีเซ็ตเซ็นเซอร์: ปิดไฟ -> รอ -> เปิดไฟ */
    HAL_GPIO_WritePin(VL53L8A1_PWR_EN_C_PORT, VL53L8A1_PWR_EN_C_PIN, GPIO_PIN_RESET);
    HAL_Delay(2);
    HAL_GPIO_WritePin(VL53L8A1_PWR_EN_C_PORT, VL53L8A1_PWR_EN_C_PIN, GPIO_PIN_SET);
    HAL_Delay(2);

    /* --- 1. เริ่มต้นเซ็นเซอร์ (โหลด firmware เข้าตัวเซ็นเซอร์) --- */
    status = VL53L8A1_RANGING_SENSOR_Init(VL53L8A1_DEV_CENTER);
    if (status != BSP_ERROR_NONE)
    {
        printf("ERROR: sensor init failed (%ld)\r\n", (long)status);
        return 1;   // แจ้งว่าล้มเหลว
    }

    /* --- T1: เช็คว่าเซ็นเซอร์ตอบ ACK ที่ address 0x52 ---
       ★ ต้องอยู่ "หลัง" RANGING_SENSOR_Init เท่านั้น
         เพราะฮาร์ดแวร์ I2C1 เพิ่งถูกเปิดใช้งานภายในฟังก์ชันนั้น
         (ลำดับ: RANGING_SENSOR_Init -> RegisterBusIO -> BSP_I2C1_Init) */
    if (my_platform_i2c_probe(0x52) == 0) {
        printf("T1 PASS: sensor ACK at 0x52\r\n");
    } else {
        printf("T1 FAIL: no ACK at 0x52\r\n");
    }

    /* --- 2. ตั้งค่าโปรไฟล์การวัด --- */
    Profile.RangingProfile = RS_PROFILE_8x8_CONTINUOUS;  // 8x8 + โหมดต่อเนื่อง
    Profile.TimingBudget   = MY_TIMING_BUDGET;           // เวลาเก็บแสง 30 ms
    Profile.Frequency      = MY_TOF_FREQ_HZ;             // ความถี่วัด (Hz)
    Profile.EnableAmbient  = 0;                          // ยังไม่เอาข้อมูลแสงรอบข้าง
    Profile.EnableSignal   = 0;                          // ยังไม่เอาความแรงสัญญาณ

    VL53L8A1_RANGING_SENSOR_ConfigProfile(VL53L8A1_DEV_CENTER, &Profile);

    /* --- 3. สั่งเริ่มวัด --- */
    status = VL53L8A1_RANGING_SENSOR_Start(VL53L8A1_DEV_CENTER, RS_MODE_BLOCKING_CONTINUOUS);
    if (status != BSP_ERROR_NONE)
    {
        printf("ERROR: sensor start failed (%ld)\r\n", (long)status);
        return 1;
    }

    printf("MY_TOF: init OK (8x8 @ %d Hz)\r\n", MY_TOF_FREQ_HZ);

    /* บอกความถี่ CPU จริง เพื่อให้ตรวจสอบการแปลง cycle -> us ย้อนหลังได้ */
    printf("CLK,%lu\r\n", (unsigned long)SystemCoreClock);

#if MY_TOF_TIMING_MODE
    /* หัวตารางของบรรทัด T ให้ Python อ่านง่าย */
    printf("H,frame,rd_calls,rd_bytes,rd_us,max_bytes,max_us,uart_us\r\n");
#endif

    return 0;   // สำเร็จ
}


/* =====================================================================
 *  my_tof_read_frame : อ่าน 1 เฟรมเข้าหน่วยความจำของเรา
 * ===================================================================== */
uint8_t my_tof_read_frame(void)
{
    uint32_t i;
    int32_t  status;

    my_platform_stats_reset();   // ล้างตัวนับก่อนอ่านเฟรมนี้

    /* ขอข้อมูลล่าสุดจากเซ็นเซอร์ (blocking = รอจนกว่าจะได้) */
    status = VL53L8A1_RANGING_SENSOR_GetDistance(VL53L8A1_DEV_CENTER, &Result);
    if (status != BSP_ERROR_NONE)
    {
        return 0;   // ยังไม่มีข้อมูลใหม่
    }

    /* เก็บค่าที่วัดได้ทันที ก่อนอย่างอื่นจะไปแก้ตัวนับ */
    m_rd_calls  = g_rd_calls;
    m_rd_bytes  = g_rd_bytes;
    m_rd_us     = my_platform_cycles_to_us(g_rd_cycles);
    m_max_bytes = g_rd_max_bytes;
    m_max_us    = my_platform_cycles_to_us(g_rd_max_cycles);

    /* คัดลอกข้อมูลจาก driver ลง array ของเราเอง
       << นี่คือ "อ่านข้อมูลเข้าสู่หน่วยความจำของ MCU" >> */
    for (i = 0; i < Result.NumberOfZones && i < MY_TOF_ZONES; i++)
    {
        g_distance_mm[i] = (uint16_t)Result.ZoneResult[i].Distance[0];
        g_status[i]      = (uint8_t)Result.ZoneResult[i].Status[0];
    }

    g_frame_count++;    // นับเฟรม
    return 1;           // อ่านสำเร็จ
}


/* =====================================================================
 *  my_tof_send_frame : ส่งข้อมูลออก UART
 * ===================================================================== */
void my_tof_send_frame(void)
{
    uint32_t t0, t1;

#if MY_TOF_TIMING_MODE

    /* โหมดวัดเวลา: พิมพ์เฉพาะสถิติ */
    t0 = my_platform_cycles();

    printf("T,%lu,%lu,%lu,%lu,%lu,%lu,%lu\r\n",
           (unsigned long)g_frame_count,
           (unsigned long)m_rd_calls,
           (unsigned long)m_rd_bytes,
           (unsigned long)m_rd_us,
           (unsigned long)m_max_bytes,
           (unsigned long)m_max_us,
           (unsigned long)m_uart_us);

    t1 = my_platform_cycles();

    /* เวลา UART วัดจากเฟรมนี้ แต่รายงานในบรรทัดถัดไป
       เพราะพิมพ์ค่าของตัวเองลงในบรรทัดตัวเองไม่ได้ */
    m_uart_us = my_platform_cycles_to_us(t1 - t0);

#else

    /* โหมดปกติ: ส่งข้อมูล 64 ค่าเป็น CSV
       รูปแบบ:  F,<เลขเฟรม>,<d0>,<d1>,...,<d63> */
    uint32_t i;

    t0 = my_platform_cycles();

    printf("F,%lu", (unsigned long)g_frame_count);
    for (i = 0; i < MY_TOF_ZONES; i++)
    {
        printf(",%u", (unsigned int)g_distance_mm[i]);
    }
    printf("\r\n");

    t1 = my_platform_cycles();
    m_uart_us = my_platform_cycles_to_us(t1 - t0);

#endif
}
