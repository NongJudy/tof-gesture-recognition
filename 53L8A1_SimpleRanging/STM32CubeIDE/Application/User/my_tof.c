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

/* ===== ฟังก์ชันจาก my_uart.c ===== */
extern void my_uart_init(void);

/* ===== ธงจากขา INT ของเซ็นเซอร์ =====
   ประกาศจริงใน app_tof.c บรรทัด 47
   ถูกตั้งเป็น 1 ใน HAL_GPIO_EXTI_Callback (app_tof_pin_conf.c) */
extern volatile uint8_t ToF_EventDetected;

/* ===== สวิตช์เปรียบเทียบ =====
   MY_TOF_USE_INT      1 = รอขา INT บอก (ASYNC) | 0 = วน poll เอง (BLOCKING)
   MY_TOF_TIMING_MODE  1 = พิมพ์บรรทัด T (วัดเวลา) | 0 = พิมพ์ F + S (ข้อมูลจริง)
   (ส่วนความละเอียด 4x4/8x8 อยู่ที่ MY_TOF_USE_4X4 ใน my_tof.h)          */
#define MY_TOF_USE_INT       1
#define MY_TOF_TIMING_MODE   1

/* ===== การทดลองที่ 1: หน่วงเวลาเพื่อทดสอบว่าเซ็นเซอร์รอเราหรือไม่ =====
   ใส่การหน่วงหลังส่งข้อมูลเสร็จ แล้ววัดว่าคาบเปลี่ยนอย่างไร
     คาบ = ฐาน + หน่วง (เส้นตรง)  -> เซ็นเซอร์รอเรา ทำงานต่อคิวกัน
     คาบ คงที่แล้วกระโดดเป็น 2 เท่า -> เซ็นเซอร์วิ่งเอง ไม่รอเรา
   ทดลองไล่ค่า: 0, 2000, 4000, 6000, 8000, 12000, 16000              */
#define MY_TOF_DELAY_US      12000

/* ===== เวลาเก็บแสงต่อการวัด (ms) =====
   ต้องน้อยกว่าคาบของความถี่ที่ตั้งไว้
   หมายเหตุ: ทดสอบแล้วว่า 5 กับ 10 ms ให้อัตราเฟรมเท่ากัน (t = 0.42)
             จึงกลับมาใช้ 10 ms เพราะเก็บแสงได้มากกว่า ค่าแม่นกว่า   */
#if MY_TOF_USE_4X4
  #define MY_TIMING_BUDGET   (10U)
#else
  #define MY_TIMING_BUDGET   (30U)
#endif

/* จำนวนเฟรมที่ใช้เฉลี่ยตอนวัดอัตราเฟรมจริง */
#define MY_RATE_WINDOW       (60U)

/* ===== หน่วยความจำของเรา (อยู่ใน RAM ของ STM32) ===== */
static uint16_t g_distance_mm[MY_TOF_ZONES];  // ระยะทาง (mm)
static uint8_t  g_status[MY_TOF_ZONES];       // สถานะแต่ละช่อง
static uint32_t g_frame_count = 0;            // นับจำนวนเฟรมที่อ่านมา

/* ===== ค่าที่วัดได้ของเฟรมล่าสุด ===== */
static uint32_t m_rd_calls  = 0;
static uint32_t m_rd_bytes  = 0;
static uint32_t m_rd_us     = 0;
static uint32_t m_max_bytes = 0;
static uint32_t m_max_us    = 0;
static uint32_t m_uart_us   = 0;

/* ===== ตัวแปรวัดอัตราเฟรมจริงที่ทำได้ ===== */
static uint32_t m_rate_t0 = 0;   // เวลาเริ่มนับ (ms)
static uint32_t m_rate_n  = 0;   // นับเฟรมในหน้าต่างนี้

/* ตัวแปรรับข้อมูลดิบจาก driver */
static RANGING_SENSOR_Result_t Result;


/* =====================================================================
 *  my_tof_init : เตรียมระบบให้พร้อมวัด
 * ===================================================================== */
uint8_t my_tof_init(void)
{
    int32_t status;
    RANGING_SENSOR_ProfileConfig_t Profile;

    /* --- 0. เปิด UART และรีเซ็ตเซ็นเซอร์ --- */
    BSP_COM_Init(COM1);
    HAL_Delay(100);              // รอ UART ตั้งตัว ไม่งั้นตัวอักษรแรก ๆ หาย

    my_uart_init();              // เปิด UART แบบ interrupt (ring buffer)
    my_platform_dwt_init();      // เปิดตัวนับ cycle ของ Cortex-M4

    HAL_GPIO_WritePin(VL53L8A1_PWR_EN_C_PORT, VL53L8A1_PWR_EN_C_PIN, GPIO_PIN_RESET);
    HAL_Delay(2);
    HAL_GPIO_WritePin(VL53L8A1_PWR_EN_C_PORT, VL53L8A1_PWR_EN_C_PIN, GPIO_PIN_SET);
    HAL_Delay(2);

    /* --- 1. เริ่มต้นเซ็นเซอร์ (โหลด firmware เข้าตัวเซ็นเซอร์) --- */
    status = VL53L8A1_RANGING_SENSOR_Init(VL53L8A1_DEV_CENTER);
    if (status != BSP_ERROR_NONE)
    {
        printf("ERROR: sensor init failed (%ld)\r\n", (long)status);
        return 1;
    }

    /* --- T1: เช็คว่าเซ็นเซอร์ตอบ ACK ที่ 0x52 (ต้องอยู่หลัง Init เท่านั้น) --- */
    if (my_platform_i2c_probe(0x52) == 0) {
        printf("T1 PASS: sensor ACK at 0x52\r\n");
    } else {
        printf("T1 FAIL: no ACK at 0x52\r\n");
    }

    /* --- 2. ตั้งค่าโปรไฟล์การวัด --- */
#if MY_TOF_USE_4X4
    Profile.RangingProfile = RS_PROFILE_4x4_CONTINUOUS;
#else
    Profile.RangingProfile = RS_PROFILE_8x8_CONTINUOUS;
#endif
    Profile.TimingBudget   = MY_TIMING_BUDGET;
    Profile.Frequency      = MY_TOF_FREQ_HZ;
    Profile.EnableAmbient  = 0;
    Profile.EnableSignal   = 0;

    VL53L8A1_RANGING_SENSOR_ConfigProfile(VL53L8A1_DEV_CENTER, &Profile);

    /* --- 3. สั่งเริ่มวัด --- */
#if MY_TOF_USE_INT
    ToF_EventDetected = 0;
    status = VL53L8A1_RANGING_SENSOR_Start(VL53L8A1_DEV_CENTER,
                                           RS_MODE_ASYNC_CONTINUOUS);
#else
    status = VL53L8A1_RANGING_SENSOR_Start(VL53L8A1_DEV_CENTER,
                                           RS_MODE_BLOCKING_CONTINUOUS);
#endif
    if (status != BSP_ERROR_NONE)
    {
        printf("ERROR: sensor start failed (%ld)\r\n", (long)status);
        return 1;
    }

    /* บอกเงื่อนไขที่ใช้ ให้ไฟล์ log อธิบายตัวเองได้ ไม่ต้องพึ่งชื่อไฟล์ */
    printf("MY_TOF: init OK (%s @ %d Hz, budget %d ms, %s, delay %d us)\r\n",
#if MY_TOF_USE_4X4
           "4x4",
#else
           "8x8",
#endif
           MY_TOF_FREQ_HZ, MY_TIMING_BUDGET,
#if MY_TOF_USE_INT
           "INT/async",
#else
           "polling/blocking",
#endif
           MY_TOF_DELAY_US);

    printf("CLK,%lu\r\n", (unsigned long)SystemCoreClock);

#if MY_TOF_TIMING_MODE
    printf("H,frame,rd_calls,rd_bytes,rd_us,max_bytes,max_us,uart_us\r\n");
#endif

    m_rate_t0 = HAL_GetTick();
    return 0;
}


/* =====================================================================
 *  my_tof_read_frame : อ่าน 1 เฟรมเข้าหน่วยความจำของเรา
 * ===================================================================== */
uint8_t my_tof_read_frame(void)
{
    uint32_t i;
    int32_t  status;

#if MY_TOF_USE_INT
    /* ยังไม่มีสัญญาณจากขา INT -> ออกทันที ไม่ยิง I2C ถามเลย */
    if (ToF_EventDetected == 0U)
    {
        return 0;
    }
    ToF_EventDetected = 0;
#endif

    my_platform_stats_reset();

    status = VL53L8A1_RANGING_SENSOR_GetDistance(VL53L8A1_DEV_CENTER, &Result);
    if (status != BSP_ERROR_NONE)
    {
        return 0;
    }

    /* เก็บค่าที่วัดได้ทันที ก่อนอย่างอื่นจะไปแก้ตัวนับ */
    m_rd_calls  = g_rd_calls;
    m_rd_bytes  = g_rd_bytes;
    m_rd_us     = my_platform_cycles_to_us(g_rd_cycles);
    m_max_bytes = g_rd_max_bytes;
    m_max_us    = my_platform_cycles_to_us(g_rd_max_cycles);

    for (i = 0; i < Result.NumberOfZones && i < MY_TOF_ZONES; i++)
    {
        g_distance_mm[i] = (uint16_t)Result.ZoneResult[i].Distance[0];
        g_status[i]      = (uint8_t)Result.ZoneResult[i].Status[0];
    }

    g_frame_count++;
    return 1;
}


/* =====================================================================
 *  my_test_delay_us : หน่วงเวลาละเอียดระดับไมโครวินาที
 *  ใช้เฉพาะการทดลอง ไม่ใช่ส่วนของระบบจริง
 *  วนอ่านรีจิสเตอร์ฮาร์ดแวร์ คอมไพเลอร์จึงตัดทิ้งไม่ได้
 * ===================================================================== */
static void my_test_delay_us(uint32_t us)
{
    uint32_t start = my_platform_cycles();
    uint32_t need  = us * (SystemCoreClock / 1000000U);

    while ((my_platform_cycles() - start) < need) { }
}


/* =====================================================================
 *  my_tof_send_frame : ส่งข้อมูลออก UART
 * ===================================================================== */
void my_tof_send_frame(void)
{
    uint32_t t0, t1, now, ms;

#if MY_TOF_TIMING_MODE

    t0 = my_platform_cycles();
    printf("T,%lu,%lu,%lu,%lu,%lu,%lu,%lu\r\n",
           (unsigned long)g_frame_count, (unsigned long)m_rd_calls,
           (unsigned long)m_rd_bytes,    (unsigned long)m_rd_us,
           (unsigned long)m_max_bytes,   (unsigned long)m_max_us,
           (unsigned long)m_uart_us);
    t1 = my_platform_cycles();
    m_uart_us = my_platform_cycles_to_us(t1 - t0);

#else

    uint32_t i;
    t0 = my_platform_cycles();

    /* บรรทัดที่ 1 - ระยะทาง */
    printf("F,%lu", (unsigned long)g_frame_count);
    for (i = 0; i < MY_TOF_ZONES; i++) {
        printf(",%u", (unsigned int)g_distance_mm[i]);
    }
    printf("\r\n");

    /* บรรทัดที่ 2 - ค่าสถานะ
       หมายเหตุ: BSP ของ ST แปลงค่าก่อนส่งให้เรา (vl53l8cx.c บรรทัด 781)
       สถานะดิบ 5 หรือ 9 -> ถูกแปลงเป็น 0 = วัดได้ถูกต้อง
       สถานะดิบ 0        -> ถูกแปลงเป็น 255 = ไม่มีข้อมูลใหม่           */
    printf("S,%lu", (unsigned long)g_frame_count);
    for (i = 0; i < MY_TOF_ZONES; i++) {
        printf(",%u", (unsigned int)g_status[i]);
    }
    printf("\r\n");

    t1 = my_platform_cycles();
    m_uart_us = my_platform_cycles_to_us(t1 - t0);

#endif

    /* ===== วัดอัตราเฟรมจริงที่ระบบทำได้ =====
       รูปแบบ  R,<เลขเฟรม>,<จำนวนเฟรม>,<เวลาที่ใช้ ms> */
    m_rate_n++;
    if (m_rate_n >= MY_RATE_WINDOW)
    {
        now = HAL_GetTick();
        ms  = now - m_rate_t0;
        printf("R,%lu,%lu,%lu\r\n",
               (unsigned long)g_frame_count, (unsigned long)m_rate_n,
               (unsigned long)ms);
        m_rate_t0 = now;
        m_rate_n  = 0;
    }

    /* ===== การทดลองที่ 1 =====
       ถ่วงเวลาให้ "งานของเรา" ใช้เวลานานขึ้น
       ถ้าเซ็นเซอร์รอเรา  -> คาบเพิ่มตามที่หน่วงแบบเส้นตรง
       ถ้าเซ็นเซอร์ไม่รอ  -> คาบคงที่ จนพลาดรอบแล้วกระโดดเป็น 2 เท่า */
#if MY_TOF_DELAY_US > 0
    my_test_delay_us(MY_TOF_DELAY_US);
#endif
}
