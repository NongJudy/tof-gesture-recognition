/**
 * my_tof.c
 * อ่านข้อมูลจากเซ็นเซอร์ ToF (VL53L8CX) เข้าหน่วยความจำ MCU แล้วส่งออก UART
 */

#include "my_tof.h"
#include <stdio.h>
#include "53l8a1_ranging_sensor.h"    // API ของบอร์ด X-NUCLEO-53L8A1
#include "stm32f4xx_nucleo.h"         // BSP_COM_Init
#include "app_tof_pin_conf.h"         // ขา PWR_EN / LPn
#include "vl53l8cx.h"                 // VL53L8CX_Object_t
#include "vl53l8cx_api.h"             // vl53l8cx_get_ranging_data, streamcount

/* ===== ฟังก์ชันจาก my_platform.c ===== */
extern uint8_t  my_platform_i2c_probe(uint16_t address);
extern void     my_platform_dwt_init(void);
extern uint32_t my_platform_cycles(void);
extern uint32_t my_platform_cycles_to_us(uint32_t cycles);
extern void     my_platform_stats_reset(void);
extern volatile uint32_t g_rd_calls, g_rd_bytes, g_rd_cycles;
extern volatile uint32_t g_rd_max_bytes, g_rd_max_cycles;

/* ===== ฟังก์ชันจาก my_uart.c ===== */
extern void my_uart_init(void);

/* ===== ธงจากขา INT (ตั้งใน HAL_GPIO_EXTI_Callback) ===== */
extern volatile uint8_t ToF_EventDetected;

/* ===== object ของเซ็นเซอร์ที่ BSP สร้างไว้ (53l8a1_ranging_sensor.c:42) ===== */
extern void *VL53L8A1_RANGING_SENSOR_CompObj[];

/* ===== สวิตช์เปรียบเทียบ ===== */
#define MY_TOF_USE_INT       1   // 1 = รอขา INT | 0 = วน poll
#define MY_TOF_TIMING_MODE    1  // 1 = บรรทัด T | 0 = บรรทัด F + S + G

/* ===== สวิตช์อ่านตรง =====
   1 = เรียก vl53l8cx_get_ranging_data() ตรง ยิง I2C ครั้งเดียว
   0 = ผ่าน BSP ยิง I2C 4 ครั้ง (2 ครั้งซ้ำซ้อน: check_data_ready + get_resolution)
   ★ โหมด 1 ได้ target_status แบบดิบ (5 = ถูกต้อง) ตรงกับชุดข้อมูลของ ST */
#define MY_TOF_FAST_READ     1

/* ===== การทดลองหน่วงเวลา (จบแล้ว) =====
   ผล: คาบ = 23.24 + 0.981 x หน่วง   (R^2 = 0.9997, 6 จุด) */
#define MY_TOF_DELAY_US      0

/* ===== เวลาเก็บแสงต่อการวัด (ms) ต้องน้อยกว่าคาบ ===== */
#if MY_TOF_USE_4X4
  #define MY_TIMING_BUDGET   (10U)
#else
  #define MY_TIMING_BUDGET   (30U)
#endif

#define MY_RATE_WINDOW       (60U)   // เฉลี่ยอัตราเฟรมทุกกี่เฟรม

/* ===== หน่วยความจำของเรา ===== */
static uint16_t g_distance_mm[MY_TOF_ZONES];  // ระยะทาง (mm)
static uint8_t  g_status[MY_TOF_ZONES];       // สถานะแต่ละช่อง
static uint32_t g_signal[MY_TOF_ZONES];       // ความแรงแสงสะท้อนต่อ SPAD
static uint32_t g_frame_count = 0;

/* ===== ค่าที่วัดได้ของเฟรมล่าสุด ===== */
static uint32_t m_rd_calls = 0, m_rd_bytes = 0, m_rd_us = 0;
static uint32_t m_max_bytes = 0, m_max_us = 0, m_uart_us = 0;

/* ===== ตัววัดอัตราเฟรมจริง ===== */
static uint32_t m_rate_t0 = 0, m_rate_n = 0;

/* ===== ★ ตัวตรวจสอบว่าทุกเฟรมเป็นข้อมูลใหม่จริง ★
   เซ็นเซอร์มีตัวนับของตัวเองชื่อ streamcount (vl53l8cx_api.h บรรทัด 277)
   ที่บวกขึ้นเองทุกครั้งที่วัดเสร็จ 1 รอบ  เราแค่อ่านมาเทียบ

   จำเป็นเพราะเราตัด vl53l8cx_check_data_ready ออกไปตอนทำโหมดอ่านตรง
   ซึ่งเป็นฟังก์ชันที่เดิมคอยเช็คว่า "ข้อมูลนี้ใหม่จริงไหม"
   ถ้าไม่ตรวจ อาจอ่านข้อมูลชุดเดิมซ้ำแล้วนับเป็นเฟรมใหม่ ทำให้ตัวเลขหลอกตา

   ผลต่างที่ควรได้คือ 1 เสมอ
     delta = 0  -> อ่านข้อมูลเดิมซ้ำ (นับ dup)
     delta > 1  -> เซ็นเซอร์ผลิตเร็วกว่าที่เราอ่าน มีเฟรมหาย (นับ skip)
   streamcount เป็น uint8_t วนกลับที่ 255 จึงต้องคำนวณผลต่างแบบ 8 บิต */
static VL53L8CX_Configuration *m_dev = NULL;
static uint8_t  m_stream = 0, m_stream_prev = 0, m_stream_delta = 0;
static uint32_t m_dup = 0, m_skip = 0;

/* ===== ตัวรับข้อมูลดิบ ===== */
#if MY_TOF_FAST_READ
static VL53L8CX_ResultsData    RawData;   // โครงสร้างของ ULD โดยตรง
#else
static RANGING_SENSOR_Result_t Result;    // โครงสร้างของ BSP
#endif


/* =====================================================================
 *  my_tof_init : เตรียมระบบให้พร้อมวัด
 * ===================================================================== */
uint8_t my_tof_init(void)
{
    int32_t status;
    RANGING_SENSOR_ProfileConfig_t Profile;

    /* --- 0. เปิด UART และรีเซ็ตเซ็นเซอร์ --- */
    BSP_COM_Init(COM1);
    HAL_Delay(100);              // รอ UART ตั้งตัว ไม่งั้นตัวอักษรแรกหาย

    my_uart_init();              // UART แบบ interrupt
    my_platform_dwt_init();      // ตัวนับ cycle ของ Cortex-M4

    HAL_GPIO_WritePin(VL53L8A1_PWR_EN_C_PORT, VL53L8A1_PWR_EN_C_PIN, GPIO_PIN_RESET);
    HAL_Delay(2);
    HAL_GPIO_WritePin(VL53L8A1_PWR_EN_C_PORT, VL53L8A1_PWR_EN_C_PIN, GPIO_PIN_SET);
    HAL_Delay(2);

    /* --- 1. เริ่มต้นเซ็นเซอร์ --- */
    status = VL53L8A1_RANGING_SENSOR_Init(VL53L8A1_DEV_CENTER);
    if (status != BSP_ERROR_NONE)
    {
        printf("ERROR: sensor init failed (%ld)\r\n", (long)status);
        return 1;
    }

    /* --- T1: เช็ค ACK ที่ 0x52 (ต้องอยู่หลัง Init) --- */
    if (my_platform_i2c_probe(0x52) == 0) {
        printf("T1 PASS: sensor ACK at 0x52\r\n");
    } else {
        printf("T1 FAIL: no ACK at 0x52\r\n");
    }

    /* --- เก็บที่อยู่ของ Dev ไว้ครั้งเดียว ใช้ทั้งการอ่านตรงและอ่าน streamcount --- */
    {
        VL53L8CX_Object_t *pObj =
            (VL53L8CX_Object_t *)VL53L8A1_RANGING_SENSOR_CompObj[VL53L8A1_DEV_CENTER];
        if (pObj == NULL)
        {
            printf("ERROR: sensor object is NULL\r\n");
            return 1;
        }
        m_dev = &pObj->Dev;
    }

    /* --- 2. ตั้งค่าโปรไฟล์ --- */
#if MY_TOF_USE_4X4
    Profile.RangingProfile = RS_PROFILE_4x4_CONTINUOUS;
#else
    Profile.RangingProfile = RS_PROFILE_8x8_CONTINUOUS;
#endif
    Profile.TimingBudget   = MY_TIMING_BUDGET;
    Profile.Frequency      = MY_TOF_FREQ_HZ;
    Profile.EnableAmbient  = 0;    // ไม่ใช้แสงรอบข้าง
    Profile.EnableSignal   = 1;    // ใช้ความแรงสัญญาณ (โมเดล ST ใช้ 8x8x2)

    VL53L8A1_RANGING_SENSOR_ConfigProfile(VL53L8A1_DEV_CENTER, &Profile);

    /* --- 3. เริ่มวัด --- */
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

    /* บอกเงื่อนไขที่ใช้ ให้ไฟล์ log อธิบายตัวเองได้ */
    printf("MY_TOF: init OK (%s @ %d Hz, budget %d ms, %s, delay %d us, "
           "signal ON, read=%s, status=%s, clk=HSE)\r\n",
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
           MY_TOF_DELAY_US,
#if MY_TOF_FAST_READ
           "direct", "raw(5=valid)"
#else
           "bsp", "bsp(0=valid)"
#endif
           );

    printf("CLK,%lu\r\n", (unsigned long)SystemCoreClock);

#if MY_TOF_TIMING_MODE
    /* หัวตาราง: 4 ช่องท้ายคือตัวตรวจสอบความถูกต้องของการนับเฟรม */
    printf("H,frame,rd_calls,rd_bytes,rd_us,max_bytes,max_us,uart_us,"
           "stream,delta,dup,skip\r\n");
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

#if MY_TOF_USE_INT
    /* ยังไม่มีสัญญาณจากขา INT -> ออกทันที ไม่ยิง I2C เลย */
    if (ToF_EventDetected == 0U)
    {
        return 0;
    }
    ToF_EventDetected = 0;
#endif

    my_platform_stats_reset();

#if MY_TOF_FAST_READ

    /* --- เรียก ULD ตรง ยิง I2C ครั้งเดียว --- */
    if (vl53l8cx_get_ranging_data(m_dev, &RawData) != VL53L8CX_STATUS_OK)
    {
        return 0;
    }

    m_rd_calls  = g_rd_calls;
    m_rd_bytes  = g_rd_bytes;
    m_rd_us     = my_platform_cycles_to_us(g_rd_cycles);
    m_max_bytes = g_rd_max_bytes;
    m_max_us    = my_platform_cycles_to_us(g_rd_max_cycles);

    /* คัดลอกเฉพาะเป้าแรกของแต่ละช่อง (j = 0)
       ULD เก็บแบบ [NB_TARGET_PER_ZONE * ช่อง + เป้าที่] */
    for (i = 0; i < MY_TOF_ZONES; i++)
    {
        uint32_t k = (uint32_t)VL53L8CX_NB_TARGET_PER_ZONE * i;

        g_distance_mm[i] = (uint16_t)RawData.distance_mm[k];
        g_status[i]      = RawData.target_status[k];   // ค่าดิบ 5 = ถูกต้อง
        g_signal[i]      = RawData.signal_per_spad[k];
    }

#else

    /* --- ผ่าน BSP ยิง I2C 4 ครั้ง --- */
    if (VL53L8A1_RANGING_SENSOR_GetDistance(VL53L8A1_DEV_CENTER, &Result)
            != BSP_ERROR_NONE)
    {
        return 0;
    }

    m_rd_calls  = g_rd_calls;
    m_rd_bytes  = g_rd_bytes;
    m_rd_us     = my_platform_cycles_to_us(g_rd_cycles);
    m_max_bytes = g_rd_max_bytes;
    m_max_us    = my_platform_cycles_to_us(g_rd_max_cycles);

    for (i = 0; i < Result.NumberOfZones && i < MY_TOF_ZONES; i++)
    {
        g_distance_mm[i] = (uint16_t)Result.ZoneResult[i].Distance[0];
        g_status[i]      = (uint8_t)Result.ZoneResult[i].Status[0];  // แปลงแล้ว 0 = ถูกต้อง
        g_signal[i]      = (uint32_t)Result.ZoneResult[i].Signal[0];
    }

#endif

    /* ===== ★ ตรวจว่าเฟรมนี้เป็นข้อมูลใหม่จริงหรือไม่ =====
       อ่านตัวนับของเซ็นเซอร์ที่ ULD เพิ่งอัปเดตให้ (vl53l8cx_api.c บรรทัด 754)
       การลบแบบ uint8_t จัดการการวนกลับที่ 255 -> 0 ให้เองโดยอัตโนมัติ */
    m_stream_prev  = m_stream;
    m_stream       = m_dev->streamcount;
    m_stream_delta = (uint8_t)(m_stream - m_stream_prev);

    if (g_frame_count > 1U)          /* ข้าม 2 เฟรมแรก เพราะ ULD ตั้งค่าเริ่มต้นเป็น 255 */
    {
        if (m_stream_delta == 0U)
        {
            m_dup++;                 /* อ่านข้อมูลเดิมซ้ำ */
        }
        else if (m_stream_delta > 1U)
        {
            m_skip += (uint32_t)(m_stream_delta - 1U);   /* มีเฟรมที่เราอ่านไม่ทัน */
        }
    }

    g_frame_count++;
    return 1;
}


/* =====================================================================
 *  my_test_delay_us : หน่วงเวลาละเอียด ใช้เฉพาะการทดลอง
 * ===================================================================== */
#if MY_TOF_DELAY_US > 0
static void my_test_delay_us(uint32_t us)
{
    uint32_t start = my_platform_cycles();
    uint32_t need  = us * (SystemCoreClock / 1000000U);
    while ((my_platform_cycles() - start) < need) { }
}
#endif


/* =====================================================================
 *  my_tof_send_frame : ส่งข้อมูลออก UART
 * ===================================================================== */
void my_tof_send_frame(void)
{
    uint32_t t0, t1, now, ms;

#if MY_TOF_TIMING_MODE

    t0 = my_platform_cycles();
    printf("T,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%u,%u,%lu,%lu\r\n",
           (unsigned long)g_frame_count, (unsigned long)m_rd_calls,
           (unsigned long)m_rd_bytes,    (unsigned long)m_rd_us,
           (unsigned long)m_max_bytes,   (unsigned long)m_max_us,
           (unsigned long)m_uart_us,
           (unsigned int)m_stream, (unsigned int)m_stream_delta,
           (unsigned long)m_dup,   (unsigned long)m_skip);
    t1 = my_platform_cycles();
    m_uart_us = my_platform_cycles_to_us(t1 - t0);

#else

    uint32_t i;
    t0 = my_platform_cycles();

    /* บรรทัดที่ 1 - ระยะทาง (mm) */
    printf("F,%lu", (unsigned long)g_frame_count);
    for (i = 0; i < MY_TOF_ZONES; i++) {
        printf(",%u", (unsigned int)g_distance_mm[i]);
    }
    printf("\r\n");

    /* บรรทัดที่ 2 - ค่าสถานะ (ความหมายดูจากบรรทัด init OK) */
    printf("S,%lu", (unsigned long)g_frame_count);
    for (i = 0; i < MY_TOF_ZONES; i++) {
        printf(",%u", (unsigned int)g_status[i]);
    }
    printf("\r\n");

    /* บรรทัดที่ 3 - ความแรงแสงสะท้อนต่อ SPAD (ช่องที่ 2 ของ input 8x8x2) */
    printf("G,%lu", (unsigned long)g_frame_count);
    for (i = 0; i < MY_TOF_ZONES; i++) {
        printf(",%lu", (unsigned long)g_signal[i]);
    }
    printf("\r\n");

    t1 = my_platform_cycles();
    m_uart_us = my_platform_cycles_to_us(t1 - t0);

#endif

    /* ===== วัดอัตราเฟรมจริง  R,<เฟรม>,<จำนวน>,<ms>,<dup>,<skip> ===== */
    m_rate_n++;
    if (m_rate_n >= MY_RATE_WINDOW)
    {
        now = HAL_GetTick();
        ms  = now - m_rate_t0;
        printf("R,%lu,%lu,%lu,%lu,%lu\r\n",
               (unsigned long)g_frame_count, (unsigned long)m_rate_n,
               (unsigned long)ms, (unsigned long)m_dup, (unsigned long)m_skip);
        m_rate_t0 = now;
        m_rate_n  = 0;
    }

#if MY_TOF_DELAY_US > 0
    my_test_delay_us(MY_TOF_DELAY_US);
#endif
}
