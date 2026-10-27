/**
 * my_tof.c
 * อ่านข้อมูลจากเซ็นเซอร์ ToF (VL53L8CX) เข้าหน่วยความจำ MCU แล้วส่งออก UART
 *
 * แก้ไข 26 ต.ค. 2026 (Phase 5.5b ขั้น 2.4 ข้อ 7): เพิ่มสวิตช์ MY_TOF_PRED_LINE
 *   = พิมพ์บรรทัด Y ทุกเฟรม (ผลทายของบอร์ด + ความน่าจะเป็น 7 ค่า) เพื่อให้ PC
 *     เทียบกับ E5 บนคอมพิวเตอร์ทีละเฟรม (pc_tools/compare_board_pc.py)
 *   ส่วนอื่นของไฟล์ไม่เปลี่ยน
 * แก้ไข 26 ต.ค. 2026 (Phase 5.6 แบบละเอียด): เพิ่มสวิตช์ MY_TOF_FRAME_TIMING
 *   = พิมพ์เวลาของ "ทุกเฟรมที่ทาย" เป็นบรรทัด Q (เดิมมีแค่ค่าเฉลี่ยทีละ 30 เฟรมในบรรทัด LAT)
 *     เพื่อหาค่าสูงสุดจริงและการกระจายของเวลาต่อเฟรม — การคำนวณและบรรทัดอื่นไม่เปลี่ยน
 * แก้ไข 27 ต.ค. 2026 (กติกา A17 ข้อ 5, ขั้น 5.4): เพิ่มสวิตช์ MY_TOF_RECENTRE
 *   = ย้ายมือมากลางภาพ (my_recentre.h) หลัง normalise ก่อนเข้าโมเดล — ต้องใช้คู่กับโมเดล E5R เท่านั้น
 *   + ตัวกันใส่โมเดลผิดคู่: ตอนบูตเทียบชื่อโมเดลจาก network.h (AI_NETWORK_ORIGIN_MODEL_NAME)
 *     ถ้าสวิตช์กับโมเดลไม่ตรงกัน บอร์ด "ไม่ทาย" (พิมพ์ Y,<เฟรม>,-3) แทนที่จะทายผิดแบบเงียบๆ
 *   ส่วนอื่นของไฟล์ไม่เปลี่ยน ; MY_TOF_RECENTRE = 0 + โมเดล E5 ทำงานเหมือนเฟิร์มแวร์ 26 ต.ค. ทุกอย่าง
 * แก้ไข 27 ต.ค. 2026 (กติกา A17 ข้อ 5, ขั้น 5.5): เพิ่มสวิตช์ MY_TOF_REPLAY
 *   = บอร์ด "ไม่อ่านเซ็นเซอร์" แต่รับเฟรมที่บันทึกไว้จาก PC ทาง UART (pc_tools/replay_board.py)
 *     แล้วประมวลผลด้วยโค้ดเดียวกับตอนใช้งานจริงทุกบรรทัด (send_frame -> infer -> send_prediction)
 *     ใช้ตรวจว่า "บอร์ดทายเหมือน PC ทุกเฟรม" บนข้อมูลชุดเดิม โดยไม่ต้องเก็บข้อมูลใหม่
 *   แพ็กเก็ต 454 ไบต์: A5 5A | seq u16 | ระยะ 64×u16 | status 64×u8 | signal 64×u32 | checksum u16
 *     (little-endian ; checksum = ผลรวมไบต์ seq..signal mod 65536)
 *   บอร์ดตอบ F/S/G (ทวนเฟรมที่ได้รับ ให้ PC ตรวจ) Y Q ... แล้วปิดท้ายด้วย K,<seq>
 *     checksum ผิด -> E,<seq>,<รับ>,<คำนวณ> และไม่ประมวลผล (PC ส่งซ้ำ)
 *   MY_TOF_REPLAY = 0 ทำงานเหมือนเดิมทุกอย่าง (อ่านเซ็นเซอร์)
 */

#include "my_tof.h"
#include <stdio.h>
#include "53l8a1_ranging_sensor.h"
#include "stm32f4xx_nucleo.h"
#include "app_tof_pin_conf.h"
#include "vl53l8cx.h"
#include "vl53l8cx_api.h"
#include <string.h>
#include "network.h"
#include "network_data.h"
#include "my_recentre.h"

extern uint8_t  my_platform_i2c_probe(uint16_t address);
extern void     my_platform_dwt_init(void);
extern uint32_t my_platform_cycles(void);
extern uint32_t my_platform_cycles_to_us(uint32_t cycles);
extern void     my_platform_stats_reset(void);
extern volatile uint32_t g_rd_calls, g_rd_bytes, g_rd_cycles;
extern volatile uint32_t g_rd_max_bytes, g_rd_max_cycles;
extern void my_uart_init(void);
extern volatile uint8_t ToF_EventDetected;
extern void *VL53L8A1_RANGING_SENSOR_CompObj[];

#define MY_TOF_USE_INT       1
#define MY_TOF_TIMING_MODE    0
#define MY_TOF_FAST_READ     1
#define MY_TOF_DELAY_US      0

/* ===== สวิตช์พิมพ์ผลทายทุกเฟรม (เพิ่ม 26 ต.ค. 2026) =====
   1 = ทุกเฟรมพิมพ์  Y,<เฟรม>,<ท่า>,<p0>,...,<p6>   (ท่า = 0..6 ตามลำดับ g_class_names)
       เฟรมที่ไม่ได้ทาย (มืออยู่นอก 100–400 มม.) พิมพ์  Y,<เฟรม>,-1
       เฟรมที่ ai_network_run ล้มเหลว               พิมพ์  Y,<เฟรม>,-2
   0 = ไม่พิมพ์ (ทำงานเหมือนเฟิร์มแวร์ก่อนหน้าทุกอย่าง)
   ใช้คู่กับ MY_TOF_TIMING_MODE = 0 เท่านั้น (ต้องมีบรรทัด F/S/G ของเฟรมเดียวกันให้ PC เทียบ)
   บรรทัด Y พิมพ์ "หลัง" จุดวัดเวลา t3 จึงไม่ทำให้ตัวเลข LAT เปลี่ยน */
#define MY_TOF_PRED_LINE     1

#if MY_TOF_PRED_LINE && MY_TOF_TIMING_MODE
  #error "MY_TOF_PRED_LINE ต้องใช้กับ MY_TOF_TIMING_MODE = 0 (โหมดส่งข้อมูล F/S/G)"
#endif

/* ===== สวิตช์พิมพ์เวลาทุกเฟรม (เพิ่ม 26 ต.ค. 2026) =====
   1 = ทุกเฟรมที่ทายสำเร็จพิมพ์  Q,<เฟรม>,<sensor_us>,<preprocess_us>,<inference_us>,<decision_us>,<total_us>
       (ตัวเลขชุดเดียวกับที่สะสมเข้าบรรทัด LAT แต่ไม่เฉลี่ย — ได้ค่าสูงสุดจริงของแต่ละเฟรม)
   0 = ไม่พิมพ์
   พิมพ์หลังจุดวัดเวลา t3 จึงไม่ทำให้ตัวเลขเวลาเปลี่ยน */
#define MY_TOF_FRAME_TIMING  1

#if MY_TOF_FRAME_TIMING && MY_TOF_TIMING_MODE
  #error "MY_TOF_FRAME_TIMING ต้องใช้กับ MY_TOF_TIMING_MODE = 0"
#endif

/* ===== สวิตช์ย้ายมือมากลางภาพ (เพิ่ม 27 ต.ค. 2026, PROJECT_FACTS §5.1w / §5.1y-S) =====
   1 = หลัง normalise ย้ายมือให้จุดศูนย์กลางอยู่กลางภาพ แบบเดียวกับ pc_tools/recentre.py ทุกบิต
       ใช้คู่กับโมเดล E5R เท่านั้น (ไฟล์ X-CUBE-AI/App ที่สร้างจาก E5R.tflite)
   0 = ไม่ย้าย ใช้คู่กับ E5 (โมเดลที่อยู่บนบอร์ดตั้งแต่ 26 ต.ค.)
   เวลาที่ใช้ย้าย นับรวมใน preprocess_us (บรรทัด LAT / Q) */
#ifndef MY_TOF_RECENTRE
#define MY_TOF_RECENTRE      1     /* ขั้น 5.6 (27 ต.ค.): 1 = คู่กับ E5R ใน X-CUBE-AI/App */
#endif

/* ชื่อโมเดลที่ต้องคู่กับสวิตช์ — X-CUBE-AI ตั้งจากชื่อไฟล์ .tflite เป็นตัวเล็ก (network.h) */
#define MY_TOF_RECENTRE_MODEL  "e5r"

/* ===== สวิตช์ replay (เพิ่ม 27 ต.ค. 2026, กติกา A17 ข้อ 5) =====
   1 = รับเฟรมจาก PC ทาง UART แทนการอ่านเซ็นเซอร์ (ดูหัวไฟล์) — ใช้ทดสอบเท่านั้น ห้ามใช้ตอนเดโม
   0 = อ่านเซ็นเซอร์ตามปกติ */
#ifndef MY_TOF_REPLAY
#define MY_TOF_REPLAY        0     /* 0 = อ่านเซ็นเซอร์ (ใช้งานจริง/เดโม/ลองสด) ; 1 = replay (ขั้น 5.5–5.6) */
#endif

#if MY_TOF_REPLAY && (MY_TOF_TIMING_MODE || !MY_TOF_PRED_LINE || MY_TOF_USE_4X4)
  #error "MY_TOF_REPLAY ต้องใช้กับ MY_TOF_TIMING_MODE = 0, MY_TOF_PRED_LINE = 1 และโหมด 8x8"
#endif

#if MY_TOF_REPLAY
extern void my_uart_rx_start(void);
extern int  my_uart_rx_get(uint8_t *b);
extern volatile uint32_t g_uart_rx_bytes, g_uart_rx_ovf, g_uart_rx_err;
#define MY_RP_SYNC0        (0xA5U)
#define MY_RP_SYNC1        (0x5AU)
#define MY_RP_PAYLOAD      (2U + 64U * 2U + 64U * 1U + 64U * 4U)   /* seq + ระยะ + status + signal = 450 */
#define MY_RP_FRAME_BYTES  (2U + MY_RP_PAYLOAD + 2U)               /* + sync 2 + checksum 2 = 454 */
#endif

#if MY_TOF_USE_4X4
  #define MY_TOF_STREAM_STEP   1U
#else
  #define MY_TOF_STREAM_STEP   4U
#endif

#if MY_TOF_USE_4X4
  #define MY_TIMING_BUDGET   (10U)
#else
  #define MY_TIMING_BUDGET   (30U)
#endif

#define MY_RATE_WINDOW       (60U)

static uint16_t g_distance_mm[MY_TOF_ZONES];
static uint8_t  g_status[MY_TOF_ZONES];
static uint32_t g_signal[MY_TOF_ZONES];
static uint32_t g_frame_count = 0;

static uint32_t m_rd_calls = 0, m_rd_bytes = 0, m_rd_us = 0;
static uint32_t m_max_bytes = 0, m_max_us = 0, m_uart_us = 0;
static uint32_t m_rate_t0 = 0, m_rate_n = 0;

static VL53L8CX_Configuration *m_dev = NULL;
static uint8_t  m_stream = 0, m_stream_prev = 0, m_stream_delta = 0;
static uint32_t m_dup = 0, m_skip = 0;
static uint32_t m_anomaly = 0;

#if !MY_TOF_REPLAY                  /* replay ไม่อ่านเซ็นเซอร์ จึงไม่ต้องมีบัฟเฟอร์ผลวัด */
#if MY_TOF_FAST_READ
static VL53L8CX_ResultsData    RawData;
#else
static RANGING_SENSOR_Result_t Result;
#endif
#endif


uint8_t my_tof_init(void)
{
    int32_t status;
    RANGING_SENSOR_ProfileConfig_t Profile;

    BSP_COM_Init(COM1);
    HAL_Delay(100);

    my_uart_init();
    my_platform_dwt_init();

    HAL_GPIO_WritePin(VL53L8A1_PWR_EN_C_PORT, VL53L8A1_PWR_EN_C_PIN, GPIO_PIN_RESET);
    HAL_Delay(2);
    HAL_GPIO_WritePin(VL53L8A1_PWR_EN_C_PORT, VL53L8A1_PWR_EN_C_PIN, GPIO_PIN_SET);
    HAL_Delay(2);

    status = VL53L8A1_RANGING_SENSOR_Init(VL53L8A1_DEV_CENTER);
    if (status != BSP_ERROR_NONE)
    {
        printf("ERROR: sensor init failed (%ld)\r\n", (long)status);
        return 1;
    }

    if (my_platform_i2c_probe(0x52) == 0) {
        printf("T1 PASS: sensor ACK at 0x52\r\n");
    } else {
        printf("T1 FAIL: no ACK at 0x52\r\n");
    }

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

#if MY_TOF_USE_4X4
    Profile.RangingProfile = RS_PROFILE_4x4_CONTINUOUS;
#else
    Profile.RangingProfile = RS_PROFILE_8x8_CONTINUOUS;
#endif
    Profile.TimingBudget   = MY_TIMING_BUDGET;
    Profile.Frequency      = MY_TOF_FREQ_HZ;
    Profile.EnableAmbient  = 0;
    Profile.EnableSignal   = 1;

    VL53L8A1_RANGING_SENSOR_ConfigProfile(VL53L8A1_DEV_CENTER, &Profile);

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

#if MY_TOF_REPLAY
    my_uart_rx_start();
    printf("MY_TOF: REPLAY mode - frames come from the PC over UART, the sensor is ignored "
           "(packet %u bytes)\r\n", (unsigned int)MY_RP_FRAME_BYTES);
#endif

#if MY_TOF_TIMING_MODE
    printf("H,frame,rd_calls,rd_bytes,rd_us,max_bytes,max_us,uart_us,"
           "stream,delta,dup,skip,anomaly\r\n");
#endif

    m_rate_t0 = HAL_GetTick();
    return 0;
}


#if MY_TOF_REPLAY
/* ===== โหมด replay: ประกอบแพ็กเก็ตจากไบต์ที่รับทาง UART =====
   คืน 1 เมื่อได้เฟรมครบและ checksum ถูก (ใส่ลง g_distance_mm / g_status / g_signal แบบเดียวกับเซ็นเซอร์)
   ก่อนรับเฟรมถัดไป พิมพ์ K,<seq> ของเฟรมก่อนหน้า (= ประมวลผลและพิมพ์ทุกบรรทัดเสร็จแล้ว) */
static uint8_t  g_rp_buf[MY_RP_PAYLOAD + 2U];   /* payload + checksum */
static uint16_t g_rp_n = 0;
static uint8_t  g_rp_state = 0;                 /* 0 = รอ A5, 1 = รอ 5A, 2 = รับ payload */
static uint8_t  g_rp_ack = 0;
static uint16_t g_rp_seq = 0;
static uint8_t  g_rp_ready = 0;

static uint32_t rp_u16(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }
static uint32_t rp_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint8_t my_tof_read_frame(void)
{
    uint8_t b;
    uint32_t i;

    (void)m_stream; (void)m_stream_prev; (void)m_stream_delta;   /* ใช้เฉพาะตอนอ่านเซ็นเซอร์ */
    (void)ToF_EventDetected;
    if (!g_rp_ready)
    {
        printf("REPLAY,ready,%u\r\n", (unsigned int)MY_RP_FRAME_BYTES);
        g_rp_ready = 1;
    }
    if (g_rp_ack)
    {
        printf("K,%u,%lu,%lu,%lu\r\n", (unsigned int)g_rp_seq, (unsigned long)g_uart_rx_bytes,
               (unsigned long)g_uart_rx_ovf, (unsigned long)g_uart_rx_err);
        g_rp_ack = 0;
    }

    while (my_uart_rx_get(&b))
    {
        if (g_rp_state == 0U)
        {
            g_rp_state = (b == MY_RP_SYNC0) ? 1U : 0U;
        }
        else if (g_rp_state == 1U)
        {
            g_rp_state = (b == MY_RP_SYNC1) ? 2U : ((b == MY_RP_SYNC0) ? 1U : 0U);
            g_rp_n = 0;
        }
        else
        {
            g_rp_buf[g_rp_n++] = b;
            if (g_rp_n >= MY_RP_PAYLOAD + 2U)
            {
                uint32_t sum = 0;
                uint32_t got = rp_u16(&g_rp_buf[MY_RP_PAYLOAD]);
                const uint8_t *p = g_rp_buf;
                g_rp_state = 0;
                for (i = 0; i < MY_RP_PAYLOAD; i++)
                {
                    sum += g_rp_buf[i];
                }
                sum &= 0xFFFFU;
                if (sum != got)
                {
                    printf("E,%lu,%lu,%lu\r\n", (unsigned long)rp_u16(g_rp_buf),
                           (unsigned long)got, (unsigned long)sum);
                    continue;                      /* ทิ้งแพ็กเก็ตนี้ PC จะส่งซ้ำ */
                }
                g_rp_seq = (uint16_t)rp_u16(p);
                p += 2;
                for (i = 0; i < MY_TOF_ZONES; i++) { g_distance_mm[i] = (uint16_t)rp_u16(p + 2U * i); }
                p += 2U * MY_TOF_ZONES;
                for (i = 0; i < MY_TOF_ZONES; i++) { g_status[i] = p[i]; }
                p += MY_TOF_ZONES;
                for (i = 0; i < MY_TOF_ZONES; i++) { g_signal[i] = rp_u32(p + 4U * i); }

                m_rd_calls = 0; m_rd_bytes = 0; m_rd_us = 0;   /* ไม่ได้อ่านเซ็นเซอร์ -> เวลาอ่าน = 0 */
                m_max_bytes = 0; m_max_us = 0;
                g_frame_count = g_rp_seq;                     /* บรรทัด F/S/G/Y/Q ใช้เลข seq ของ PC */
                g_rp_ack = 1;
                return 1;
            }
        }
    }
    return 0;
}

#else  /* MY_TOF_REPLAY == 0 : อ่านเซ็นเซอร์ตามปกติ */

uint8_t my_tof_read_frame(void)
{
    uint32_t i;

#if MY_TOF_USE_INT
    if (ToF_EventDetected == 0U)
    {
        return 0;
    }
    ToF_EventDetected = 0;
#endif

    my_platform_stats_reset();

#if MY_TOF_FAST_READ

    if (vl53l8cx_get_ranging_data(m_dev, &RawData) != VL53L8CX_STATUS_OK)
    {
        return 0;
    }

    m_rd_calls  = g_rd_calls;
    m_rd_bytes  = g_rd_bytes;
    m_rd_us     = my_platform_cycles_to_us(g_rd_cycles);
    m_max_bytes = g_rd_max_bytes;
    m_max_us    = my_platform_cycles_to_us(g_rd_max_cycles);

    for (i = 0; i < MY_TOF_ZONES; i++)
    {
        uint32_t k = (uint32_t)VL53L8CX_NB_TARGET_PER_ZONE * i;

        g_distance_mm[i] = (uint16_t)RawData.distance_mm[k];
        g_status[i]      = RawData.target_status[k];
        g_signal[i]      = RawData.signal_per_spad[k];
    }

#else

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
        g_status[i]      = (uint8_t)Result.ZoneResult[i].Status[0];
        g_signal[i]      = (uint32_t)Result.ZoneResult[i].Signal[0];
    }

#endif

    m_stream_prev  = m_stream;
    m_stream       = m_dev->streamcount;
    m_stream_delta = (uint8_t)(m_stream - m_stream_prev);

    if (g_frame_count > 1U)
    {
        if (m_stream_delta == 0U)
        {
            m_dup++;
        }
        else if ((m_stream_delta % MY_TOF_STREAM_STEP) != 0U)
        {
            m_anomaly++;
        }
        else if (m_stream_delta > MY_TOF_STREAM_STEP)
        {
            m_skip += (uint32_t)(m_stream_delta / MY_TOF_STREAM_STEP) - 1U;
        }
    }

    g_frame_count++;
    return 1;
}

#endif /* MY_TOF_REPLAY */


#if MY_TOF_DELAY_US > 0
static void my_test_delay_us(uint32_t us)
{
    uint32_t start = my_platform_cycles();
    uint32_t need  = us * (SystemCoreClock / 1000000U);
    while ((my_platform_cycles() - start) < need) { }
}
#endif


void my_tof_send_frame(void)
{
    uint32_t t0, t1, now, ms;

#if MY_TOF_TIMING_MODE

    t0 = my_platform_cycles();
    printf("T,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%u,%u,%lu,%lu,%lu\r\n",
           (unsigned long)g_frame_count, (unsigned long)m_rd_calls,
           (unsigned long)m_rd_bytes,    (unsigned long)m_rd_us,
           (unsigned long)m_max_bytes,   (unsigned long)m_max_us,
           (unsigned long)m_uart_us,
           (unsigned int)m_stream, (unsigned int)m_stream_delta,
           (unsigned long)m_dup,   (unsigned long)m_skip,
           (unsigned long)m_anomaly);
    t1 = my_platform_cycles();
    m_uart_us = my_platform_cycles_to_us(t1 - t0);

#else

    uint32_t i;
    t0 = my_platform_cycles();

    printf("F,%lu", (unsigned long)g_frame_count);
    for (i = 0; i < MY_TOF_ZONES; i++) {
        printf(",%u", (unsigned int)g_distance_mm[i]);
    }
    printf("\r\n");

    printf("S,%lu", (unsigned long)g_frame_count);
    for (i = 0; i < MY_TOF_ZONES; i++) {
        printf(",%u", (unsigned int)g_status[i]);
    }
    printf("\r\n");

    printf("G,%lu", (unsigned long)g_frame_count);
    for (i = 0; i < MY_TOF_ZONES; i++) {
        printf(",%lu", (unsigned long)g_signal[i]);
    }
    printf("\r\n");

    t1 = my_platform_cycles();
    m_uart_us = my_platform_cycles_to_us(t1 - t0);

#endif

    m_rate_n++;
    if (m_rate_n >= MY_RATE_WINDOW)
    {
        now = HAL_GetTick();
        ms  = now - m_rate_t0;
        printf("R,%lu,%lu,%lu,%lu,%lu,%lu\r\n",
               (unsigned long)g_frame_count, (unsigned long)m_rate_n,
               (unsigned long)ms, (unsigned long)m_dup, (unsigned long)m_skip,
               (unsigned long)m_anomaly);
        m_rate_t0 = now;
        m_rate_n  = 0;
    }

#if MY_TOF_DELAY_US > 0
    my_test_delay_us(MY_TOF_DELAY_US);
#endif
}


/* ============================================================
 *  ส่วนต่อ AI inference — Phase 3
 * ============================================================ */

static ai_handle g_network = AI_HANDLE_NULL;

AI_ALIGNED(4)
static ai_u8 g_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

AI_ALIGNED(4)
static ai_float g_ai_in[AI_NETWORK_IN_1_SIZE];
AI_ALIGNED(4)
static ai_float g_ai_out[AI_NETWORK_OUT_1_SIZE];

static int8_t g_pred_class = -1;
static float  g_pred_conf  = 0.0f;

/* 1 = สวิตช์ MY_TOF_RECENTRE กับโมเดลใน network.h เข้าคู่กัน (ตรวจใน my_ai_init) */
static uint8_t g_model_ok = 0;

#if MY_TOF_RECENTRE
AI_ALIGNED(4)
static ai_float g_rc_tmp[AI_NETWORK_IN_1_SIZE];   /* บัฟเฟอร์ชั่วคราวตอนย้ายมือ */
#endif

/* ★ ลำดับนี้ตรงกับ CLASSES ใน train_st_cnn2d.py (สคริปต์ที่เทรน production_model.keras
   จริง) — ยืนยันแล้ว 21 ก.ย. 2026 ห้ามสลับกลับไปใช้ลำดับ sort ตามเลข label ของ ST */
static const char *g_class_names[7] = {
    "Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime"
};

#define MY_AI_DIST_MEAN         (295.0f)
#define MY_AI_DIST_STD          (196.0f)
#define MY_AI_SIG_MEAN          (281.0f)
#define MY_AI_SIG_STD           (452.0f)
#define MY_AI_DEFAULT_DISTANCE  (4000.0f)
#define MY_AI_DEFAULT_SIGNAL    (0.0f)
#define MY_AI_MAX_DISTANCE_MM   (400.0f)
#define MY_AI_MIN_DISTANCE_MM   (100.0f)
#define MY_AI_BACKGROUND_MM     (120.0f)
#define MY_AI_VALID_STATUS_1    (5U)
#define MY_AI_VALID_STATUS_2    (9U)

#define MY_AI_VOTE_WINDOW  (15)
static int8_t  g_vote_buf[MY_AI_VOTE_WINDOW];
static uint8_t g_vote_idx = 0;
static uint8_t g_vote_filled_count = 0;
static int8_t  g_last_stable = -2;

/* ===== ★ Phase 3, ส่วนสุดท้าย: วัด 4-stage latency (DWT, เหมือน Phase 1) =====
   วัดเฉพาะตอน inference รันจริง (มีมือในระยะ) สะสมแล้วพิมพ์ค่าเฉลี่ยทุก
   MY_LAT_WINDOW ครั้ง กัน UART ท่วมตอนรันสดๆ */
#define MY_LAT_WINDOW (30U)
static uint32_t g_lat_count = 0;
static uint32_t g_lat_sum_sensor_us = 0;
static uint32_t g_lat_sum_pre_us    = 0;
static uint32_t g_lat_sum_inf_us    = 0;
static uint32_t g_lat_sum_dec_us    = 0;

uint8_t my_ai_init(void)
{
    ai_handle act_addr[] = { g_activations };

    ai_error err = ai_network_create_and_init(&g_network, act_addr, NULL);
    if (err.type != AI_ERROR_NONE)
    {
        printf("ERROR: ai_network_create_and_init failed (type=%d code=%d)\r\n",
               (int)err.type, (int)err.code);
        return 1;
    }

    printf("MY_AI: network ready (in=%d floats, out=%d classes, "
           "activations=%d bytes, vote_window=%d, lat_window=%d, pred_line=%d, "
           "frame_timing=%d)\r\n",
           AI_NETWORK_IN_1_SIZE, AI_NETWORK_OUT_1_SIZE,
           AI_NETWORK_DATA_ACTIVATIONS_SIZE, MY_AI_VOTE_WINDOW, MY_LAT_WINDOW,
           MY_TOF_PRED_LINE, MY_TOF_FRAME_TIMING);

    /* ตัวกันใส่โมเดลผิดคู่: E5R ต้องย้ายมือ (สวิตช์ = 1) ; โมเดลอื่น (E5) ต้องไม่ย้าย (สวิตช์ = 0) */
    {
        uint8_t is_e5r = (strcmp(AI_NETWORK_ORIGIN_MODEL_NAME, MY_TOF_RECENTRE_MODEL) == 0) ? 1U : 0U;
        g_model_ok = (is_e5r == (uint8_t)MY_TOF_RECENTRE) ? 1U : 0U;
        printf("MY_AI: model=%s recentre=%d check=%s\r\n", AI_NETWORK_ORIGIN_MODEL_NAME,
               MY_TOF_RECENTRE, g_model_ok ? "OK" : "MISMATCH");
        if (!g_model_ok)
        {
            printf("ERROR: model '%s' needs MY_TOF_RECENTRE=%d - inference disabled (Y,<frame>,-3)\r\n",
                   AI_NETWORK_ORIGIN_MODEL_NAME, (int)is_e5r);
            return 1;
        }
    }
    return 0;
}

void my_tof_infer(void)
{
    uint32_t i;
    float hand_dist;
    uint8_t valid[MY_TOF_ZONES];

    for (i = 0; i < MY_TOF_ZONES; i++)
    {
        valid[i] = (g_status[i] == MY_AI_VALID_STATUS_1 ||
                    g_status[i] == MY_AI_VALID_STATUS_2) ? 1U : 0U;
    }

    hand_dist = MY_AI_DEFAULT_DISTANCE;
    for (i = 0; i < MY_TOF_ZONES; i++)
    {
        if (valid[i] && (float)g_distance_mm[i] < hand_dist)
        {
            hand_dist = (float)g_distance_mm[i];
        }
    }

    if (hand_dist < MY_AI_MIN_DISTANCE_MM || hand_dist > MY_AI_MAX_DISTANCE_MM)
    {
        g_pred_class = -1;
        g_pred_conf  = 0.0f;
#if MY_TOF_PRED_LINE
        /* ไม่ได้ทาย: ให้ PC ตรวจว่า Python ก็ตัดสินว่า "นอกช่วง" ตรงกัน */
        printf("Y,%lu,-1\r\n", (unsigned long)g_frame_count);
#endif
    }
    else if (!g_model_ok)
    {
        /* สวิตช์กับโมเดลไม่เข้าคู่ (ดูบรรทัด MY_AI: ตอนบูต) → ไม่ทาย ดีกว่าทายผิดแบบเงียบๆ */
        g_pred_class = -1;
        g_pred_conf  = 0.0f;
#if MY_TOF_PRED_LINE
        printf("Y,%lu,-3\r\n", (unsigned long)g_frame_count);
#endif
    }
    else
    {
        uint32_t t0, t1, t2, t3;
#if MY_TOF_RECENTRE
        uint8_t hand[MY_TOF_ZONES];               /* โซนมือ = ระยะหลังตัดฉากหลัง < 2000 มม. */
        int32_t rc_dr, rc_dc;
#endif

        t0 = my_platform_cycles();

        for (i = 0; i < MY_TOF_ZONES; i++)
        {
            float dist_mm  = (float)g_distance_mm[i];
            float sig      = (float)g_signal[i];
            uint8_t zone_ok = valid[i] && (dist_mm <= hand_dist + MY_AI_BACKGROUND_MM);

            float dist_out = zone_ok ? dist_mm : MY_AI_DEFAULT_DISTANCE;
            float sig_out  = zone_ok ? sig     : MY_AI_DEFAULT_SIGNAL;

            g_ai_in[i * 2 + 0] = (dist_out - MY_AI_DIST_MEAN) / MY_AI_DIST_STD;
            g_ai_in[i * 2 + 1] = (sig_out  - MY_AI_SIG_MEAN)  / MY_AI_SIG_STD;
#if MY_TOF_RECENTRE
            hand[i] = (dist_out < 2000.0f) ? 1U : 0U;   /* = recentre.py: X[...,0] < (2000-295)/196 */
#endif
        }

#if MY_TOF_RECENTRE
        /* ย้ายมือมากลางภาพ — ค่าว่างคำนวณแบบเดียวกับโซนที่ไม่ใช่มือข้างบน (ระยะ 4000, signal 0) */
        (void)my_recentre(g_ai_in, hand,
                          (MY_AI_DEFAULT_DISTANCE - MY_AI_DIST_MEAN) / MY_AI_DIST_STD,
                          (MY_AI_DEFAULT_SIGNAL - MY_AI_SIG_MEAN) / MY_AI_SIG_STD,
                          g_rc_tmp, &rc_dr, &rc_dc);
        (void)rc_dr;
        (void)rc_dc;
#endif

        t1 = my_platform_cycles();   /* จบ preprocess */

        {
            ai_buffer *ai_input  = ai_network_inputs_get(g_network, NULL);
            ai_buffer *ai_output = ai_network_outputs_get(g_network, NULL);

            ai_input[0].data  = AI_HANDLE_PTR(g_ai_in);
            ai_output[0].data = AI_HANDLE_PTR(g_ai_out);

            ai_i32 batches = ai_network_run(g_network, ai_input, ai_output);

            t2 = my_platform_cycles();   /* จบ inference */

            if (batches != 1)
            {
                ai_error err = ai_network_get_error(g_network);
                printf("ERROR: ai_network_run failed (type=%d code=%d)\r\n",
                       (int)err.type, (int)err.code);
                g_pred_class = -1;
                g_pred_conf  = 0.0f;
#if MY_TOF_PRED_LINE
                printf("Y,%lu,-2\r\n", (unsigned long)g_frame_count);
#endif
            }
            else
            {
                int8_t best_idx = 0;
                float  best_val = g_ai_out[0];
                for (i = 1; i < AI_NETWORK_OUT_1_SIZE; i++)
                {
                    if (g_ai_out[i] > best_val)
                    {
                        best_val = g_ai_out[i];
                        best_idx = (int8_t)i;
                    }
                }
                g_pred_class = best_idx;
                g_pred_conf  = best_val;

                t3 = my_platform_cycles();   /* จบ decision (argmax) */

                /* เก็บสถิติ latency เฉพาะตอน inference สำเร็จจริง */
                uint32_t pre_us = my_platform_cycles_to_us(t1 - t0);
                uint32_t inf_us = my_platform_cycles_to_us(t2 - t1);
                uint32_t dec_us = my_platform_cycles_to_us(t3 - t2);
                g_lat_sum_sensor_us += m_rd_us;
                g_lat_sum_pre_us    += pre_us;
                g_lat_sum_inf_us    += inf_us;
                g_lat_sum_dec_us    += dec_us;
                g_lat_count++;

                if (g_lat_count >= MY_LAT_WINDOW)
                {
                    uint32_t sensor_avg = g_lat_sum_sensor_us / g_lat_count;
                    uint32_t pre_avg    = g_lat_sum_pre_us    / g_lat_count;
                    uint32_t inf_avg    = g_lat_sum_inf_us    / g_lat_count;
                    uint32_t dec_avg    = g_lat_sum_dec_us    / g_lat_count;
                    uint32_t total_avg  = sensor_avg + pre_avg + inf_avg + dec_avg;

                    printf("LAT,%lu,n=%lu,sensor_us=%lu,preprocess_us=%lu,"
                           "inference_us=%lu,decision_us=%lu,total_us=%lu\r\n",
                           (unsigned long)g_frame_count, (unsigned long)g_lat_count,
                           (unsigned long)sensor_avg, (unsigned long)pre_avg,
                           (unsigned long)inf_avg, (unsigned long)dec_avg,
                           (unsigned long)total_avg);

                    g_lat_count = 0;
                    g_lat_sum_sensor_us = 0;
                    g_lat_sum_pre_us    = 0;
                    g_lat_sum_inf_us    = 0;
                    g_lat_sum_dec_us    = 0;
                }

#if MY_TOF_PRED_LINE
                /* ผลทายของเฟรมนี้ ใช้เลขเฟรมเดียวกับบรรทัด F/S/G ที่ my_tof_send_frame()
                   เพิ่งพิมพ์ (main.c: read_frame -> send_frame -> infer)
                   %.6f พิมพ์ได้เพราะลิงก์ด้วย -u _printf_float (เปิดไว้ตั้งแต่ Phase 3) */
                printf("Y,%lu,%d", (unsigned long)g_frame_count, (int)best_idx);
                for (i = 0; i < AI_NETWORK_OUT_1_SIZE; i++)
                {
                    printf(",%.6f", (double)g_ai_out[i]);
                }
                printf("\r\n");
#endif

#if MY_TOF_FRAME_TIMING
                /* เวลาของเฟรมนี้เฟรมเดียว (ไม่เฉลี่ย) — ตัวเลขชุดเดียวกับที่บวกเข้า LAT ข้างบน */
                printf("Q,%lu,%lu,%lu,%lu,%lu,%lu\r\n", (unsigned long)g_frame_count,
                       (unsigned long)m_rd_us, (unsigned long)pre_us, (unsigned long)inf_us,
                       (unsigned long)dec_us,
                       (unsigned long)(m_rd_us + pre_us + inf_us + dec_us));
#endif
            }
        }
    }

    g_vote_buf[g_vote_idx] = g_pred_class;
    g_vote_idx = (uint8_t)((g_vote_idx + 1) % MY_AI_VOTE_WINDOW);
    if (g_vote_filled_count < MY_AI_VOTE_WINDOW)
    {
        g_vote_filled_count++;
    }
}

void my_tof_send_prediction(void)
{
    int16_t votes[7] = { 0 };
    uint8_t none_votes = 0;
    uint8_t i;
    int8_t  best;
    int16_t best_count;

    if (g_vote_filled_count < MY_AI_VOTE_WINDOW)
    {
        return;
    }

    for (i = 0; i < MY_AI_VOTE_WINDOW; i++)
    {
        int8_t c = g_vote_buf[i];
        if (c < 0) { none_votes++; }
        else       { votes[c]++; }
    }

    best       = -1;
    best_count = (int16_t)none_votes;
    for (i = 0; i < 7; i++)
    {
        if (votes[i] > best_count)
        {
            best_count = votes[i];
            best = (int8_t)i;
        }
    }

    if (best != g_last_stable)
    {
        g_last_stable = best;

        printf("VOTE,%lu,none=%u,Fist=%d,FlatHand=%d,Dislike=%d,Like=%d,"
               "Love=%d,CrossHands=%d,BreakTime=%d\r\n",
               (unsigned long)g_frame_count, none_votes,
               votes[0], votes[1], votes[2], votes[3],
               votes[4], votes[5], votes[6]);

        if (best < 0)
        {
            printf("P,%lu,None\r\n", (unsigned long)g_frame_count);
        }
        else
        {
            printf("P,%lu,%s\r\n", (unsigned long)g_frame_count,
                   g_class_names[best]);
        }
    }
}
