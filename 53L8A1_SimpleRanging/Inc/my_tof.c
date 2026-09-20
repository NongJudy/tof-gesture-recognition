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
#include "network.h"
#include "network_data.h"

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
#define MY_TOF_TIMING_MODE    0  // 1 = บรรทัด T | 0 = บรรทัด F + S + G

/* ===== สวิตช์อ่านตรง =====
   1 = เรียก vl53l8cx_get_ranging_data() ตรง ยิง I2C ครั้งเดียว
   0 = ผ่าน BSP ยิง I2C 4 ครั้ง (2 ครั้งซ้ำซ้อน: check_data_ready + get_resolution)
   ★ โหมด 1 ได้ target_status แบบดิบ (5 = ถูกต้อง) ตรงกับชุดข้อมูลของ ST */
#define MY_TOF_FAST_READ     1

/* ===== การทดลองหน่วงเวลา (จบแล้ว) =====
   ผล: คาบ = 23.24 + 0.981 x หน่วง   (R^2 = 0.9997, 6 จุด) */
#define MY_TOF_DELAY_US      0

/* ===== จำนวนที่ streamcount ควรเพิ่มต่อหนึ่งเฟรม =====
   streamcount ของเซ็นเซอร์นับ "การวัดภายใน" ไม่ใช่ "เฟรมที่ส่งออก"
   (vl53l8cx_api.h:276 ระบุว่า auto-incremented at each range)

   วัดจริง 10 ก.ย. 2026 พบว่า
     โหมด 4x4 : delta = 1 คงที่ทุกเฟรม
     โหมด 8x8 : delta = 4 คงที่ทุกเฟรม ไม่แกว่งเลย
   ถ้าเป็นเฟรมหายจริง ค่าต้องแกว่งไม่แน่นอน
   การที่คงที่เป๊ะแสดงว่าเป็นพฤติกรรมปกติของเซ็นเซอร์

   ก่อนแก้จุดนี้ โค้ดนับ skip เพิ่มทีละ 3 ทุกเฟรมในโหมด 8x8
   ทั้งที่ไม่มีเฟรมหายจริงแม้แต่เฟรมเดียว */
#if MY_TOF_USE_4X4
  #define MY_TOF_STREAM_STEP   1U
#else
  #define MY_TOF_STREAM_STEP   4U
#endif

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

/* ===== ★ ตัวตรวจสอบว่าทุกเฟรมเป็นข้อมูลใหม่จริง ★ */
static VL53L8CX_Configuration *m_dev = NULL;
static uint8_t  m_stream = 0, m_stream_prev = 0, m_stream_delta = 0;
static uint32_t m_dup = 0, m_skip = 0;
static uint32_t m_anomaly = 0;      /* delta ที่ไม่ใช่พหุคูณของ MY_TOF_STREAM_STEP */

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
    printf("H,frame,rd_calls,rd_bytes,rd_us,max_bytes,max_us,uart_us,"
           "stream,delta,dup,skip,anomaly\r\n");
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
        g_status[i]      = RawData.target_status[k];   // ค่าดิบ 5 = ถูกต้อง
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
static ai_float g_ai_in[AI_NETWORK_IN_1_SIZE];    /* 128 = 8*8*2 */
AI_ALIGNED(4)
static ai_float g_ai_out[AI_NETWORK_OUT_1_SIZE];  /* 7 คลาส */

static int8_t g_pred_class = -1;
static float  g_pred_conf  = 0.0f;

/* ชื่อคลาส เรียงตามลำดับ label ของ ST (st_dataset.py: HAND_POSTURE_DICT)
   FlatHand=20, Like=21, Love=24, Dislike=25, BreakTime=27, CrossHands=28, Fist=32 */
static const char *g_class_names[7] = {
    "FlatHand", "Like", "Love", "Dislike", "BreakTime", "CrossHands", "Fist"
};

/* ค่าคงที่ normalize เดียวกับ ST (data_loader.py:156-157 / st_dataset.py) */
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

/* =====================================================================
 *  my_ai_init : สร้าง + init เครือข่าย AI ครั้งเดียวตอนเริ่มโปรแกรม
 * ===================================================================== */
uint8_t my_ai_init(void)
{
    ai_handle act_addr[] = { g_activations };
    ai_handle wt_addr[]  = { ai_network_data_weights_get() };

    ai_error err = ai_network_create_and_init(&g_network, act_addr, wt_addr);
    if (err.type != AI_ERROR_NONE)
    {
        printf("ERROR: ai_network_create_and_init failed (type=%d code=%d)\r\n",
               (int)err.type, (int)err.code);
        return 1;
    }

    printf("MY_AI: network ready (in=%d floats, out=%d classes, "
           "activations=%d bytes)\r\n",
           AI_NETWORK_IN_1_SIZE, AI_NETWORK_OUT_1_SIZE,
           AI_NETWORK_DATA_ACTIVATIONS_SIZE);
    return 0;
}

/* =====================================================================
 *  my_tof_infer : preprocess เฟรมล่าสุด (g_distance_mm/g_status/g_signal)
 *  ตามสูตรเดียวกับ st_dataset.py แล้วเรียกโมเดล เก็บผลใน g_pred_class
 * ===================================================================== */
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
        g_pred_class = -1;   /* ไม่มีท่ามือในระยะที่สนใจ */
        g_pred_conf  = 0.0f;
        return;
    }

    for (i = 0; i < MY_TOF_ZONES; i++)
    {
        float dist_mm  = (float)g_distance_mm[i];
        float sig      = (float)g_signal[i];
        uint8_t zone_ok = valid[i] && (dist_mm <= hand_dist + MY_AI_BACKGROUND_MM);

        float dist_out = zone_ok ? dist_mm : MY_AI_DEFAULT_DISTANCE;
        float sig_out  = zone_ok ? sig     : MY_AI_DEFAULT_SIGNAL;

        g_ai_in[i * 2 + 0] = (dist_out - MY_AI_DIST_MEAN) / MY_AI_DIST_STD;
        g_ai_in[i * 2 + 1] = (sig_out  - MY_AI_SIG_MEAN)  / MY_AI_SIG_STD;
    }

    {
        ai_buffer *ai_input  = ai_network_inputs_get(g_network, NULL);
        ai_buffer *ai_output = ai_network_outputs_get(g_network, NULL);

        ai_input[0].data  = AI_HANDLE_PTR(g_ai_in);
        ai_output[0].data = AI_HANDLE_PTR(g_ai_out);

        ai_i32 batches = ai_network_run(g_network, ai_input, ai_output);
        ai_i32 batches = ai_network_run(g_network, ai_input, ai_output);
        printf("DBG: batches=%ld in0=%d in1=%d out0=%d out1=%d\r\n",
               (long)batches,
               (int)(g_ai_in[0] * 1000), (int)(g_ai_in[1] * 1000),
               (int)(g_ai_out[0] * 1000), (int)(g_ai_out[1] * 1000));
        if (batches != 1)
        {
            ai_error err = ai_network_get_error(g_network);
            printf("ERROR: ai_network_run failed (type=%d code=%d)\r\n",
                   (int)err.type, (int)err.code);
            g_pred_class = -1;
            g_pred_conf  = 0.0f;
            return;
        }
    }

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
    }
}

/* =====================================================================
 *  my_tof_send_prediction : ส่งผลทำนายล่าสุดออก UART
 * ===================================================================== */
void my_tof_send_prediction(void)
{
    if (g_pred_class < 0)
    {
        printf("P,%lu,None,0.000\r\n", (unsigned long)g_frame_count);
    }
    else
    {
        printf("P,%lu,%s,%.3f\r\n", (unsigned long)g_frame_count,
               g_class_names[g_pred_class], (double)g_pred_conf);
    }
}
