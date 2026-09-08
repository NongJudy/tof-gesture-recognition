/**
 * @file    my_calib.c
 * @brief   สร้างคลื่นสี่เหลี่ยมด้วย TIM3 ช่อง 3 ออกทางขา PC8
 *
 * ดูเหตุผลของการมีไฟล์นี้ทั้งหมดใน my_calib.h
 *
 * ทำไมเขียนรีจิสเตอร์ตรง ไม่ใช้ HAL
 * ---------------------------------
 * โปรเจคนี้ปิดโมดูล TIM ของ HAL ไว้ เปิดเฉพาะ GPIO, I2C, UART
 * ยืนยันจากข้อความคอมไพเลอร์ 8 ก.ย. 2026
 *     unknown type name TIM_HandleTypeDef
 *
 * ทางเลือกมีสองทาง
 *   1. เปิดโมดูล TIM เพิ่ม ต้องแก้ stm32f4xx_hal_conf.h และเพิ่มไฟล์ไดรเวอร์
 *      เสี่ยงกระทบส่วนที่ทำงานได้อยู่แล้ว และเพิ่มขนาดโปรแกรม
 *   2. เขียนรีจิสเตอร์ของ TIM3 ตรง ใช้แค่นิยามรีจิสเตอร์จาก CMSIS
 *      ซึ่งมีอยู่เสมอ ไม่ขึ้นกับว่าเปิดโมดูล HAL ตัวไหน
 *
 * เลือกทางที่สอง เพราะไม่แตะการตั้งค่าเดิมของโปรเจคเลยแม้แต่บรรทัดเดียว
 * และตรงกับแนวทางที่ใช้มาตลอด คือเขียนชั้นล่างเอง ไม่พึ่งโค้ดตัวอย่าง
 *
 * ผลพลอยได้ ตอนสอบอธิบายได้ว่าตั้งค่าตัวจับเวลาอย่างไร ทีละรีจิสเตอร์
 */

#include "my_calib.h"

#include "stm32f4xx_hal.h"

#include <stdio.h>

/* ==== ค่าคงที่ของฮาร์ดแวร์ ==== */

/* TIM3 บน STM32F4 เป็นตัวนับ 16 บิต ค่า ARR สูงสุดจึงเท่ากับ 65535
 * ถ้าใส่เกิน ค่าจะถูกตัดทิ้งเงียบ แล้วได้ความถี่ผิดโดยไม่มีคำเตือน
 */
#define MY_CAL_ARR_MAX 65535U

/* PC8 ทำหน้าที่ TIM3_CH3 ได้ผ่าน alternate function หมายเลข 2 */
#define MY_CAL_GPIO_PORT GPIOC
#define MY_CAL_GPIO_PIN  GPIO_PIN_8
#define MY_CAL_GPIO_AF   GPIO_AF2_TIM3

/* ค่าที่ต้องเขียนลงรีจิสเตอร์ CCMR2 สำหรับช่อง 3
 *   บิต 1:0  CC3S = 00   กำหนดให้ช่องนี้เป็นเอาต์พุต
 *   บิต 6:4  OC3M = 011  โหมด toggle ขาสลับสถานะเมื่อตัวนับตรงกับค่าเปรียบเทียบ
 * ที่มา RM0383 หัวข้อ TIMx capture compare mode register 2
 */
#define MY_CAL_CCMR2_TOGGLE_CH3 (3U << 4)

/* บิต 8 ของ CCER คือ CC3E เปิดสัญญาณออกที่ขาของช่อง 3
 * บิต 9 คือ CC3P ปล่อยเป็น 0 หมายถึง active high
 */
#define MY_CAL_CCER_ENABLE_CH3 (1U << 8)

static bool s_running = false;

/* หมายเหตุเรื่องการส่งข้อความออก UART
 *
 * โปรเจคนี้มีฟังก์ชัน _write อยู่แล้วใน my_uart.c บรรทัด 130
 * printf จึงส่งออกทาง USART2 ให้เองโดยอัตโนมัติ
 *
 * ห้ามประกาศ extern UART_HandleTypeDef huart2 ที่นี่
 * เพราะในโปรเจคนี้ huart2 ไม่ใช่ตัวแปร แต่เป็น macro จาก stm32f4xx_nucleo.h
 * ซึ่งขยายเป็น hcom_uart ตามที่ my_uart.c บรรทัด 21 ระบุไว้
 * การประกาศซ้ำจะทำให้คอมไพล์ไม่ผ่าน
 */

/**
 * @brief  หาความถี่จริงที่เข้าตัวนับ TIM3
 *
 * ทำไมไม่ฝังเลข 84000000 ลงไปตรง
 *
 * ตัวจับเวลาบนบัส APB1 ไม่ได้รับสัญญาณนาฬิกาเท่ากับ PCLK1 เสมอไป
 * ถ้าตัวหารของ APB1 มากกว่าหนึ่ง ฮาร์ดแวร์จะคูณสองให้กับตัวจับเวลา
 * อ้างอิง RM0383 หัวข้อ Clock tree
 *
 * โปรเจคนี้ตั้ง APB1CLKDivider เป็น RCC_HCLK_DIV2 ไว้ใน main.c
 * จึงเข้าเงื่อนไขคูณสอง PCLK1 ได้ 42 MHz ตัวจับเวลาจึงได้ 84 MHz
 *
 * การอ่านค่าจริงจากรีจิสเตอร์แทนการฝังตัวเลข ทำให้โค้ดยังถูกต้อง
 * แม้วันหน้าเปลี่ยนการตั้งนาฬิกา และตรงกับบทเรียนจาก TRAP หมายเลข 12
 * ที่ว่าต้องระบุนาฬิกาอ้างอิงเสมอ ห้ามสมมติ
 *
 * @return ความถี่ที่เข้า TIM3 หน่วย Hz
 */
static uint32_t calib_timer_clock_hz(void)
{
    const uint32_t pclk1 = HAL_RCC_GetPCLK1Freq();
    const uint32_t ppre1 = (RCC->CFGR & RCC_CFGR_PPRE1) >> RCC_CFGR_PPRE1_Pos;

    /* ค่า 0 ถึง 3 หมายถึงหารด้วยหนึ่ง ส่วน 4 ถึง 7 หมายถึงหารมากกว่าหนึ่ง
     * กรณีหลังฮาร์ดแวร์จะคูณสองให้ตัวจับเวลา
     */
    return (ppre1 < 4U) ? pclk1 : (pclk1 * 2U);
}

/**
 * @brief  หาค่า PSC และ ARR ที่ให้ความถี่ใกล้เป้าหมายที่สุด
 *
 * โหมด toggle สลับสถานะขาหนึ่งครั้งต่อหนึ่งคาบของตัวนับ
 * คลื่นที่ออกมาจึงมีความถี่เป็นครึ่งหนึ่งของอัตราการนับจบรอบ
 * ต้องตั้งให้ตัวนับจบรอบที่สองเท่าของความถี่ที่ต้องการ
 *
 * เลือกโหมด toggle แทน PWM เพราะได้รอบการทำงานห้าสิบเปอร์เซ็นต์พอดีเสมอ
 * ไม่ขึ้นกับว่าค่า ARR เป็นเลขคู่หรือคี่
 *
 * @param[in]  timer_clk_hz  นาฬิกาที่เข้าตัวจับเวลา
 * @param[in]  target_hz     ความถี่คลื่นที่ต้องการ
 * @param[out] psc           ค่า prescaler ที่ได้
 * @param[out] arr           ค่า auto reload ที่ได้
 * @return true ถ้าหาค่าที่ใช้ได้จริง
 */
static bool calib_solve_dividers(uint32_t timer_clk_hz,
                                 uint32_t target_hz,
                                 uint32_t *psc,
                                 uint32_t *arr)
{
    if (target_hz == 0U || timer_clk_hz == 0U || psc == NULL || arr == NULL) {
        return false;
    }

    const uint32_t toggle_rate = 2U * target_hz;

    for (uint32_t p = 0U; p <= 0xFFFFU; ++p) {
        const uint32_t counts = timer_clk_hz / ((p + 1U) * toggle_rate);

        if (counts == 0U) {
            return false;   /* ความถี่สูงเกินกว่าที่ตัวจับเวลาจะสร้างได้ */
        }
        if (counts <= (MY_CAL_ARR_MAX + 1U)) {
            *psc = p;
            *arr = counts - 1U;
            return true;
        }
    }
    return false;
}

bool MyCalib_Start(MyCalibInfo *info)
{
    if (s_running) {
        return true;    /* เรียกซ้ำไม่เป็นอันตราย */
    }

    const uint32_t timer_clk = calib_timer_clock_hz();
    uint32_t psc = 0U;
    uint32_t arr = 0U;

    if (!calib_solve_dividers(timer_clk, (uint32_t)MY_CAL_FREQ_HZ, &psc, &arr)) {
        return false;
    }

    /* ==== เปิดสัญญาณนาฬิกาให้พอร์ตและตัวจับเวลา ==== */
    __HAL_RCC_GPIOC_CLK_ENABLE();
    RCC->APB1ENR |= RCC_APB1ENR_TIM3EN;
    (void)RCC->APB1ENR;     /* อ่านกลับ เพื่อรอให้การเขียนมีผลจริงก่อนใช้งาน */

    /* ==== ตั้งค่าขา PC8 ====
     * push pull เพื่อให้ขาขับได้ทั้งขึ้นและลง ขอบสัญญาณจึงคมที่สุด
     * ความเร็วสูงเพื่อลดเวลาไต่ขอบ ซึ่งสำคัญเมื่อวัดคาบด้วยความละเอียดสูง
     */
    GPIO_InitTypeDef gpio = {0};
    gpio.Pin       = MY_CAL_GPIO_PIN;
    gpio.Mode      = GPIO_MODE_AF_PP;
    gpio.Pull      = GPIO_NOPULL;
    gpio.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Alternate = MY_CAL_GPIO_AF;
    HAL_GPIO_Init(MY_CAL_GPIO_PORT, &gpio);

    /* ==== ตั้งค่า TIM3 ทีละรีจิสเตอร์ ==== */

    TIM3->CR1  = 0U;                        /* หยุดตัวนับก่อนตั้งค่า */
    TIM3->PSC  = psc;                       /* ตัวหารความถี่ขาเข้า */
    TIM3->ARR  = arr;                       /* นับถึงค่านี้แล้วเริ่มใหม่ */
    TIM3->CCR3 = 0U;                        /* สลับขาเมื่อตัวนับเท่ากับศูนย์ */

    TIM3->CCMR2 = MY_CAL_CCMR2_TOGGLE_CH3;  /* ช่องสามเป็นเอาต์พุต โหมด toggle */
    TIM3->CCER  = MY_CAL_CCER_ENABLE_CH3;   /* เปิดสัญญาณออกที่ขา */

    /* บังคับให้ค่า PSC ที่เพิ่งเขียนมีผลทันที
     * ปกติ PSC จะถูกโหลดเข้าใช้งานเมื่อจบรอบถัดไปเท่านั้น
     * การสั่ง update event ด้วยมือ ทำให้เริ่มด้วยค่าที่ถูกต้องตั้งแต่รอบแรก
     */
    TIM3->EGR = TIM_EGR_UG;

    TIM3->CR1 = TIM_CR1_CEN;                /* เริ่มนับ */

    s_running = true;

    if (info != NULL) {
        const double actual =
            (double)timer_clk / ((double)(psc + 1U) * (double)(arr + 1U) * 2.0);

        info->timer_clk_hz = timer_clk;
        info->prescaler    = psc;
        info->period       = arr;
        info->target_hz    = (uint32_t)MY_CAL_FREQ_HZ;
        info->actual_hz    = actual;
        info->error_ppm    =
            (actual - (double)MY_CAL_FREQ_HZ) / (double)MY_CAL_FREQ_HZ * 1e6;
    }
    return true;
}

void MyCalib_Stop(void)
{
    if (!s_running) {
        return;
    }

    TIM3->CR1  = 0U;    /* หยุดนับ */
    TIM3->CCER = 0U;    /* ปิดสัญญาณออก */

    /* คืนขาเป็นอินพุตลอย เพื่อไม่ให้ขับสัญญาณค้างไว้โดยไม่ตั้งใจ */
    GPIO_InitTypeDef gpio = {0};
    gpio.Pin  = MY_CAL_GPIO_PIN;
    gpio.Mode = GPIO_MODE_INPUT;
    gpio.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(MY_CAL_GPIO_PORT, &gpio);

    s_running = false;
}

void MyCalib_Report(const MyCalibInfo *info)
{
    if (info == NULL) {
        return;
    }

    /* คำนวณด้วยจำนวนเต็มล้วน ไม่ใช้ทศนิยม
     *
     * เหตุผล โปรเจคนี้คอมไพล์ด้วย nano.specs ซึ่งตัดการพิมพ์ %f ออก
     * เพื่อประหยัดหน่วยความจำ ถ้าใช้ %f จะได้ช่องว่างแทนตัวเลข
     * ยืนยันจากผลรันจริง 8 ก.ย. 2026 ได้ CAL,84000000,0,41999,1000,,
     *
     * ผลพลอยได้ จำนวนเต็มไม่มีการปัดเศษเลย จึงแม่นกว่าทศนิยมด้วยซ้ำ
     *
     * d คือจำนวนครั้งที่ตัวนับต้องนับ เพื่อให้ได้คลื่นครบหนึ่งลูก
     * คูณสองเพราะโหมด toggle สลับขาหนึ่งครั้งต่อหนึ่งคาบของตัวนับ
     */
    const uint64_t d   = 2ULL * (uint64_t)(info->prescaler + 1U)
                              * (uint64_t)(info->period + 1U);
    const uint64_t clk = (uint64_t)info->timer_clk_hz;

    /* แยกเป็นส่วนจำนวนเต็ม กับส่วนทศนิยมหกตำแหน่ง โดยไม่ใช้ชนิด double */
    const uint32_t hz_whole = (uint32_t)(clk / d);
    const uint32_t hz_micro = (uint32_t)(((clk % d) * 1000000ULL) / d);

    /* ความคลาดเคลื่อน หน่วยส่วนในพันล้าน
     * ใช้หน่วยละเอียดกว่า ppm เพราะค่าที่คาดหวังคือศูนย์
     * ถ้าไม่เป็นศูนย์ ต้องเห็นให้ชัดว่าต่างเท่าไหร่
     */
    const int64_t target_x_d = (int64_t)info->target_hz * (int64_t)d;
    const int64_t diff       = (int64_t)clk - target_x_d;
    const int64_t err_ppb    = (diff * 1000000000LL) / target_x_d;

    printf("CAL,%lu,%lu,%lu,%lu,%lu.%06lu,%ld\r\n",
           (unsigned long)info->timer_clk_hz,
           (unsigned long)info->prescaler,
           (unsigned long)info->period,
           (unsigned long)info->target_hz,
           (unsigned long)hz_whole,
           (unsigned long)hz_micro,
           (long)err_ppb);
}

/* ==========================================================================
 * ข้อจำกัดที่ทราบ
 *
 * 1. TIM3 ต้องไม่ถูกใช้ที่อื่น
 *    ตรวจแล้ว 8 ก.ย. 2026 ด้วยการค้นทั้งโปรเจค พบเฉพาะในตารางเวกเตอร์ของ
 *    ไฟล์ startup ซึ่งเป็นรายการว่างที่มีอยู่ในทุกโปรเจค ไม่ใช่การใช้งานจริง
 *
 * 2. โค้ดนี้เขียนรีจิสเตอร์ TIM3 ทับทั้งหมด ไม่ได้อ่านค่าเดิมมาผสม
 *    ถ้าอนาคตมีส่วนอื่นใช้ TIM3 ด้วย ต้องเปลี่ยนวิธีเขียนใหม่
 *
 * 3. ความแม่นของคลื่นที่ออกมา เท่ากับความแม่นของ HSE เท่านั้น
 *    ถ้า HSE เพี้ยน คลื่นนี้ก็เพี้ยนตาม
 *    ซึ่งไม่เป็นปัญหา เพราะจุดประสงค์คือเทียบว่านาฬิกาบอร์ดกับนาฬิกา
 *    logic analyzer ตรงกันหรือไม่ ไม่ใช่การหาค่าความถี่สัมบูรณ์
 *
 * 4. ยังไม่ได้ตรวจว่าขา PC8 โผล่พ้นบอร์ดขยายจริงหรือไม่
 *    ต้องตรวจด้วยตาและด้วยมัลติมิเตอร์ก่อนต่อสาย
 *
 * 5. ฟิลด์ actual_hz และ error_ppm ในโครงสร้าง MyCalibInfo ยังเป็นชนิด double
 *    คำนวณไว้เผื่อโค้ดส่วนอื่นเรียกใช้ แต่ MyCalib_Report ไม่ได้ใช้พิมพ์
 *    เพราะเหตุผลเรื่อง nano.specs ข้างต้น
 *
 * 6. ระหว่างโหมดสอบเทียบ เซ็นเซอร์ ToF ถูกเริ่มต้นไปแล้วโดย my_tof_init
 *    ถ้าไม่ได้เสียบบอร์ดขยาย จะขึ้นข้อความ sensor init failed ซึ่งไม่เป็นไร
 *    โหมดนี้ไม่ใช้เซ็นเซอร์เลย
 * ========================================================================== */
