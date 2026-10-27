/**
 ******************************************************************************
 * @file    my_uart.c
 * @brief   ส่งข้อมูลออก UART แบบ interrupt (non-blocking)
 *          ใช้ ring buffer + USART2 TX interrupt
 *
 * @note    เขียนเองตามคำสั่ง อ.Seal ("USART ส่งค่า ใช้ interrupt mode")
 *
 * หลักการ:
 *   printf -> _write() -> หย่อนตัวอักษรลง ring buffer -> คืนค่าทันที
 *   ฮาร์ดแวร์ส่งเสร็จ 1 ไบต์ -> ยิง interrupt -> หยิบไบต์ถัดไปส่ง
 *   CPU จึงไม่ต้องยืนรอ เอาเวลาไปทำงานอื่นได้
 *
 * @note    _write() ใน syscalls.c ประกาศเป็น weak เราจึงเขียนทับได้
 *          โดยไม่ต้องแก้ไฟล์ของ ST เลย
 *
 * แก้ไข 27 ต.ค. 2026 (กติกา A17 ข้อ 5, ขั้น 5.5 replay): เพิ่ม "รับ" ข้อมูลแบบ interrupt (RX)
 *   my_uart_rx_start() เริ่มรับทีละไบต์ด้วย HAL_UART_Receive_IT -> ไบต์ลง rx ring buffer
 *   my_uart_rx_get()   หยิบไบต์ออกจากบัฟเฟอร์ (ไม่รอ คืน 0 ถ้าว่าง)
 *   ถ้าไม่เรียก my_uart_rx_start() (โหมดปกติ) ส่วนรับไม่ทำงานเลย การส่งเหมือนเดิมทุกอย่าง
 ******************************************************************************
 */

#include <stdint.h>
#include "stm32f4xx_hal.h"
#include "stm32f4xx_nucleo.h"      // hcom_uart[] และ macro huart2

/* ── สวิตช์เปรียบเทียบ ───────────────────────────────────────
   1 = ส่งแบบ interrupt (ของใหม่)
   0 = ส่งแบบ blocking  (ของเดิม ไว้เทียบผล)
   เปลี่ยนค่าเดียว เงื่อนไขอื่นเหมือนกันหมด -> เทียบได้อย่างเป็นธรรม */
#define MY_UART_USE_IRQ      1

/* ขนาดบัฟเฟอร์ ต้องเป็นเลขยกกำลัง 2 เพื่อใช้ & แทน % (เร็วกว่ามาก) */
#define UART_TX_BUF_SIZE     2048U
#define UART_TX_BUF_MASK     (UART_TX_BUF_SIZE - 1U)

#define MY_UART_HANDLE       (&hcom_uart[COM1])
#define MY_UART_IRQn         USART2_IRQn

/* ── ring buffer ────────────────────────────────────────────
   volatile เพราะถูกแก้ทั้งจากโปรแกรมหลักและจาก interrupt */
static volatile uint8_t  tx_buf[UART_TX_BUF_SIZE];
static volatile uint16_t tx_head = 0;   // ตำแหน่งเขียน (โปรแกรมหลักแก้)
static volatile uint16_t tx_tail = 0;   // ตำแหน่งอ่าน  (interrupt แก้)
static volatile uint8_t  tx_busy = 0;   // กำลังส่งอยู่หรือไม่

/* ไบต์ที่กำลังส่ง ต้องเป็น static ห้ามเป็นตัวแปรท้องถิ่น
   เพราะ HAL เก็บ "ที่อยู่" ไว้ แล้วค่อยอ่านทีหลังตอน interrupt
   ถ้าใช้ตัวแปรท้องถิ่น มันจะหายไปก่อน -> ส่งข้อมูลขยะ */
static volatile uint8_t  tx_byte;

/* สถิติ: นับตัวอักษรที่ถูกส่ง และจำนวนครั้งที่บัฟเฟอร์เต็ม */
volatile uint32_t g_uart_chars = 0;
volatile uint32_t g_uart_full  = 0;

/* ── ส่วนรับ (RX) สำหรับโหมด replay ─────────────────────────
   ขนาดต้องเป็นเลขยกกำลัง 2 ; 2048 > 1 แพ็กเก็ต replay (454 ไบต์) หลายเท่า */
#define UART_RX_BUF_SIZE     2048U
#define UART_RX_BUF_MASK     (UART_RX_BUF_SIZE - 1U)
static volatile uint8_t  rx_buf[UART_RX_BUF_SIZE];
static volatile uint16_t rx_head = 0;   // ตำแหน่งเขียน (interrupt แก้)
static volatile uint16_t rx_tail = 0;   // ตำแหน่งอ่าน  (โปรแกรมหลักแก้)
static volatile uint8_t  rx_byte;       // ไบต์ที่ HAL กำลังรับ (ต้อง static เหตุผลเดียวกับ tx_byte)
static volatile uint8_t  rx_active = 0; // 1 = เรียก my_uart_rx_start() แล้ว
volatile uint32_t g_uart_rx_bytes = 0;  // ไบต์ที่รับได้ทั้งหมด
volatile uint32_t g_uart_rx_ovf   = 0;  // ไบต์ที่ทิ้งเพราะ rx ring buffer เต็ม
volatile uint32_t g_uart_rx_err   = 0;  // จำนวนครั้งที่ HAL แจ้ง error (overrun / noise / framing)

/* ── เปิดใช้ interrupt ของ USART2 ───────────────────────────
   ต้องเรียกหลัง BSP_COM_Init() เพราะ UART ต้องถูกตั้งค่าก่อน */
void my_uart_init(void)
{
    tx_head = 0; tx_tail = 0; tx_busy = 0;
    g_uart_chars = 0; g_uart_full = 0;

#if MY_UART_USE_IRQ
    // ลำดับความสำคัญ 5 = ต่ำกว่างานวิกฤต แต่สูงกว่างานทั่วไป
    HAL_NVIC_SetPriority(MY_UART_IRQn, 5, 0);
    HAL_NVIC_EnableIRQ(MY_UART_IRQn);
#endif
}

/* ── ส่วนรับ (RX) ────────────────────────────────────────────
   ต้องใช้คู่กับ MY_UART_USE_IRQ = 1 (USART2_IRQHandler อยู่ในไฟล์นี้) */

/* สั่ง HAL ให้รับไบต์ถัดไป 1 ไบต์ — เรียกตอนเริ่ม และทุกครั้งที่รับเสร็จ/มี error */
static void uart_rx_arm(void)
{
    (void)HAL_UART_Receive_IT(MY_UART_HANDLE, (uint8_t *)&rx_byte, 1U);
}

void my_uart_rx_start(void)
{
    uint32_t prim = __get_PRIMASK();
    __disable_irq();                 // กันชนกับ interrupt ฝั่งส่งระหว่างตั้งค่า
    rx_head = 0; rx_tail = 0;
    g_uart_rx_bytes = 0; g_uart_rx_ovf = 0; g_uart_rx_err = 0;
    rx_active = 1U;
    uart_rx_arm();
    __set_PRIMASK(prim);
}

/* หยิบ 1 ไบต์จากบัฟเฟอร์รับ: คืน 1 = ได้ไบต์ใน *b, 0 = ว่าง (ไม่รอ)
   ถ้าการรับหยุดไปเอง (HAL ไม่พร้อม) จะสั่งรับใหม่ให้ — กันกรณี re-arm ใน interrupt ไม่สำเร็จ */
int my_uart_rx_get(uint8_t *b)
{
    if (rx_active && MY_UART_HANDLE->RxState == HAL_UART_STATE_READY)
    {
        uint32_t prim = __get_PRIMASK();
        __disable_irq();
        if (MY_UART_HANDLE->RxState == HAL_UART_STATE_READY)
        {
            uart_rx_arm();
        }
        __set_PRIMASK(prim);
    }
    if (rx_tail == rx_head)
    {
        return 0;
    }
    *b = rx_buf[rx_tail];
    rx_tail = (uint16_t)((rx_tail + 1U) & UART_RX_BUF_MASK);
    return 1;
}

#if MY_UART_USE_IRQ

/* หยิบไบต์ถัดไปจากบัฟเฟอร์ส่ง — เรียกได้ทั้งจาก main และจาก interrupt */
static void uart_send_next(void)
{
    tx_byte = tx_buf[tx_tail];
    tx_tail = (uint16_t)((tx_tail + 1U) & UART_TX_BUF_MASK);
    (void)HAL_UART_Transmit_IT(MY_UART_HANDLE, (uint8_t *)&tx_byte, 1U);
}

/* หย่อนตัวอักษร 1 ตัวลงบัฟเฟอร์ */
static void uart_put(uint8_t c)
{
    uint32_t prim;
    uint16_t next = (uint16_t)((tx_head + 1U) & UART_TX_BUF_MASK);

    // บัฟเฟอร์เต็ม -> รอให้ interrupt ระบายออกก่อน
    // เลือก "รอ" แทน "ทิ้ง" เพราะข้อมูลวัดผลหายไม่ได้
    while (next == tx_tail) {
        g_uart_full++;
    }

    tx_buf[tx_head] = c;
    tx_head = next;

    /* ปิด interrupt ชั่วคราวตอนแตะ tx_busy/tx_tail
       เพราะ interrupt ก็แก้ตัวแปรพวกนี้ ถ้าชนกันจะได้ค่าผิด
       เก็บ PRIMASK เดิมไว้ก่อน กันกรณีถูกเรียกตอน interrupt ปิดอยู่แล้ว */
    prim = __get_PRIMASK();
    __disable_irq();
    if (tx_busy == 0U) {
        tx_busy = 1U;
        uart_send_next();
    }
    __set_PRIMASK(prim);

    g_uart_chars++;
}

/* ── ฮาร์ดแวร์ส่งเสร็จ 1 ไบต์ -> HAL เรียกฟังก์ชันนี้ ─────── */
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance != USART2) {
        return;                      // ไม่ใช่ของเรา ปล่อยผ่าน
    }

    if (tx_tail != tx_head) {
        uart_send_next();            // ยังมีค้าง ส่งต่อ
    } else {
        tx_busy = 0U;                // หมดแล้ว หยุดพัก
    }
}

/* ── รับเสร็จ 1 ไบต์ -> HAL เรียกฟังก์ชันนี้ (ใน interrupt) ── */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    uint16_t next;
    if (huart->Instance != USART2) {
        return;
    }
    next = (uint16_t)((rx_head + 1U) & UART_RX_BUF_MASK);
    if (next == rx_tail) {
        g_uart_rx_ovf++;             // บัฟเฟอร์เต็ม: ทิ้งไบต์นี้ (PC จะเห็นจาก checksum แล้วส่งซ้ำ)
    } else {
        rx_buf[rx_head] = rx_byte;
        rx_head = next;
    }
    g_uart_rx_bytes++;
    uart_rx_arm();                   // รับไบต์ถัดไปทันที
}

/* ── HAL แจ้ง error (เช่น overrun) -> นับไว้ แล้วสั่งรับต่อ ── */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance != USART2) {
        return;
    }
    g_uart_rx_err++;
    if (rx_active && huart->RxState == HAL_UART_STATE_READY) {
        uart_rx_arm();
    }
}

/* ── ตัวรับสัญญาณ interrupt ของ USART2 ─────────────────────
   ชื่อนี้ถูกกำหนดไว้ในตาราง vector ของ STM32 ห้ามเปลี่ยนชื่อ */
void USART2_IRQHandler(void)
{
    HAL_UART_IRQHandler(MY_UART_HANDLE);
}

#endif /* MY_UART_USE_IRQ */

/* ── เขียนทับ _write() ของ syscalls.c (ประกาศเป็น weak) ────
   printf ทุกครั้งจะวิ่งมาที่นี่ ไม่ผ่าน __io_putchar ของ ST อีก */
int _write(int file, char *ptr, int len)
{
    int i;
    (void)file;

    for (i = 0; i < len; i++) {
#if MY_UART_USE_IRQ
        uart_put((uint8_t)ptr[i]);
#else
        (void)HAL_UART_Transmit(MY_UART_HANDLE, (uint8_t *)&ptr[i], 1U, 100U);
        g_uart_chars++;
#endif
    }
    return len;
}
