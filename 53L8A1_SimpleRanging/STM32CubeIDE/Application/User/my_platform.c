/**
 ******************************************************************************
 * @file    my_platform.c
 * @brief   Platform layer เชื่อม STM32 กับ VL53L8CX ผ่าน I2C
 *          + เครื่องมือวัดเวลาระดับ cycle สำหรับงานวิจัย
 *
 * @note    เขียนเองทั้งหมดตามคำสั่ง อ.Seal
 *          ("Dev driver ส่วนเชื่อมต่อระหว่าง Arm cortex-M4 กับ ToF sensor")
 *
 * ASSUMPTIONS:
 *   1. I2C1 ที่ PB8(SCL)/PB9(SDA) 400 kHz — handle ชื่อ hi2c1
 *   2. GPIO ทั้งสองขา = "No pull-up and no pull-down"
 *      (pull-up 2.2k อยู่บนชีลด์: R18/R19 ฝั่ง host, R2/R3 ฝั่งเซ็นเซอร์)
 *   3. p_platform->address = 0x52 (BSP ตั้งให้ใน vl53l8cx.c บรรทัด 111)
 *   4. BLOCKING ล้วน — ห้ามเรียกจากใน ISR
 ******************************************************************************
 */

#include <string.h>
#include "platform.h"
#include "stm32f4xx_hal.h"

/* ── ค่าคงที่ ──────────────────────────────────────────────── */

extern I2C_HandleTypeDef hi2c1;

#define TOF_I2C_HANDLE        (&hi2c1)
#define TOF_I2C_CHUNK_SIZE    (512U)  // HAL รับ uint16_t แต่ ULD ส่ง uint32_t
#define TOF_I2C_TIMEOUT_MS    (100U)  // 8.6 เท่าของ worst case 11.6 ms
#define TOF_STATUS_OK         (0U)    // ULD ถือว่า 0 = สำเร็จ
#define TOF_STATUS_ERROR      (255U)

/* ── ตัวแปรเก็บสถิติ (ให้ my_tof.c อ่านได้) ─────────────────
   volatile เพราะค่าถูกแก้ในฟังก์ชันที่คอมไพเลอร์อาจ optimize ทิ้ง */

volatile uint32_t g_rd_calls      = 0;  // จำนวนครั้งที่เรียก RdMulti
volatile uint32_t g_rd_bytes      = 0;  // ไบต์รวมที่อ่านมา
volatile uint32_t g_rd_cycles     = 0;  // cycle รวมที่ใช้
volatile uint32_t g_rd_max_bytes  = 0;  // ไบต์ของก้อนที่ใหญ่ที่สุด
volatile uint32_t g_rd_max_cycles = 0;  // cycle ของก้อนใหญ่ที่สุดนั้น

/* ── ตัวนับ cycle ของ Cortex-M4 (DWT) ──────────────────────
   ที่ 84 MHz: 1 นับ = 11.9 ns ละเอียดกว่า HAL_GetTick() ล้านเท่า */

void my_platform_dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  // เปิดหน่วย trace
    DWT->CYCCNT = 0U;                                // ล้างตัวนับ
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;            // สั่งให้เริ่มนับ
}

uint32_t my_platform_cycles(void)
{
    return DWT->CYCCNT;
}

/* แปลง cycle เป็นไมโครวินาที โดยอ่านความถี่ CPU จริงตอนรัน
   ไม่ hardcode 84 MHz เพราะถ้าเปลี่ยน clock ทีหลังตัวเลขจะผิดทันที */
uint32_t my_platform_cycles_to_us(uint32_t cycles)
{
    return cycles / (SystemCoreClock / 1000000U);
}

void my_platform_stats_reset(void)
{
    g_rd_calls = 0; g_rd_bytes = 0; g_rd_cycles = 0;
    g_rd_max_bytes = 0; g_rd_max_cycles = 0;
}

/* ── 1. อ่านหลายไบต์ — คอขวดของทั้งระบบ + จุดวัดเวลา ─────── */

uint8_t VL53L8CX_RdMulti(VL53L8CX_Platform *p_platform,
                         uint16_t RegisterAdress,
                         uint8_t *p_values,
                         uint32_t size)
{
    uint32_t remaining = size;
    uint32_t offset    = 0U;
    uint32_t t_start;
    uint32_t elapsed;
    uint16_t chunk;

    // กัน null ก่อนแตะฮาร์ดแวร์ ไม่งั้น HardFault หาสาเหตุยากมาก
    if ((p_platform == NULL) || (p_values == NULL)) {
        return TOF_STATUS_ERROR;
    }

    t_start = DWT->CYCCNT;          // ── เริ่มจับเวลา ──

    while (remaining > 0U) {
        chunk = (remaining > TOF_I2C_CHUNK_SIZE)
                    ? (uint16_t)TOF_I2C_CHUNK_SIZE
                    : (uint16_t)remaining;

        // register index เดินตาม offset เพราะเซ็นเซอร์ auto-increment pointer
        if (HAL_I2C_Mem_Read(TOF_I2C_HANDLE,
                             p_platform->address,      // 0x52 ตรง ๆ ห้าม shift
                             (uint16_t)(RegisterAdress + offset),
                             I2C_MEMADD_SIZE_16BIT,    // 16 บิต ตาม DS14161
                             &p_values[offset],
                             chunk,
                             TOF_I2C_TIMEOUT_MS) != HAL_OK) {
            return TOF_STATUS_ERROR;   // fail fast ไม่ซ่อมเอง ไม่ซ่อนปัญหา
        }

        offset    += chunk;
        remaining -= chunk;
    }

    // ลบแบบ unsigned -> ถูกต้องแม้ตัวนับล้น 32 บิต (ทุก ~51 วินาทีที่ 84 MHz)
    elapsed = DWT->CYCCNT - t_start;    // ── หยุดจับเวลา ──

    g_rd_calls  += 1U;
    g_rd_bytes  += size;
    g_rd_cycles += elapsed;

    // เก็บก้อนที่ใหญ่ที่สุดแยกไว้ = การโอนข้อมูลจริง
    // (ก้อนเล็ก 1-4 ไบต์คือการ poll เช็คสถานะ ไม่ใช่ข้อมูล)
    if (size > g_rd_max_bytes) {
        g_rd_max_bytes  = size;
        g_rd_max_cycles = elapsed;
    }

    return TOF_STATUS_OK;
}

/* ── 2. เขียนหลายไบต์ — ใช้หนักตอน init (อัปโหลด firmware) ─ */

uint8_t VL53L8CX_WrMulti(VL53L8CX_Platform *p_platform,
                         uint16_t RegisterAdress,
                         uint8_t *p_values,
                         uint32_t size)
{
    uint32_t remaining = size;
    uint32_t offset    = 0U;
    uint16_t chunk;

    if ((p_platform == NULL) || (p_values == NULL)) {
        return TOF_STATUS_ERROR;
    }

    while (remaining > 0U) {
        chunk = (remaining > TOF_I2C_CHUNK_SIZE)
                    ? (uint16_t)TOF_I2C_CHUNK_SIZE
                    : (uint16_t)remaining;

        if (HAL_I2C_Mem_Write(TOF_I2C_HANDLE,
                              p_platform->address,
                              (uint16_t)(RegisterAdress + offset),
                              I2C_MEMADD_SIZE_16BIT,
                              &p_values[offset], chunk,
                              TOF_I2C_TIMEOUT_MS) != HAL_OK) {
            return TOF_STATUS_ERROR;
        }

        offset    += chunk;
        remaining -= chunk;
    }
    return TOF_STATUS_OK;
}

/* ── 3. อ่าน/เขียน 1 ไบต์ — เรียกตัวข้างบนซ้ำ ──────────────
   เขียนซ้ำ = บั๊กซ้ำ ถ้าแก้ chunk ทีหลังจะได้แก้ที่เดียว      */

uint8_t VL53L8CX_RdByte(VL53L8CX_Platform *p_platform,
                        uint16_t RegisterAdress, uint8_t *p_value)
{
    return VL53L8CX_RdMulti(p_platform, RegisterAdress, p_value, 1U);
}

uint8_t VL53L8CX_WrByte(VL53L8CX_Platform *p_platform,
                        uint16_t RegisterAdress, uint8_t value)
{
    uint8_t tmp = value;    // ต้องมีตัวแปรจริง เพราะ WrMulti ขอ pointer
    return VL53L8CX_WrMulti(p_platform, RegisterAdress, &tmp, 1U);
}

/* ── 4. สลับ endian — เซ็นเซอร์ big-endian / M4 little-endian ─
   คืนค่า void ตาม prototype ใน platform.h บรรทัด 143          */

void VL53L8CX_SwapBuffer(uint8_t *buffer, uint16_t size)
{
    uint32_t i, word;

    if (buffer == NULL) {
        return;             // ออกเฉย ๆ ไม่มีค่าให้คืน
    }

    for (i = 0U; (i + 4U) <= (uint32_t)size; i += 4U) {
        memcpy(&word, &buffer[i], 4U);  // memcpy กัน unaligned access
        word = __REV(word);             // CMSIS: คำสั่ง REV ของ M4 จบใน 1 cycle
        memcpy(&buffer[i], &word, 4U);
    }
}

/* ── 5. หน่วงเวลา — เรียก HAL ตรง ไม่ผ่าน GetTick pointer ─── */

uint8_t VL53L8CX_WaitMs(VL53L8CX_Platform *p_platform, uint32_t TimeMs)
{
    uint32_t start = HAL_GetTick();
    (void)p_platform;                   // ไม่ได้ใช้ แต่ต้องมีตาม prototype

    // ลบแบบ unsigned -> ยังถูกต้องแม้ tick ล้น 32 บิต (ทุก ~49 วัน)
    while ((HAL_GetTick() - start) < TimeMs) { }

    return TOF_STATUS_OK;
}

/* ── 6. ฟังก์ชันเสริมของเราเอง (ไม่ใช่ของ ULD) ใช้ทดสอบ T1 ── */

uint8_t my_platform_i2c_probe(uint16_t address)
{
    return (HAL_I2C_IsDeviceReady(TOF_I2C_HANDLE, address, 3, 100) == HAL_OK)
               ? TOF_STATUS_OK : TOF_STATUS_ERROR;
}
