/**
 ******************************************************************************
 * @file    my_platform.c
 * @brief   Platform layer เชื่อม STM32F411RE เข้ากับ VL53L8CX ผ่าน I2C
 *          เรียก HAL โดยตรง ไม่ผ่าน function pointer ของ BSP
 *
 * @note    เขียนเองทั้งหมดตามคำสั่ง อ.Seal ("Dev driver for ToF เอง")
 *
 * ASSUMPTIONS (ถ้าข้อไหนไม่จริง โค้ดนี้ใช้ไม่ได้):
 *   1. I2C1 ที่ PB8(SCL)/PB9(SDA) 400 kHz  — handle ชื่อ hi2c1
 *   2. GPIO ทั้งสองขา = "No pull-up and no pull-down"
 *      (pull-up 2.2k อยู่บนชีลด์: R18/R19 ฝั่ง host, R2/R3 ฝั่งเซ็นเซอร์)
 *   3. p_platform->address = 0x52 (BSP ตั้งให้แล้วใน vl53l8cx.c บรรทัด 111)
 *   4. BLOCKING ล้วน — ห้ามเรียกจากใน ISR
 ******************************************************************************
 */

#include <string.h>
#include "platform.h"
#include "stm32f4xx_hal.h"

/* ── ค่าคงที่ปรับได้ ───────────────────────────────────────── */

extern I2C_HandleTypeDef hi2c1;      // ประกาศไว้ที่ stm32f4xx_nucleo_bus.c

#define TOF_I2C_HANDLE        (&hi2c1)
#define TOF_I2C_CHUNK_SIZE    (512U)  // HAL รับ uint16_t แต่ ULD ส่ง uint32_t
#define TOF_I2C_TIMEOUT_MS    (100U)  // 8.6 เท่าของ worst case 11.6 ms
#define TOF_STATUS_OK         (0U)    // ULD ถือว่า 0 = สำเร็จ
#define TOF_STATUS_ERROR      (255U)

/* ── 1. อ่านหลายไบต์ — คอขวดของทั้งระบบอยู่ที่ฟังก์ชันนี้ ──── */

uint8_t VL53L8CX_RdMulti(VL53L8CX_Platform *p_platform,
                         uint16_t RegisterAdress,
                         uint8_t *p_values,
                         uint32_t size)
{
    uint32_t remaining = size;
    uint32_t offset    = 0U;
    uint16_t chunk;

    // กัน null ก่อนแตะฮาร์ดแวร์ ไม่งั้น HardFault หาสาเหตุยากมาก
    if ((p_platform == NULL) || (p_values == NULL)) {
        return TOF_STATUS_ERROR;
    }

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
    return TOF_STATUS_OK;
}

/* ── 2. เขียนหลายไบต์ — ใช้หนักตอน init (อัปโหลด firmware) ─── */

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
                              &p_values[offset],
                              chunk,
                              TOF_I2C_TIMEOUT_MS) != HAL_OK) {
            return TOF_STATUS_ERROR;
        }

        offset    += chunk;
        remaining -= chunk;
    }
    return TOF_STATUS_OK;
}

/* ── 3. อ่าน/เขียน 1 ไบต์ — เรียกตัวข้างบนซ้ำ ────────────────
   เขียนซ้ำ = บั๊กซ้ำ ถ้าแก้ chunk ทีหลังจะได้แก้ที่เดียว        */

uint8_t VL53L8CX_RdByte(VL53L8CX_Platform *p_platform,
                        uint16_t RegisterAdress,
                        uint8_t *p_value)
{
    return VL53L8CX_RdMulti(p_platform, RegisterAdress, p_value, 1U);
}

uint8_t VL53L8CX_WrByte(VL53L8CX_Platform *p_platform,
                        uint16_t RegisterAdress,
                        uint8_t value)
{
    uint8_t tmp = value;    // ต้องมีตัวแปรจริง เพราะ WrMulti ขอ pointer
    return VL53L8CX_WrMulti(p_platform, RegisterAdress, &tmp, 1U);
}

/* ── 4. สลับ endian — เซ็นเซอร์ big-endian / M4 little-endian ──
   คืนค่า void เพราะแค่สลับไบต์ในหน่วยความจำ ไม่มีทางล้มเหลว     */

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

/* ── 5. หน่วงเวลา — เรียก HAL ตรง ไม่ผ่าน GetTick pointer ──── */

uint8_t VL53L8CX_WaitMs(VL53L8CX_Platform *p_platform, uint32_t TimeMs)
{
    uint32_t start = HAL_GetTick();
    (void)p_platform;                   // ไม่ได้ใช้ แต่ต้องมีตาม prototype

    // ลบแบบ unsigned -> ยังถูกต้องแม้ tick ล้น 32 บิต (ทุก ~49 วัน)
    while ((HAL_GetTick() - start) < TimeMs) { }

    return TOF_STATUS_OK;
}

/* ── 6. ฟังก์ชันเสริมของเราเอง (ไม่ใช่ของ ULD) ใช้ทดสอบ T1 ─── */

uint8_t my_platform_i2c_probe(uint16_t address)
{
    return (HAL_I2C_IsDeviceReady(TOF_I2C_HANDLE, address, 3, 100) == HAL_OK)
               ? TOF_STATUS_OK : TOF_STATUS_ERROR;
}

