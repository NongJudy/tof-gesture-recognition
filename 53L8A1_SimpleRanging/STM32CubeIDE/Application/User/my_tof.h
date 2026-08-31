#ifndef MY_TOF_H          // กันไฟล์นี้ถูก include ซ้ำ
#define MY_TOF_H

#include <stdint.h>

/* ===== สวิตช์เลือกความละเอียด =====
   1 = 4x4 ที่ 60 Hz   สำหรับท่าทางเคลื่อนไหว (ต้องการเฟรมถี่)
   0 = 8x8 ที่ 15 Hz   สำหรับท่าทางนิ่ง (ต้องการรายละเอียดเชิงพื้นที่)

   ที่มา: UM3109 ตารางที่ 2 ระบุความถี่สูงสุด 4x4 = 60 Hz, 8x8 = 15 Hz
   เหตุผลที่ต้องมีทั้งสอง: Boner et al. 2022 พบว่าท่าทางอยู่ในมุมมอง
   เพียง 0.2 วินาที ที่ 15 Hz จะได้แค่ 3 เฟรม ไม่พอดูการเคลื่อนไหว   */
#define MY_TOF_USE_4X4      1

#if MY_TOF_USE_4X4
  #define MY_TOF_ZONES      16      // จำนวน zone โหมด 4x4
  #define MY_TOF_FREQ_HZ    60      // ความถี่วัด (Hz)
#else
  #define MY_TOF_ZONES      64      // จำนวน zone โหมด 8x8
  #define MY_TOF_FREQ_HZ    15      // ความถี่วัด (Hz)
#endif

/* ===== ฟังก์ชันที่เราเขียนเอง ===== */

/* เริ่มต้นเซ็นเซอร์: init, ตั้งความละเอียด, ตั้งความถี่, เริ่มวัด
   คืนค่า 0 = สำเร็จ */
uint8_t my_tof_init(void);

/* อ่าน 1 เฟรมเข้าหน่วยความจำ MCU
   คืนค่า 1 = ได้ข้อมูลใหม่, 0 = ยังไม่มี */
uint8_t my_tof_read_frame(void);

/* ส่งเฟรมล่าสุดออก UART เป็น CSV */
void my_tof_send_frame(void);

#endif /* MY_TOF_H */
