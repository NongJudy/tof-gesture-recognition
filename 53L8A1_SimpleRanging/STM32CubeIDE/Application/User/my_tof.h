#ifndef MY_TOF_H          // กันไฟล์นี้ถูก include ซ้ำ
#define MY_TOF_H

#include <stdint.h>

/* ===== ค่าคงที่ของระบบเรา ===== */
#define MY_TOF_ZONES        64      // จำนวน zone โหมด 8x8
#define MY_TOF_FREQ_HZ      15      // ความถี่วัด (8x8 สูงสุด 15 Hz ตาม UM3109)

/* ===== ฟังก์ชันที่เราเขียนเอง ===== */

/* เริ่มต้นเซ็นเซอร์: init, ตั้ง 8x8, ตั้งความถี่, เริ่มวัด
   คืนค่า 0 = สำเร็จ */
uint8_t my_tof_init(void);

/* อ่าน 1 เฟรมเข้าหน่วยความจำ MCU
   คืนค่า 1 = ได้ข้อมูลใหม่, 0 = ยังไม่มี */
uint8_t my_tof_read_frame(void);

/* ส่งเฟรมล่าสุดออก UART เป็น CSV */
void my_tof_send_frame(void);

#endif /* MY_TOF_H */
