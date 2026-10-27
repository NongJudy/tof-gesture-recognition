/**
 * my_recentre.h
 * ย้ายมือให้อยู่กลางภาพ 8x8 ก่อนส่งเข้าโมเดล E5R (เพิ่ม 27 ต.ค. 2026, PROJECT_FACTS §5.1w / §5.1y-U)
 *
 * ต้องได้ผล "ตรงกับ pc_tools/recentre.py ทุกบิต" เพราะ E5R ถูกเทรนด้วยภาพที่ผ่าน recentre.py:
 *   1. โซนมือ = โซนที่ระยะหลังตัดฉากหลัง < 2000 มม. (ใน Python: X[...,0] < (2000-295)/196)
 *   2. cnt = จำนวนโซนมือ, sum_r = ผลรวมเลขแถว, sum_c = ผลรวมเลขคอลัมน์ (จำนวนเต็มทั้งหมด)
 *   3. dr = floor((4*cnt - sum_r) / cnt), dc = floor((4*cnt - sum_c) / cnt)
 *      ภาษา C: เครื่องหมาย / ของจำนวนเต็ม "ปัดเข้าหาศูนย์" เมื่อผลเป็นลบ ไม่ใช่ floor
 *      จึงต้องแก้เอง (my_rc_floor_div) ไม่อย่างนั้นเฟรมที่ dr ติดลบจะเลื่อนผิดไป 1 ช่อง
 *   4. ภาพใหม่ [r][c] = ภาพเดิม [r-dr][c-dc] ; ช่องที่ไม่มีต้นทาง = ค่าว่าง (ระยะ 4000, signal 0 หลัง normalise)
 *      ช่องที่เลื่อนหลุดตาราง = หายไป ; ไม่มีโซนมือ = ไม่เลื่อน
 *   รูปแบบข้อมูล x: 128 float เรียง [แถว][คอลัมน์][ช่อง] = g_ai_in ของ my_tof.c (ช่อง 0 = ระยะ, 1 = signal)
 */
#ifndef MY_RECENTRE_H
#define MY_RECENTRE_H

#include <stdint.h>

#define MY_RC_N  8     /* ภาพ 8x8 */

/* หารแบบปัดลง (floor) สำหรับตัวหาร b > 0 — เหมือน // ของ Python */
static inline int32_t my_rc_floor_div(int32_t a, int32_t b)
{
    int32_t q = a / b;                       /* C ปัดเข้าหาศูนย์ */
    if ((a % b) != 0 && ((a < 0) != (b < 0)))
    {
        q--;                                 /* ผลลบที่มีเศษ → ลดลง 1 ให้เป็น floor */
    }
    return q;
}

/**
 * ย้ายมือมากลางภาพ (แก้ x ในที่)
 * @param x        ข้อมูลเข้าโมเดล 128 float (normalise แล้ว)
 * @param hand     64 ค่า: 1 = โซนมือ (ระยะหลังตัดฉากหลัง < 2000 มม.), 0 = ไม่ใช่
 * @param empty_d  ค่าระยะของช่องว่างหลัง normalise  ((4000-295)/196)
 * @param empty_s  ค่า signal ของช่องว่างหลัง normalise ((0-281)/452)
 * @param tmp      บัฟเฟอร์ชั่วคราว 128 float
 * @param dr_out   (ออก) จำนวนแถวที่เลื่อน ; dc_out (ออก) จำนวนคอลัมน์ที่เลื่อน
 * @return         จำนวนโซนมือ (0 = ไม่เลื่อน)
 */
static inline int32_t my_recentre(float *x, const uint8_t *hand, float empty_d, float empty_s,
                                  float *tmp, int32_t *dr_out, int32_t *dc_out)
{
    int32_t cnt = 0, sum_r = 0, sum_c = 0, dr, dc, r, c, i;

    for (r = 0; r < MY_RC_N; r++)
    {
        for (c = 0; c < MY_RC_N; c++)
        {
            if (hand[r * MY_RC_N + c])
            {
                cnt++;
                sum_r += r;
                sum_c += c;
            }
        }
    }
    *dr_out = 0;
    *dc_out = 0;
    if (cnt == 0)
    {
        return 0;                            /* ไม่มีมือ → ไม่เลื่อน (เหมือน Python) */
    }
    dr = my_rc_floor_div(4 * cnt - sum_r, cnt);
    dc = my_rc_floor_div(4 * cnt - sum_c, cnt);
    *dr_out = dr;
    *dc_out = dc;
    if (dr == 0 && dc == 0)
    {
        return cnt;                          /* อยู่กลางแล้ว ไม่ต้องคัดลอก */
    }

    for (i = 0; i < MY_RC_N * MY_RC_N; i++)  /* เริ่มจากภาพว่างทั้งภาพ */
    {
        tmp[i * 2 + 0] = empty_d;
        tmp[i * 2 + 1] = empty_s;
    }
    for (r = 0; r < MY_RC_N; r++)
    {
        int32_t sr = r - dr;                 /* แถวต้นทาง */
        if (sr < 0 || sr >= MY_RC_N) { continue; }
        for (c = 0; c < MY_RC_N; c++)
        {
            int32_t sc = c - dc;             /* คอลัมน์ต้นทาง */
            if (sc < 0 || sc >= MY_RC_N) { continue; }
            tmp[(r * MY_RC_N + c) * 2 + 0] = x[(sr * MY_RC_N + sc) * 2 + 0];
            tmp[(r * MY_RC_N + c) * 2 + 1] = x[(sr * MY_RC_N + sc) * 2 + 1];
        }
    }
    for (i = 0; i < MY_RC_N * MY_RC_N * 2; i++)
    {
        x[i] = tmp[i];
    }
    return cnt;
}

#endif /* MY_RECENTRE_H */
