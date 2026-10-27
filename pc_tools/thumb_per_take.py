"""thumb_per_take.py — นิ้วโป้งของ Like / Dislike "ชัดทุกรอบ" หรือ "ชัดแค่บางรอบ"? (PROJECT_FACTS §5.1y-O)

posture_maps.py ให้ค่าเฉลี่ยรวมของ User 5 (Like −0.11, Dislike +0.05 ≈ สมมาตร) ซึ่งอธิบายได้ 3 แบบ:
  (ก) บางรอบชัด บางรอบไม่ชัด → ค่าเฉลี่ยหักล้างกัน       ⇒ ดูได้จาก "รายรอบ" (take) ต่างกันมาก
  (ข) นิ้วโป้งโผล่ แต่ข้อมือ/แขนก็โผล่อีกฝั่ง → สมมาตร      ⇒ มือยาวทั้งขึ้นและลง (ช่วงแถวกว้าง, โซนเยอะ)
  (ค) นิ้วโป้งเล็กกว่า 1 โซนที่ 8×8 → มองไม่เห็น           ⇒ มือสั้นทั้งขึ้นและลง (ช่วงแถวแคบ)
สคริปต์นี้แยก 3 แบบด้วยตัวเลขรายรอบ ไม่เทรน ไม่ทายอะไร (เชิงพรรณนาล้วน)

ข้อมูล: data\\phase4 + data\\phase4_extra (ยืน, ข้อมูลเทรน) และ data\\phase56 (นั่ง, ชุดคัดเลือก — ใช้ได้เพราะไม่ได้เลือกโมเดล)
ไม่เปิด data\\phase56_stand / phase57 (ต้องปิดไว้จนได้โมเดลสุดท้าย)

ต่อรอบ (เฉพาะเฟรมในช่วงและมีโซนมือ):
  ลง−ขึ้น  = (แถวสุดท้ายของมือ − แถวจุดศูนย์กลาง) − (แถวจุดศูนย์กลาง − แถวแรกของมือ)   <0 = ยื่นไปทางแถว 0 (ไกลตัว)
  %→0 / %สมมาตร / %→7 = สัดส่วนเฟรมที่ ลง−ขึ้น ≤ −0.5 / อยู่ระหว่าง / ≥ +0.5
  ขึ้น, ลง (แถว), ช่วงแถว = แถวสุดท้าย − แถวแรก + 1, โซนมือ
  ระยะมือ (มม.) = ค่าเฉลี่ยระยะของโซนมือ ; ขนาดจริง = โซนมือ × (ระยะ/300)²  (หน่วย "โซนที่ระยะ 300 มม.")
    — 1 โซนกว้างตามระยะ (มุมคงที่) พื้นที่จึงโตตามระยะยกกำลังสอง (ไอเดียของ Chin 20 ต.ค.)

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\thumb_per_take.py
ค่าเริ่มต้น: --users User5 User6 --stand-dirs data\\phase4 data\\phase4_extra --sit-dir data\\phase56 --out-dir results_thumb
ผลลัพธ์: results_thumb\\  report.txt, per_take.csv, summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase56 as e56     # noqa: E402  (load_takes — ตัวเดียวกับที่ใช้ประเมินทุกรอบ)
import posture_maps as pm           # noqa: E402  (extents — นิยามเดียวกับ posture_maps.py)

FOCUS = ("Like", "Dislike")
FORBIDDEN = ("phase56_stand", "phase57")
CLEAR = 0.5          # |ลง−ขึ้น| ≥ 0.5 แถว ถือว่า "ยื่นชัด" ไปฝั่งนั้น (ครึ่งโซน)
REF_MM = 300.0       # ระยะอ้างอิงของ "ขนาดจริง"
DIST_MEAN, DIST_STD = 295.0, 196.0   # ค่า normalise ระยะของ st_dataset (เหมือน recentre.py)


def take_stats(X: np.ndarray) -> dict | None:
    """ตัวเลขของ 1 รอบ; None ถ้าไม่มีเฟรมที่มีโซนมือ"""
    e = pm.extents(X)
    h = e["has"]
    if not h.any():
        return None
    d = (e["row_down"] - e["row_up"])[h]
    span = (e["row_down"] + e["row_up"])[h] + 1
    hand = X[h][..., 0] < pm.rc.HAND_THRESH                       # โซนมือของเฟรมที่มีมือ
    mm = X[h][..., 0] * DIST_STD + DIST_MEAN                       # กลับเป็นมิลลิเมตร
    dist = (np.where(hand, mm, 0).sum(axis=(1, 2)) / hand.sum(axis=(1, 2)))
    phys = e["zones"][h] * (dist / REF_MM) ** 2
    sd = lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")  # noqa: E731
    return {"frames": int(h.sum()), "down_minus_up": float(d.mean()), "down_minus_up_sd": sd(d),
            "pct_to_row0": float((d <= -CLEAR).mean() * 100), "pct_symmetric": float((np.abs(d) < CLEAR).mean() * 100),
            "pct_to_row7": float((d >= CLEAR).mean() * 100),
            "up_rows": float(e["row_up"][h].mean()), "down_rows": float(e["row_down"][h].mean()),
            "row_span": float(span.mean()), "zones": float(e["zones"][h].mean()),
            "dist_mm": float(dist.mean()), "dist_mm_sd": sd(dist), "phys_size": float(phys.mean())}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", nargs="+", default=["User5", "User6"])
    ap.add_argument("--stand-dirs", type=Path, nargs="+", default=[Path("data/phase4"), Path("data/phase4_extra")])
    ap.add_argument("--sit-dir", type=Path, default=Path("data/phase56"), help="ชุดนั่งของ User5 (ใส่ none เพื่อข้าม)")
    ap.add_argument("--out-dir", type=Path, default=Path("results_thumb"))
    args = ap.parse_args(argv)

    sources = [("stand/train", d) for d in args.stand_dirs]
    if str(args.sit_dir).lower() != "none":
        sources.append(("sit/phase56", args.sit_dir))
    for _, d in sources:
        if any(w in str(d).lower() for w in FORBIDDEN):
            raise SystemExit(f"{d}: this set must stay closed until the final model is chosen (§5.1y-A16)")
    st = e56.import_st_dataset()

    rows: list[dict] = []
    for cond, d in sources:
        for t in e56.load_takes(d, st, with_board=False):
            if t["user"] not in args.users or t["cls"] not in FOCUS:
                continue
            s = take_stats(t["X"])
            if s is None:
                continue
            rows.append({"user": t["user"], "condition": cond, "folder": d.name, "posture": t["cls"], "take": t["take"],
                         "file": t["file"].name, **s})
    if not rows:
        raise SystemExit("no Like/Dislike takes found for these users")
    rows.sort(key=lambda r: (r["user"], r["posture"], r["condition"], r["folder"], r["take"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "per_take.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

    # สรุปต่อกลุ่ม: ค่าเฉลี่ยของ "ค่าเฉลี่ยรายรอบ" + ช่วงต่ำสุด–สูงสุดระหว่างรอบ (บอกว่ารอบต่างกันมากไหม)
    summary: dict = {}
    for r in rows:
        summary.setdefault(f"{r['user']}|{r['posture']}|{r['condition']}", []).append(r)
    summ = {}
    for k, rs in summary.items():
        v = np.array([x["down_minus_up"] for x in rs])
        summ[k] = {"takes": len(rs), "mean_of_takes": float(v.mean()), "min_take": float(v.min()), "max_take": float(v.max()),
                   "takes_clear_to_row0": int((v <= -CLEAR).sum()), "takes_clear_to_row7": int((v >= CLEAR).sum()),
                   "row_span": float(np.mean([x["row_span"] for x in rs])), "zones": float(np.mean([x["zones"] for x in rs])),
                   "dist_mm": float(np.mean([x["dist_mm"] for x in rs])), "phys_size": float(np.mean([x["phys_size"] for x in rs]))}
    (args.out_dir / "summary.json").write_text(json.dumps({"groups": summ, "takes": rows}, indent=2, ensure_ascii=False, default=str),
                                               encoding="utf-8")

    L = ["=" * 104, "  Thumb per take · Like / Dislike · does the hand extend clearly toward row 0 (away) or row 7 (toward)? · §5.1y-O",
         "=" * 104, f"  down−up < 0 → extends toward row 0 ; 'clear' = |down−up| ≥ {CLEAR} row ; mean ± SD over frames of each take",
         f"  {'user':<7}{'posture':<9}{'condition':<13}{'folder':<14}{'take':>5}{'frames':>7}{'down−up':>14}{'%→0':>6}{'%sym':>6}"
         f"{'%→7':>6}{'up':>6}{'down':>6}{'span':>6}{'zones':>7}{'dist':>6}{'phys':>6}"]
    last = None
    for r in rows:
        key = (r["user"], r["posture"])
        if last is not None and key != last:
            L.append("")
        last = key
        L.append(f"  {r['user']:<7}{r['posture']:<9}{r['condition']:<13}{r['folder'][:13]:<14}{r['take']:>5}{r['frames']:>7}"
                 f"{r['down_minus_up']:>+8.2f} ±{r['down_minus_up_sd']:4.2f}{r['pct_to_row0']:>6.0f}{r['pct_symmetric']:>6.0f}"
                 f"{r['pct_to_row7']:>6.0f}{r['up_rows']:>6.2f}{r['down_rows']:>6.2f}{r['row_span']:>6.2f}{r['zones']:>7.1f}"
                 f"{r['dist_mm']:>6.0f}{r['phys_size']:>6.1f}")
    L += ["", "  Per group (take means): mean [min … max] · takes clear →0 / →7 · row span · zones · hand distance mm · size at 300 mm"]
    for k, s in summ.items():
        L.append(f"  {k:<32}{s['mean_of_takes']:>+6.2f} [{s['min_take']:+.2f} … {s['max_take']:+.2f}]  "
                 f"{s['takes_clear_to_row0']}/{s['takes']} →0, {s['takes_clear_to_row7']}/{s['takes']} →7   "
                 f"span {s['row_span']:.2f}  zones {s['zones']:.1f}  dist {s['dist_mm']:.0f}  phys {s['phys_size']:.1f}")
    L += ["", "  Reading: takes that differ a lot → (ก) some clear, some not ; long span both ways → (ข) thumb + wrist ;",
          "  short span both ways → (ค) thumb not resolved at 8×8. Descriptive only; same hand definition as recentre.py.",
          f"  output: {args.out_dir}", "=" * 104]
    text = "\n".join(L)
    (args.out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return summ


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. "ยื่น" วัดจากจุดศูนย์กลางของโซนมือ — ถ้าแขนกับนิ้วโป้งยื่นคนละฝั่งเท่าๆ กัน ค่าจะใกล้ 0 (ดูคอลัมน์ up/down/span ประกอบ)
#  2. ชุดนั่ง (phase56) ใช้แค่บรรยาย ไม่ได้ใช้เลือกโมเดล — ไม่ผิดกติกา A16
#  3. "ขนาดจริง" ประเมินต่ำไปถ้ามือล้นขอบภาพ (span = 8) — ส่วนที่หลุดภาพไม่ถูกนับ
#  4. ถ้าชื่อไฟล์ไม่มี __UserN__ / __takeN  load_takes จะ error (รูปแบบเดียวกับ evaluate_phase56)
# ─────────────────────────────────────────────────────────────────────────
