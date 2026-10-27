"""sit_vs_stand.py — มือของ User 5 ตอนนั่งต่างจากตอนยืน "ทุกท่า" หรือแค่บางท่า? (PROJECT_FACTS §5.1y-O)

Chin (27 ต.ค. 16:27): ยืนกับนั่ง ระดับแขนสูงต่ำต่างกัน ข้อมืออาจใกล้เซนเซอร์กว่าตอนนั่ง และตำแหน่งก็ไม่เท่ากันทั้งสองแบบ
→ เช็คทุกท่า (7 ท่า) ด้วยตัวเลขรายรอบ (take) เทียบ "ช่วงของรอบยืน" กับ "รอบนั่ง"

ข้อมูล: ยืน = data\\phase4 + data\\phase4_extra (ข้อมูลเทรน) · นั่ง = data\\phase56 (ชุดคัดเลือก ใช้บรรยายเท่านั้น)
ไม่เปิด data\\phase56_stand / phase57

ตัวเลขต่อรอบ (ค่าเฉลี่ยของเฟรมที่มีมือ):
  row, col   = จุดศูนย์กลางมือ (แถว 0 = ฝั่งไกลตัว, §5.1y-O)      → ตำแหน่ง
  top, bot   = แถวแรก / แถวสุดท้ายของมือ                        → ขอบบน/ล่าง
  rspan, cspan = ความยาวตามแถว / ความกว้างตามคอลัมน์ (โซน)       → รูปร่าง
  zones, dist (มม.), phys = โซนมือ × (ระยะ/300)²                → ขนาด / ระยะ
  tilt (มม.) = ระยะเฉลี่ยของโซนมือ "ฝั่งแถว 7 (ใกล้ตัว)" − "ฝั่งแถว 0 (ไกลตัว)" เทียบจุดศูนย์กลาง
               ติดลบ = ฝั่งใกล้ตัว (ข้อมือ) อยู่ใกล้เซนเซอร์กว่า → ทดสอบคำพูดของ Chin ตรงๆ
ตาราง: ค่ายืน (เฉลี่ยของรอบยืน) → ค่านั่ง (เฉลี่ยของ 2 รอบนั่ง) ; "*" = รอบนั่ง "ทุกรอบ" อยู่นอกช่วง [ต่ำสุด…สูงสุด]
ของรอบยืน และอยู่ฝั่งเดียวกัน (แปลว่าตอนนั่งออกนอกสิ่งที่โมเดลเคยเห็นตอนเทรน) — เชิงพรรณนา ไม่ใช่การทดสอบสถิติ (นั่งมีแค่ 2 รอบ/ท่า)

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\sit_vs_stand.py
ผลลัพธ์: results_sit_vs_stand\\  report.txt, per_take.csv, summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase56 as e56     # noqa: E402  (load_takes / CLASSES)
import recentre as rc               # noqa: E402  (HAND_THRESH — นิยามโซนมือเดียวกับทุกสคริปต์)

FORBIDDEN = ("phase56_stand", "phase57")
DIST_MEAN, DIST_STD = 295.0, 196.0   # ค่า normalise ระยะของ st_dataset (เหมือน recentre.py)
REF_MM = 300.0
METRICS = ["row", "col", "top", "bot", "rspan", "cspan", "zones", "dist", "phys", "tilt"]
FMT = {"row": ".2f", "col": ".2f", "top": ".2f", "bot": ".2f", "rspan": ".2f", "cspan": ".2f",
       "zones": ".1f", "dist": ".0f", "phys": ".1f", "tilt": "+.0f"}


def take_metrics(X: np.ndarray) -> dict | None:
    """ตัวเลขของ 1 รอบ (ค่าเฉลี่ยของเฟรมที่มีโซนมือ); None ถ้าไม่มีเฟรมที่มีมือ"""
    m = X[..., 0] < rc.HAND_THRESH
    has = m.any(axis=(1, 2))
    if not has.any():
        return None
    m = m[has]
    mm = X[has][..., 0] * DIST_STD + DIST_MEAN
    rows = np.broadcast_to(np.arange(8)[None, :, None], m.shape)
    cols = np.broadcast_to(np.arange(8)[None, None, :], m.shape)
    cnt = m.sum(axis=(1, 2))
    cr = (m * rows).sum(axis=(1, 2)) / cnt
    cc = (m * cols).sum(axis=(1, 2)) / cnt
    top = np.where(m, rows, 99).min(axis=(1, 2))
    bot = np.where(m, rows, -99).max(axis=(1, 2))
    left = np.where(m, cols, 99).min(axis=(1, 2))
    right = np.where(m, cols, -99).max(axis=(1, 2))
    dist = np.where(m, mm, 0).sum(axis=(1, 2)) / cnt
    near = m & (rows > cr[:, None, None])          # ฝั่งแถว 7 (ใกล้ตัว) ของจุดศูนย์กลาง
    far = m & (rows < cr[:, None, None])           # ฝั่งแถว 0 (ไกลตัว)
    with warnings.catch_warnings():                # เฟรมที่มีมือแถวเดียว → ฝั่งใดฝั่งหนึ่งว่าง → nan (ข้ามไป)
        warnings.simplefilter("ignore", RuntimeWarning)
        tilt = (np.where(near, mm, 0).sum(axis=(1, 2)) / near.sum(axis=(1, 2))
                - np.where(far, mm, 0).sum(axis=(1, 2)) / far.sum(axis=(1, 2)))
    vals = {"row": cr, "col": cc, "top": top, "bot": bot, "rspan": bot - top + 1, "cspan": right - left + 1,
            "zones": cnt, "dist": dist, "phys": cnt * (dist / REF_MM) ** 2, "tilt": tilt}
    out = {"frames": int(has.sum()), "tilt_frames": int(np.isfinite(tilt).sum())}
    for k, v in vals.items():
        v = np.asarray(v, dtype=float)
        out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out


def compare(stand: list[float], sit: list[float]) -> str:
    """'*' ถ้ารอบนั่งทุกรอบอยู่นอกช่วงของรอบยืน และอยู่ฝั่งเดียวกัน"""
    s, t = np.array(stand), np.array(sit)
    s, t = s[np.isfinite(s)], t[np.isfinite(t)]
    if len(s) == 0 or len(t) == 0:
        return "?"
    if (t > s.max()).all() or (t < s.min()).all():
        return "*"
    return ""


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default="User5")
    ap.add_argument("--stand-dirs", type=Path, nargs="+", default=[Path("data/phase4"), Path("data/phase4_extra")])
    ap.add_argument("--sit-dir", type=Path, default=Path("data/phase56"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_sit_vs_stand"))
    args = ap.parse_args(argv)
    for d in [args.sit_dir] + list(args.stand_dirs):
        if any(w in str(d).lower() for w in FORBIDDEN):
            raise SystemExit(f"{d}: this set must stay closed until the final model is chosen (§5.1y-A16)")
    st = e56.import_st_dataset()

    rows: list[dict] = []
    for cond, dirs in (("stand", args.stand_dirs), ("sit", [args.sit_dir])):
        for d in dirs:
            for t in e56.load_takes(d, st, with_board=False):
                if t["user"] != args.user:
                    continue
                s = take_metrics(t["X"])
                if s is not None:
                    rows.append({"condition": cond, "folder": d.name, "posture": t["cls"], "take": t["take"],
                                 "file": t["file"].name, **s})
    if not rows:
        raise SystemExit(f"no takes for {args.user}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "per_take.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

    res: dict = {}
    for c in e56.CLASSES:
        st_r = [r for r in rows if r["posture"] == c and r["condition"] == "stand"]
        si_r = [r for r in rows if r["posture"] == c and r["condition"] == "sit"]
        if not st_r or not si_r:
            continue
        res[c] = {"stand_takes": len(st_r), "sit_takes": len(si_r)}
        for k in METRICS:
            a, b = [r[k] for r in st_r], [r[k] for r in si_r]
            res[c][k] = {"stand_mean": float(np.nanmean(a)), "stand_min": float(np.nanmin(a)), "stand_max": float(np.nanmax(a)),
                         "sit": [float(x) for x in b], "sit_mean": float(np.nanmean(b)), "flag": compare(a, b)}
        res[c]["n_flags"] = sum(res[c][k]["flag"] == "*" for k in METRICS)
    (args.out_dir / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    W = 13
    L = ["=" * 150, f"  Sit vs stand · {args.user} · every posture · standing takes (training data) → sitting takes (phase56) · §5.1y-O",
         "=" * 150, "  cell = standing mean → sitting mean ; * = every sitting take outside the standing take range, same side",
         "  row/col = hand centre (row 0 = away from user) · top/bot = first/last hand row · rspan/cspan = length/width (zones)",
         "  phys = zones × (dist/300)² · tilt = distance of the hand's row-7 side (toward user) − row-0 side; negative = wrist side closer",
         "  " + f"{'posture':<11}{'takes':>6}" + "".join(f"{k:>{W}}" for k in METRICS) + f"{'#*':>4}"]
    for c, r in res.items():
        cells = ""
        for k in METRICS:
            v = r[k]
            cells += f"{format(v['stand_mean'], FMT[k])}→{format(v['sit_mean'], FMT[k])}{v['flag'] or ' '}".rjust(W)
        L.append(f"  {c:<11}{str(r['stand_takes']) + '/' + str(r['sit_takes']):>6}{cells}{r['n_flags']:>4}")
    L += ["", "  takes = standing/sitting. Descriptive only (2 sitting takes per posture; no test is possible). Same hand",
          "  definition as recentre.py. Errors of E5 on phase56 per posture: see §5.1q / results_a15.", f"  output: {args.out_dir}",
          "=" * 150]
    text = "\n".join(L)
    (args.out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return res


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. "*" ใช้ช่วงของรอบยืน (≈ 6 รอบ/ท่า) — ถ้ารอบยืนน้อย ช่วงจะแคบ และติด "*" ง่ายเกินจริง
#  2. tilt ใช้เฉพาะเฟรมที่มีมือทั้งสองฝั่งของจุดศูนย์กลาง (คอลัมน์ tilt_frames ใน CSV บอกจำนวน)
#  3. ระยะของ ToF = ระยะตามแนวลำแสงของแต่ละโซน ไม่ใช่ความสูงตั้งฉาก — โซนขอบภาพจะดูไกลกว่าจริงเล็กน้อย
#  4. ชุดนั่งใช้บรรยายเท่านั้น ไม่ได้ใช้เลือกโมเดล
# ─────────────────────────────────────────────────────────────────────────
