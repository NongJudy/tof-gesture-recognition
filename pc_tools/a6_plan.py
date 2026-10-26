"""a6_plan.py — A6: สร้างแผนการเลื่อนมือ (เป้าหมายตำแหน่งต่อรอบ) จากชุดฝึกของผู้ทดสอบเอง (PROJECT_FACTS §5.1s)

ทำครั้งเดียว "ก่อน" เก็บข้อมูล A6 แล้ว commit ไฟล์แผนไว้ → เป้าหมายถูกตรึงก่อนมีข้อมูล
  ตำแหน่งปกติของแต่ละท่า = centroid มือเฉลี่ยของคนนั้นในชุดฝึก (data\\phase4) วิธีเดียวกับ §5.1j
  5 ตำแหน่ง: natural, row+1, row-1, col+1, col-1   (เลื่อน 1 โซน ≈ 2.8 ซม. ที่ 270 มม.)
  2 รอบต่อตำแหน่ง → 10 รอบต่อท่า, ลำดับในแต่ละท่าสุ่มด้วย random.Random(seed + เลขท่า)

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\a6_plan.py --user 5
ผลลัพธ์: data\\a6_shift\\a6_plan_User5.json  (+ ตารางบนจอ)
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from evaluate_phase56 import CLASSES, centroid, import_st_dataset, load_takes

POSITIONS = {"natural": (0.0, 0.0), "row+1": (1.0, 0.0), "row-1": (-1.0, 0.0), "col+1": (0.0, 1.0), "col-1": (0.0, -1.0)}
REPEATS = 2
SEED = 66


def make_plan(natural: dict[str, tuple[float, float]], seed: int = SEED, repeats: int = REPEATS) -> dict:
    """แผนต่อท่า: รายการรอบ (ตำแหน่ง, เป้าแถว, เป้าคอลัมน์) เรียงตามลำดับที่สุ่มแล้ว"""
    plan = {}
    for ci, cls in enumerate(CLASSES):
        order = [p for p in POSITIONS for _ in range(repeats)]
        random.Random(seed + ci).shuffle(order)
        r0, c0 = natural[cls]
        takes = []
        for k, pos in enumerate(order, start=1):
            dr, dc = POSITIONS[pos]
            tr, tc = r0 + dr, c0 + dc
            if not (0.5 <= tr <= 6.5 and 0.5 <= tc <= 6.5):
                raise SystemExit(f"{cls} {pos}: target ({tr:.2f}, {tc:.2f}) too close to the edge of the 8x8 grid")
            takes.append({"take": k, "position": pos, "target_row": round(tr, 2), "target_col": round(tc, 2)})
        plan[cls] = {"natural_row": round(r0, 2), "natural_col": round(c0, 2), "takes": takes}
    return plan


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--ref-dir", type=Path, default=Path("data/phase4"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    st = import_st_dataset()
    user = f"User{args.user}"
    ref = [t for t in load_takes(args.ref_dir, st, with_board=False) if t["user"] == user]
    natural = {}
    for cls in CLASSES:
        masks = [t["hand"] for t in ref if t["cls"] == cls and len(t["hand"])]
        if not masks:
            raise SystemExit(f"no {cls} takes for {user} in {args.ref_dir}")
        natural[cls] = centroid(masks)
    plan = {"generated": datetime.now().isoformat(timespec="seconds"), "user": args.user, "ref_dir": str(args.ref_dir),
            "seed": SEED, "repeats": REPEATS, "positions": POSITIONS, "classes": make_plan(natural)}
    out = args.out or Path("data/a6_shift") / f"a6_plan_{user}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} already exists — the plan is frozen; delete it only if no A6 data has been collected")
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"A6 plan for {user}  (natural = centroid in {args.ref_dir}; shift = 1 zone)")
    print(f"{'posture':<11} {'natural r,c':>12}   take order")
    for cls, p in plan["classes"].items():
        print(f"{cls:<11} {p['natural_row']:>5.2f},{p['natural_col']:>5.2f}   "
              + " ".join(t["position"] for t in p["takes"]))
    print(f"saved → {out}")
    return out


if __name__ == "__main__":
    main()
