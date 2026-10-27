"""posture_maps.py — ตรวจ "แขนอยู่ฝั่งไหนของภาพ" ในทุกท่า ก่อนออกแบบ arm-dropout (PROJECT_FACTS §5.1y-D)
ใช้เฉพาะข้อมูลเทรน: ST 4 คน + data\\phase4 + data\\phase4_extra (ไม่เปิด phase56 / phase56_stand / a5_love)

ตอบ 2 คำถาม (ข้อมูลเชิงพรรณนา):
  1. แต่ละท่า (7 ท่า) × แต่ละกลุ่ม (ST 4 คน, User5, User6) มือ "ยื่นยาว" ไปทางแถวสูง (แถว 7) หรือแถวต่ำ (แถว 0) มากกว่ากัน?
       ความยาวลง  = แถวสุดท้ายของมือ − แถวจุดศูนย์กลาง   (ยื่นไปทางแถว 7)
       ความยาวขึ้น = แถวจุดศูนย์กลาง − แถวแรกของมือ      (ยื่นไปทางแถว 0)
       ส่วนต่าง (ลง − ขึ้น) > 0 = มือ/แขนยื่นไปทางแถว 7 → ถ้าเป็นบวกทุกกลุ่ม แขนน่าจะอยู่ฝั่งแถวสูงจริง
     ทำแบบเดียวกันกับคอลัมน์ (ขวา − ซ้าย) เพื่อดูว่าแขนเฉียงไปข้างไหม
  2. ภาพเฉลี่ยของ "โซนที่เป็นมือ" ทุกท่า (ก่อน re-centre = ตำแหน่งจริงในภาพ) → mean_maps_all_postures.png
     ใช้ดูด้วยตาว่าท่าไหนมี "หาง" (แขน) และท่าสองมือใช้แขนเป็นส่วนของรูปร่างไหม

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\posture_maps.py --st-data C:\\ds\\ST_VL53L8CX_handposture_dataset --ours data\\phase4 data\\phase4_extra
ผลลัพธ์: results_posture_maps\\  report.txt, summary.json, mean_maps_all_postures.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase4 as ev        # noqa: E402
import train_phase55 as p55         # noqa: E402  (load_all — ตัวเดียวกับที่ใช้เทรนทุกรอบ)
import recentre as rc               # noqa: E402

FORBIDDEN = ("phase56", "phase57", "a5_love", "a6_", "rejected", "dryrun")
GROUPS = [("ST users 1-4", (1, 2, 3, 4)), ("User5 (Chin)", (5,)), ("User6 (friend)", (6,))]


def extents(X: np.ndarray) -> dict[str, np.ndarray]:
    """ต่อเฟรม: จุดศูนย์กลาง + ระยะที่มือยื่นไปแต่ละทิศ (หน่วย = แถว/คอลัมน์)"""
    m = rc.HAND_THRESH > X[..., 0]
    has = m.any(axis=(1, 2))
    rows = np.arange(8)[None, :, None] * np.ones((1, 1, 8))
    cols = np.arange(8)[None, None, :] * np.ones((1, 8, 1))
    cnt = np.maximum(m.sum(axis=(1, 2)), 1)
    cr = (m * rows).sum(axis=(1, 2)) / cnt
    cc = (m * cols).sum(axis=(1, 2)) / cnt
    big = 99
    rmin = np.where(m, rows, big).min(axis=(1, 2))
    rmax = np.where(m, rows, -big).max(axis=(1, 2))
    cmin = np.where(m, cols, big).min(axis=(1, 2))
    cmax = np.where(m, cols, -big).max(axis=(1, 2))
    return {"has": has, "row_down": rmax - cr, "row_up": cr - rmin, "col_right": cmax - cc, "col_left": cc - cmin,
            "zones": m.sum(axis=(1, 2))}


def summarise(X: np.ndarray) -> dict:
    """ค่าเฉลี่ย ± SD ของความยาวแต่ละทิศ และส่วนต่าง (ลง − ขึ้น), (ขวา − ซ้าย)"""
    e = extents(X)
    h = e["has"]
    d_row = (e["row_down"] - e["row_up"])[h]
    d_col = (e["col_right"] - e["col_left"])[h]
    f = lambda a: (float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else float("nan"))  # noqa: E731
    return {"frames": int(h.sum()), "row_down": f(e["row_down"][h]), "row_up": f(e["row_up"][h]),
            "row_down_minus_up": f(d_row), "share_frames_down_longer": float((d_row > 0).mean() * 100),
            "col_right_minus_left": f(d_col), "hand_zones": f(e["zones"][h])}


def main(argv: list[str] | None = None, data: dict | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st-data", type=Path, required=True)
    ap.add_argument("--ours", type=Path, nargs="+", required=True, help=r"data\phase4 data\phase4_extra")
    ap.add_argument("--train-script", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results_posture_maps"))
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args(argv)
    for d in args.ours:
        if any(w in str(d).lower() for w in FORBIDDEN):
            raise SystemExit(f"--ours {d}: test / held-out set — this check uses training data only")

    if data is None:
        here = Path(__file__).resolve().parent
        tr_path = args.train_script or next((p for p in (here / "train_st_cnn2d.py", here.parent / "train_st_cnn2d.py")
                                             if p.is_file()), None)
        if tr_path is None:
            raise SystemExit("train_st_cnn2d.py not found — use --train-script <path>")
        tr = ev.import_module_from(Path(tr_path), "train_st_cnn2d")
        if list(tr.CLASSES) != ev.MODEL_CLASSES:
            raise SystemExit(f"class order differs: {tr.CLASSES}")
        print("Loading ST + our training data (train_phase55.load_all) ...")
        data = p55.load_all(tr, args.st_data, args.ours)
    classes = ev.MODEL_CLASSES
    X, y, u = data["X"], data["y"], data["user"]
    print(f"  frames {len(y):,}; users {sorted(set(u.tolist()))}")

    res: dict = {}
    maps: dict = {}
    for gname, users in GROUPS:
        gm = np.isin(u, users)
        for k, c in enumerate(classes):
            sel = gm & (y == k)
            if not sel.any():
                continue
            res[f"{gname}|{c}"] = summarise(X[sel])
            maps[(gname, c)] = (rc.HAND_THRESH > X[sel][..., 0]).mean(axis=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.no_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(GROUPS), len(classes), figsize=(15, 6.6), dpi=150)
        im = None
        for gi, (gname, _) in enumerate(GROUPS):
            for ci, c in enumerate(classes):
                ax = axes[gi, ci]
                if (gname, c) in maps:
                    im = ax.imshow(maps[(gname, c)], vmin=0, vmax=1, cmap="viridis")
                    ax.set_title(f"{c}\nn = {res[f'{gname}|{c}']['frames']}", fontsize=7)
                ax.set_xticks([0, 7])
                ax.set_yticks([0, 7])
                ax.tick_params(labelsize=6)
                if ci == 0:
                    ax.set_ylabel(f"{gname}\nrow", fontsize=7)
        if im is not None:
            fig.colorbar(im, ax=axes, shrink=0.7, label="share of frames where the zone is hand")
        fig.suptitle("Mean hand mask BEFORE re-centring (training data only) — where does each posture extend?", fontsize=9)
        fig.savefig(args.out_dir / "mean_maps_all_postures.png", bbox_inches="tight")
        plt.close(fig)

    L = ["=" * 92, "  Posture maps · where does the hand/forearm extend? · training data only · §5.1y-D", "=" * 92,
         "  down−up > 0  → hand extends further toward row 7 than toward row 0 (mean ± SD, rows)",
         "  right−left > 0 → extends further toward column 7",
         f"  {'group':<16}{'posture':<12}{'frames':>7}{'down−up':>15}{'% down>up':>11}{'right−left':>15}{'zones':>9}"]
    for gname, _ in GROUPS:
        for c in classes:
            r = res.get(f"{gname}|{c}")
            if r is None:
                continue
            L.append(f"  {gname:<16}{c:<12}{r['frames']:>7}{r['row_down_minus_up'][0]:>+8.2f} ±{r['row_down_minus_up'][1]:4.2f}"
                     f"{r['share_frames_down_longer']:>10.0f}%{r['col_right_minus_left'][0]:>+8.2f} ±{r['col_right_minus_left'][1]:4.2f}"
                     f"{r['hand_zones'][0]:>9.1f}")
        L.append("")
    L += ["  Descriptive only. Uses the same hand definition as recentre.py (distance < 2000 mm after preprocessing).",
          f"  output: {args.out_dir}", "=" * 92]
    text = "\n".join(L)
    (args.out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return res


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ตัวเลขนี้ "บอกทิศที่มือยื่นไป" ไม่ได้บอกตรงๆ ว่าส่วนที่ยื่นคือแขน — ต้องดูประกอบกับภาพเฉลี่ย
#  2. ใช้ข้อมูลเทรนเท่านั้น (ไม่แตะชุดทดสอบ) — ปลอดภัยต่อกติกา §5.1y
#  3. ST ไม่มีข้อมูลว่าแต่ละคนยืนฝั่งไหนของบอร์ด — ถ้าทิศของ ST ต่างจากเรา arm-dropout ต้องคิดใหม่
# ─────────────────────────────────────────────────────────────────────────
