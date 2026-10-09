"""diagnose_position.py — หาสาเหตุที่ Love/Like ทายผิดใน Phase 4 (§5.1e-E) โดยไม่ต้องเก็บข้อมูลใหม่

คำถาม: มือเราอยู่ "คนละตำแหน่ง" กับมือในข้อมูล ST (ปัญหาตำแหน่ง) หรือภาพเรา "กลับด้าน"
เมื่อเทียบกับของ ST (ปัญหาการวางเซ็นเซอร์)?

ตรวจ 2 อย่าง:
  1. Footprint (รอยมือ) — แต่ละท่า: แผนที่ 8x8 ว่าโซนไหนเป็นมือบ่อยแค่ไหน, จุดศูนย์กลางมือ (แถว, คอลัมน์)
     เทียบ ST แยกทีละคน (User1-4, เห็นความแปรปรวนในกลุ่ม ST) กับของเรา
  2. Shift / flip test — เอาเฟรมของเรามาเลื่อน dx, dy = -2..+2 โซน และกลับภาพบน-ล่าง / ซ้าย-ขวา
     แล้วให้โมเดลตัวที่ deploy ทายใหม่ ดูว่าท่าไหนกลับมาถูกเมื่อแปลงแบบไหน
     โซนที่ว่างจากการเลื่อน = ฉากหลัง (ระยะ 4000, แสง 0 ก่อน normalize) แบบเดียวกับโซนที่ใช้ไม่ได้

ข้อควรระวังในการแปลผล (เขียนไว้ก่อนเห็นผล):
  - ข้อ 2 เป็นการสำรวจ (exploratory) ห้ามเลือก shift ที่ดีที่สุดแล้วรายงานเป็น "ความแม่นของระบบ"
  - คำทำนายก่อนรัน: ถ้าเป็นปัญหาตำแหน่ง → จุดศูนย์กลางมือ Love/Like ของเราจะอยู่นอกช่วงของ ST
    ทั้ง 4 คน และจะมี shift เล็กๆ (1-2 โซน) ที่ทำให้ Love/Like ถูกขึ้นมาก โดยท่าอื่นไม่พัง
    ถ้าเป็นปัญหาการวางเซ็นเซอร์ → footprint ของ "ทุกท่า" จะกลับด้านเหมือนกันหมด และ vflip/hflip
    จะช่วยหลายท่าพร้อมกัน — แต่ข้อนี้ขัดกับที่ 5 ท่าทายถูก 96-100 % อยู่แล้ว จึงคาดว่าไม่ใช่
  - n = 1 คน (User 5) — เป็นการวินิจฉัย ไม่ใช่ข้อสรุป

วิธีรัน (ที่ root ของ repo, ต้องมี evaluate_phase4.py และ st_dataset.py ใน pc_tools):
    python pc_tools\\diagnose_position.py --st-data <โฟลเดอร์ ST_VL53L8CX_handposture_dataset>
        --data-dir data\\phase4 --model results_phase2_final\\production_model.keras

ผลลัพธ์: results_phase4\\diag_<เวลา>\\ footprint.png, footprint.csv, shift_test.csv, shift_test.png, report.txt
ไม่มีการสุ่ม — รันซ้ำได้ผลเท่าเดิม
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase4 as ev  # noqa: E402  (ใช้ตัวโหลด/ตรวจป้าย/โหลดโมเดลชุดเดียวกับการประเมินผล)

BG_D = (4000.0 - 295.0) / 196.0      # ค่าระยะ "ฉากหลัง" หลัง normalize (เหมือน st_dataset)
BG_S = (0.0 - 281.0) / 452.0         # ค่าแสง "ฉากหลัง" หลัง normalize
FG_THRESHOLD = BG_D - 1e-3           # ต่ำกว่านี้ = โซนนั้นเป็นมือ
SHIFTS = range(-2, 3)
FOCUS = ["Love", "Like"]


def hand_mask(X: np.ndarray) -> np.ndarray:
    """(n,8,8) True = โซนที่เป็นมือ (ผ่านกฎ valid + ตัดฉากหลัง 120 มม. แล้ว)"""
    return X[..., 0] < FG_THRESHOLD


def footprint(X: np.ndarray) -> tuple[np.ndarray, tuple[float, float], float]:
    """คืน (แผนที่สัดส่วนเฟรมที่โซนนั้นเป็นมือ, จุดศูนย์กลาง (แถว, คอลัมน์), จำนวนโซนมือเฉลี่ย)"""
    m = hand_mask(X).astype(float)
    mean_map = m.mean(axis=0)
    total = mean_map.sum()
    rows, cols = np.mgrid[0:8, 0:8]
    if total == 0:
        return mean_map, (float("nan"), float("nan")), 0.0
    centroid = (float((mean_map * rows).sum() / total), float((mean_map * cols).sum() / total))
    return mean_map, centroid, float(m.sum(axis=(1, 2)).mean())


def shift_frames(X: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """เลื่อนภาพ dy แถว (+ = ลง) dx คอลัมน์ (+ = ขวา) ช่องที่ว่างเติมค่าฉากหลัง"""
    out = np.empty_like(X)
    out[..., 0], out[..., 1] = BG_D, BG_S
    src_r = slice(max(0, -dy), 8 - max(0, dy))
    dst_r = slice(max(0, dy), 8 - max(0, -dy))
    src_c = slice(max(0, -dx), 8 - max(0, dx))
    dst_c = slice(max(0, dx), 8 - max(0, -dx))
    out[:, dst_r, dst_c, :] = X[:, src_r, src_c, :]
    return out


def load_st_by_user_class(st_dir: Path, std) -> dict[tuple[str, str], np.ndarray]:
    """โหลดข้อมูล ST ด้วย st_dataset.load_dataset (ตัดคลาส None) แยกตาม (คน, ท่า) — map ด้วยชื่อ"""
    classes = [c for c in ev.MODEL_CLASSES if (st_dir / c).is_dir()]
    data = std.load_dataset(st_dir, class_names=classes, verbose=False)
    out: dict[tuple[str, str], np.ndarray] = {}
    for idx, name in enumerate(data.class_names):          # ลำดับของ st_dataset ≠ ของโมเดล (§9 #38)
        for user in sorted(set(data.groups.tolist())):
            sel = (data.y == idx) & (data.groups == user)
            if sel.any():
                out[(user, name)] = data.X[sel]
    return out


def main(argv: list[str] | None = None, predictor_factory=ev.load_keras_predictor) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st-data", type=Path, required=True, help="โฟลเดอร์ ST_VL53L8CX_handposture_dataset")
    ap.add_argument("--data-dir", type=Path, required=True, help=r"เช่น data\phase4")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--st-dataset", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results_phase4"))
    ap.add_argument("--user", default="User5")
    args = ap.parse_args(argv)
    if "extra" in args.data_dir.name.lower():
        raise SystemExit("use data\\phase4 (test set), not phase4_extra")
    for p in (args.st_data, args.data_dir):
        if not p.is_dir():
            raise SystemExit(f"folder not found: {p}")

    std = ev.import_module_from(ev.find_st_dataset(args.st_dataset), "st_dataset")
    print("Loading our takes ...")
    takes = [ev.load_take(f, std) for f in sorted(args.data_dir.glob("*/*/npz/*.npz"))]
    takes = [t for t in takes if t.user == args.user]
    if not takes:
        raise SystemExit(f"no takes for {args.user} in {args.data_dir}")
    ours = {c: np.concatenate([t.X for t in takes if t.cls == c]) for c in ev.MODEL_CLASSES
            if any(t.cls == c for t in takes)}
    print("Loading ST dataset (may take a minute) ...")
    st = load_st_by_user_class(args.st_data, std)
    st_users = sorted({u for u, _ in st}, key=lambda u: int(u[4:]))

    # ---------------- 1) footprint ----------------
    fp_rows = []
    for cls in ev.MODEL_CLASSES:
        for u in st_users + [args.user]:
            X = ours.get(cls) if u == args.user else st.get((u, cls))
            if X is None or not len(X):
                continue
            _, (r, c), nz = footprint(X)
            fp_rows.append({"class": cls, "source": u, "frames": len(X), "centroid_row": r,
                            "centroid_col": c, "mean_hand_zones": nz})

    # ---------------- 2) shift / flip test ----------------
    predict, n_out, _ = predictor_factory(args.model)
    if n_out != len(ev.MODEL_CLASSES):
        raise SystemExit(f"model has {n_out} outputs, expected 7")
    transforms = {f"shift dy={dy:+d} dx={dx:+d}": (lambda X, dy=dy, dx=dx: shift_frames(X, dy, dx))
                  for dy in SHIFTS for dx in SHIFTS}
    transforms["flip up-down"] = lambda X: X[:, ::-1, :, :]
    transforms["flip left-right"] = lambda X: X[:, :, ::-1, :]
    transforms["rotate 180"] = lambda X: X[:, ::-1, ::-1, :]
    shift_rows = []
    for name, fn in transforms.items():
        row = {"transform": name}
        correct = total = 0
        for cls, X in ours.items():
            pred = np.asarray(predict(fn(X))).argmax(axis=1)
            acc = float((pred == ev.MODEL_CLASSES.index(cls)).mean() * 100)
            row[cls] = acc
            correct += int((pred == ev.MODEL_CLASSES.index(cls)).sum())
            total += len(pred)
        row["overall"] = correct / total * 100
        shift_rows.append(row)
    base = next(r for r in shift_rows if r["transform"] == "shift dy=+0 dx=+0")

    # ---------------- รายงาน ----------------
    L = [ev.LINE, f"  Position / orientation diagnosis  ·  {args.user} vs ST users {', '.join(st_users)}", ev.LINE,
         "  1) Hand centroid (row, col) on the 8×8 grid — row 0 = top, col 0 = left as stored",
         f"     {'posture':<11}{'ST range of 4 users (row | col)':<36}{args.user:<16}outside ST range?"]
    for cls in ev.MODEL_CLASSES:
        s = [r for r in fp_rows if r["class"] == cls and r["source"] != args.user]
        o = [r for r in fp_rows if r["class"] == cls and r["source"] == args.user]
        if not s or not o:
            continue
        rr = [x["centroid_row"] for x in s]
        cc = [x["centroid_col"] for x in s]
        orow, ocol = o[0]["centroid_row"], o[0]["centroid_col"]
        out_r = not (min(rr) <= orow <= max(rr))
        out_c = not (min(cc) <= ocol <= max(cc))
        flag = ("row " if out_r else "") + ("col" if out_c else "")
        L.append(f"     {cls:<11}{min(rr):.2f}–{max(rr):.2f} | {min(cc):.2f}–{max(cc):.2f}"
                 f"{'':<12}{orow:.2f} | {ocol:.2f}    {flag or 'no'}")
    L.append("")
    L.append(f"  2) Accuracy of {args.user} frames after transforming them (exploratory)")
    L.append(f"     {'transform':<22}" + "".join(f"{c[:9]:>10}" for c in ev.MODEL_CLASSES) + f"{'overall':>10}")
    show = [base] + sorted([r for r in shift_rows if r is not base],
                           key=lambda r: -(r["Love"] + r["Like"]))[:6]
    show += [r for r in shift_rows if r["transform"].startswith(("flip", "rotate")) and r not in show]
    for r in show:
        L.append(f"     {r['transform']:<22}" + "".join(f"{r[c]:>10.1f}" for c in ev.MODEL_CLASSES)
                 + f"{r['overall']:>10.1f}")
    L.append("     (first row = no change; then the 6 transforms that help Love+Like most; then flips)")
    L.append(f"  n = 1 subject — diagnosis only. Best shift is chosen after seeing the data: do not report it as accuracy.")
    L.append(ev.LINE)
    report = "\n".join(L)
    print("\n" + report)

    out = args.out_dir / f"diag_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    with (out / "footprint.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fp_rows[0]))
        w.writeheader()
        w.writerows(fp_rows)
    with (out / "shift_test.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["transform"] + ev.MODEL_CLASSES + ["overall"])
        w.writeheader()
        w.writerows(shift_rows)
    made = save_figures(out, ours, st, st_users, shift_rows, args.user)
    print(f"\nSaved to {out.resolve()}\n  report.txt, footprint.csv, shift_test.csv"
          + (", footprint.png, shift_test.png" if made else "  (no PNG: matplotlib not installed)"))
    return {"footprint": fp_rows, "shift": shift_rows, "out": out}


def save_figures(out: Path, ours: dict, st: dict, st_users: list[str], shift_rows: list[dict],
                 user: str) -> bool:
    """footprint.png: แถว = ท่า, คอลัมน์ = ST แต่ละคน + ของเรา ; shift_test.png: Love/Like ตาม dx,dy"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    cols = st_users + [user]
    fig, axes = plt.subplots(len(ev.MODEL_CLASSES), len(cols), figsize=(1.9 * len(cols), 1.9 * 7), dpi=130)
    for i, cls in enumerate(ev.MODEL_CLASSES):
        for j, u in enumerate(cols):
            ax = axes[i, j]
            X = ours.get(cls) if u == user else st.get((u, cls))
            ax.set_xticks([])
            ax.set_yticks([])
            if X is None or not len(X):
                continue
            m, (r, c), _ = footprint(X)
            ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
            ax.plot(c, r, "r+", ms=9, mew=2)
            if i == 0:
                ax.set_title(u + (" (ours)" if u == user else ""), fontsize=9)
            if j == 0:
                ax.set_ylabel(cls, fontsize=9)
    fig.suptitle("Hand footprint: fraction of frames each zone is hand (red + = centroid)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "footprint.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(FOCUS) + 1, figsize=(4.0 * (len(FOCUS) + 1), 3.6), dpi=130)
    for ax, key in zip(axes, FOCUS + ["overall"]):
        grid = np.zeros((5, 5))
        for r in shift_rows:
            if r["transform"].startswith("shift"):
                dy = int(r["transform"].split("dy=")[1].split()[0])
                dx = int(r["transform"].split("dx=")[1])
                grid[dy + 2, dx + 2] = r[key]
        im = ax.imshow(grid, cmap="Greens", vmin=0, vmax=100)
        ax.set_xticks(range(5), [str(v) for v in SHIFTS])
        ax.set_yticks(range(5), [str(v) for v in SHIFTS])
        ax.set_xlabel("dx (zones, + = right)")
        ax.set_ylabel("dy (zones, + = down)")
        ax.set_title(f"{key} accuracy after shift (%)", fontsize=9)
        for a in range(5):
            for b in range(5):
                ax.text(b, a, f"{grid[a, b]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out / "shift_test.png", bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. --st-data ผิดโฟลเดอร์ → st_dataset หยุดพร้อมข้อความว่าไม่พบคลาส
#  2. การเลื่อนภาพบนตาราง 8x8 หยาบมาก (1 โซน = ช่องเซ็นเซอร์ทั้งช่อง) — บอกทิศทางได้ บอกระยะเป๊ะไม่ได้
#  3. "แถว 0 = บน" คือบนตามลำดับที่ข้อมูลถูกเก็บในไฟล์ ไม่จำเป็นต้องเป็นบนจริงในโลก
#     ใช้เทียบ ST กับเราได้เพราะทั้งคู่เก็บด้วยลำดับโซนของไดรเวอร์ ULD เดียวกัน [ยังไม่ได้ตรวจกับเฟิร์มแวร์]
#  4. shift ที่ดีที่สุดถูกเลือกหลังเห็นข้อมูล — ใช้ชี้ทาง ไม่ใช่ตัวเลขผลลัพธ์
# ─────────────────────────────────────────────────────────────────────────
