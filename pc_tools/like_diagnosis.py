"""like_diagnosis.py — วินิจฉัยว่าทำไม E5R ทาย Like ของ User 5 ตอนนั่งได้แค่ 52 % (PROJECT_FACTS §5.1y-R, §5.1y-A16)
ใช้ข้อมูลที่มีอยู่แล้วเท่านั้น — ไม่เปิด data\\phase56_stand (ชุดยืนต้องปิดไว้จนได้โมเดลตัวสุดท้าย)

ตอบ 4 คำถาม (ทั้งหมดเป็น "การวินิจฉัย" ไม่ใช่ตัวเลขความแม่นยำสำหรับรายงาน):
  ก. บั๊กไหม?  ให้ E5R ทายข้อมูล "ที่มันใช้เทรน" ของ User 5 (data\\phase4 + data\\phase4_extra)
               ถ้า Like/Dislike ตรงนั้นแม่นเกือบหมด → ระบบ (re-centre, ลำดับคลาส, การโหลด) ทำงานถูก
               ถ้าต่ำ → น่าจะมีบั๊ก (เพราะโมเดลเคยเห็นเฟรมพวกนี้ตอนเทรนแล้ว)
  ข. ทุกโมเดลพลาดเหมือนกันไหม?  โมเดลเดี่ยวสูตร R 5 ตัว และสูตร B 5 ตัว ทีละตัว บน phase56 Like/Dislike
  ค. รูปร่างมือ Like ตอนนั่งต่างจากตอนเทรนไหม?  ภาพเฉลี่ยของ "โซนที่เป็นมือ" หลัง re-centre
               Like/Dislike × ชุดเทรน (phase4 + phase4_extra) / ชุดนั่ง (phase56) + ค่าความคล้าย (correlation)
               และสัดส่วนเฟรม Like ตอนนั่งที่ "คล้าย Dislike ของชุดเทรน มากกว่า Like ของชุดเทรน"
  ง. รายเฟรม (CSV): E5 บนบอร์ด, E5R, ความน่าจะเป็น Like/Dislike, ตำแหน่งมือเดิม, ระยะเลื่อน, จำนวนโซนมือ

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\like_diagnosis.py
ค่าเริ่มต้น: --user User5 --train-dirs data\\phase4 data\\phase4_extra --sel-dir data\\phase56
            --e5 results_phase55_final\\deploy\\E5.keras --e5r results_phase55c_final\\deploy\\E5R.keras
            --r-models results_phase55c_final --b-models results_phase55_final --out-dir results_like_diag
ผลลัพธ์: results_like_diag\\  report.txt, summary.json, per_frame_phase56.csv, mean_hand_maps.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase56 as e56     # noqa: E402  (load_takes / predict_takes / sha256_file / CLASSES)
import recentre as rc               # noqa: E402

CLASSES = e56.CLASSES
FOCUS = ["Like", "Dislike"]
FORBIDDEN = ("phase56_stand", "phase57")       # ชุดที่ต้องปิดไว้ (§5.1y-A16)


def hand_mask(X: np.ndarray) -> np.ndarray:
    """โซนที่เป็นมือ (ระยะ < 2000 มม. ก่อน normalise) → (n, 8, 8) bool — นิยามเดียวกับ recentre.py"""
    return X[..., 0] < rc.HAND_THRESH


def raw_centroid(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """จุดศูนย์กลางมือ (แถว, คอลัมน์) ต่อเฟรม ก่อน re-centre"""
    m = hand_mask(X)
    cnt = np.maximum(m.sum(axis=(1, 2)), 1)
    r = (m * np.arange(8)[None, :, None]).sum(axis=(1, 2)) / cnt
    c = (m * np.arange(8)[None, None, :]).sum(axis=(1, 2)) / cnt
    return r, c


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """สหสัมพันธ์ของภาพ 8×8 สองภาพ (ใช้วัดว่ารูปร่างคล้ายกันแค่ไหน)"""
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def load_singles(models_dir: Path) -> list[tuple[str, Path]]:
    """อ่านโมเดลเดี่ยวจาก manifest.json พร้อมตรวจ sha256 (กันไฟล์ถูกสลับ/แก้)"""
    man = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    if man.get("smoke_test"):
        raise SystemExit(f"{models_dir}: manifest says smoke_test — not usable")
    if man.get("class_order") not in (None, CLASSES):
        raise SystemExit(f"{models_dir}: class order differs {man.get('class_order')}")
    out = []
    for m in sorted(man["models"], key=lambda m: int(m["seed"])):
        f = models_dir / m["model_file"]
        if e56.sha256_file(f) != m["model_sha256"]:
            raise SystemExit(f"sha256 mismatch: {f}")
        out.append((f"{man.get('arm', '?')}_seed{m['seed']}", f))
    return out


def recall_on(takes: list[dict], cls: dict[int, np.ndarray], names: list[str]) -> dict[str, float]:
    """recall (%) ของท่าใน names บนเฟรมในช่วงของ takes"""
    out = {}
    for c in names:
        k = CLASSES.index(c)
        seqs = [cls[i][t["mask"]] for i, t in enumerate(takes) if t["cls"] == c]
        out[c] = float((np.concatenate(seqs) == k).mean() * 100) if seqs else float("nan")
    return out


def main(argv: list[str] | None = None, load_model: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default="User5")
    ap.add_argument("--train-dirs", type=Path, nargs="+", default=[Path("data/phase4"), Path("data/phase4_extra")])
    ap.add_argument("--sel-dir", type=Path, default=Path("data/phase56"))
    ap.add_argument("--e5", type=Path, default=Path("results_phase55_final/deploy/E5.keras"))
    ap.add_argument("--e5r", type=Path, default=Path("results_phase55c_final/deploy/E5R.keras"))
    ap.add_argument("--r-models", type=Path, default=Path("results_phase55c_final"))
    ap.add_argument("--b-models", type=Path, default=Path("results_phase55_final"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_like_diag"))
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args(argv)

    for d in [args.sel_dir] + list(args.train_dirs):
        if any(w in str(d).lower() for w in FORBIDDEN):
            raise SystemExit(f"{d}: this set must stay closed until the final model is chosen (§5.1y-A16)")
    st = e56.import_st_dataset()
    if load_model is None:
        import tensorflow as tf

        def load_model(path: Path):  # noqa: E306
            mdl = tf.keras.models.load_model(str(path), compile=False)
            return lambda X: mdl.predict(X, batch_size=512, verbose=0)

    singles_r, singles_b = load_singles(args.r_models), load_singles(args.b_models)
    print("Loading data ...")
    sel = [t for t in e56.load_takes(args.sel_dir, st) if t["user"] == args.user]
    trn = [t for d in args.train_dirs for t in e56.load_takes(d, st, with_board=False) if t["user"] == args.user]
    if not sel or not trn:
        raise SystemExit(f"no takes for {args.user} (selection {len(sel)}, training {len(trn)})")
    print(f"  {args.user}: selection set {len(sel)} takes, training sets {len(trn)} takes")
    R = lambda f: (lambda X: f(rc.recentre(X)))   # noqa: E731  โมเดลสูตร R ต้องได้ข้อมูล re-centre เสมอ

    # ---- ก. บั๊กไหม: E5R / E5 บนข้อมูลเทรนของคนนี้ ----
    print("A. bug check on training data ...")
    m_e5r, m_e5 = load_model(args.e5r), load_model(args.e5)
    tr_r, _ = e56.predict_takes(trn, R(m_e5r))
    tr_b, _ = e56.predict_takes(trn, m_e5)
    bug = {"E5R_train_recall": recall_on(trn, tr_r, CLASSES), "E5_train_recall": recall_on(trn, tr_b, CLASSES)}

    # ---- ข. ทีละโมเดลบนชุดนั่ง ----
    print("B. every single model on the selection set ...")
    per_model = {}
    sel_r, sel_r_prob = e56.predict_takes(sel, R(m_e5r))
    per_model["E5R"] = recall_on(sel, sel_r, CLASSES)
    board = {i: np.where(t["board"]["cls"] == -9, -1, t["board"]["cls"]) for i, t in enumerate(sel)}
    per_model["E5_board"] = recall_on(sel, board, CLASSES)
    for name, f in singles_r:
        cls, _ = e56.predict_takes(sel, R(load_model(f)))
        per_model[name] = recall_on(sel, cls, CLASSES)
    for name, f in singles_b:
        cls, _ = e56.predict_takes(sel, load_model(f))
        per_model[name] = recall_on(sel, cls, CLASSES)

    # ---- ค. รูปร่างมือหลัง re-centre ----
    print("C. mean hand maps after re-centring ...")
    maps, shape = {}, {}
    for tag, takes in (("train", trn), ("sitting", sel)):
        for c in FOCUS:
            X = np.concatenate([t["X"] for t in takes if t["cls"] == c])
            maps[f"{c}|{tag}"] = hand_mask(rc.recentre(X)).mean(axis=0)
            shape[f"{c}|{tag}"] = {"frames": int(len(X)), "hand_zones_mean": float(hand_mask(X).sum(axis=(1, 2)).mean())}
    sims = {f"{a} vs {b}": corr(maps[a], maps[b]) for a, b in
            (("Like|sitting", "Like|train"), ("Like|sitting", "Dislike|train"),
             ("Dislike|sitting", "Dislike|train"), ("Dislike|sitting", "Like|train"), ("Like|train", "Dislike|train"))}
    Xl = rc.recentre(np.concatenate([t["X"] for t in sel if t["cls"] == "Like"]))
    hm = hand_mask(Xl).reshape(len(Xl), -1).astype(float)
    tl, td = maps["Like|train"].ravel(), maps["Dislike|train"].ravel()
    closer_dislike = float(np.mean([corr(h, td) > corr(h, tl) for h in hm]) * 100)

    # ---- ง. รายเฟรมของ Like/Dislike ชุดนั่ง ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, t in enumerate(sel):
        if t["cls"] not in FOCUS:
            continue
        r0, c0 = raw_centroid(t["X"])
        dr, dc, cnt = rc.shifts(t["X"])
        idx = np.nonzero(t["mask"])[0]
        for k, j in enumerate(idx):
            p = sel_r_prob[i][k]
            rows.append({"class": t["cls"], "take": t["take"], "frame_idx": int(j),
                         "E5_board": int(board[i][j]), "E5R": int(sel_r[i][j]),
                         "E5R_p_Like": round(float(p[3]), 4), "E5R_p_Dislike": round(float(p[2]), 4),
                         "centroid_row": round(float(r0[k]), 2), "centroid_col": round(float(c0[k]), 2),
                         "shift_dr": int(dr[k]), "shift_dc": int(dc[k]), "hand_zones": int(cnt[k])})
    with (args.out_dir / "per_frame_phase56.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    if not args.no_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = ["Like|train", "Like|sitting", "Dislike|train", "Dislike|sitting"]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), dpi=150)
        for ax, k in zip(axes, keys):
            im = ax.imshow(maps[k], vmin=0, vmax=1, cmap="viridis")
            c, tag = k.split("|")
            ax.set_title(f"{c} — {'training (standing)' if tag == 'train' else 'phase56 (sitting)'}\n"
                         f"n = {shape[k]['frames']} frames", fontsize=8)
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            ax.tick_params(labelsize=6)
            ax.set_xlabel("column", fontsize=7)
            ax.set_ylabel("row", fontsize=7)
        fig.colorbar(im, ax=axes, shrink=0.8, label="share of frames where the zone is hand")
        fig.suptitle(f"{args.user}: mean hand mask after re-centring", fontsize=9)
        fig.savefig(args.out_dir / "mean_hand_maps.png", bbox_inches="tight")
        plt.close(fig)

    res = {"user": args.user, "bug_check_training_data": bug, "selection_recall_per_model": per_model,
           "shape": shape, "map_correlations": sims, "sitting_like_frames_closer_to_training_dislike_pct": closer_dislike,
           "models": {"E5": e56.sha256_file(args.e5)[:16], "E5R": e56.sha256_file(args.e5r)[:16]}}
    (args.out_dir / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["=" * 78, f"  Like diagnosis  ·  {args.user}  ·  existing data only (phase56_stand NOT opened)  ·  §5.1y-A16", "=" * 78,
         "  A. bug check — models on the data they were TRAINED on (diagnostic, not accuracy)",
         "    recall %       " + "  ".join(f"{c[:6]:>6}" for c in CLASSES)]
    for k, lab in (("E5_train_recall", "E5"), ("E5R_train_recall", "E5R")):
        L.append(f"    {lab:<14} " + "  ".join(f"{bug[k][c]:6.1f}" for c in CLASSES))
    L += ["", "  B. every model on phase56 (sitting) — recall % Like / Dislike"]
    for name, rec in per_model.items():
        L.append(f"    {name:<14} Like {rec['Like']:6.1f}   Dislike {rec['Dislike']:6.1f}")
    L += ["", "  C. hand shape after re-centring (correlation of mean hand maps; 1 = identical)"]
    for k, v in sims.items():
        L.append(f"    {k:<36} {v:6.3f}")
    L.append(f"    sitting Like frames closer to training-Dislike than to training-Like: {closer_dislike:.1f} %")
    for k, v in shape.items():
        L.append(f"    {k:<18} frames {v['frames']:>5}   mean hand zones {v['hand_zones_mean']:5.1f}")
    L += ["", f"  D. per-frame table: {args.out_dir / 'per_frame_phase56.csv'}  ({len(rows)} Like/Dislike frames)",
          "  Training-data numbers are a bug check only — never report them as accuracy.", "=" * 78]
    text = "\n".join(L)
    (args.out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return res


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ตัวเลขข้อ ก. มาจากข้อมูลที่โมเดลเคยเห็นตอนเทรน — ใช้ตรวจบั๊กเท่านั้น ห้ามรายงานเป็นความแม่นยำ
#  2. ภาพเฉลี่ยข้อ ค. ใช้ "นิยามโซนมือ" เดียวกับ recentre.py (ระยะ < 2000 มม.) — เป็นรูปร่างที่โมเดล R เห็นจริง
#  3. ชุดนั่งมีท่าละ 2 รอบ — รูปแบบที่เห็นเป็นข้อสังเกต ไม่ใช่ข้อสรุปทางสถิติ
#  4. สคริปต์ไม่ยอมเปิด phase56_stand / phase57 (ต้องปิดไว้จนได้โมเดลตัวสุดท้าย)
# ─────────────────────────────────────────────────────────────────────────
