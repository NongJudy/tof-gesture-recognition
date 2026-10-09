"""phase55a_shift_aug.py — Phase 5.5a: สุ่มเลื่อนภาพตอนเทรน "อย่างเดียว" ช่วยให้ทายมือ User 5 ถูกขึ้นไหม

Pre-registration: PROJECT_FACTS §5.1g (เขียนก่อนรัน 10 ต.ค. 2026) — ห้ามเปลี่ยนค่าตั้งต้นหลังเห็นผล
  แขน baseline : เทรนบนข้อมูล ST 4 คน + สุ่มกลับซ้าย-ขวา (เหมือน train_st_cnn2d.py)
  แขน shift    : เหมือน baseline + สุ่มเลื่อนภาพ dy, dx ∈ {-2..+2} โซน (ช่องว่างเติมค่าฉากหลัง)
  ทั้งสองแขน  : ข้อมูลเทรนคัดลอก K=5 ชุด (แต่ละชุดสุ่มใหม่) → จำนวนตัวอย่างต่อ epoch เท่ากัน
               validation = 10 % ของเฟรม ST ชุดเดียวกัน (สุ่มด้วย seed) ไม่ augment ใช้แค่ early stopping
               seeds 42, 43, 44 → แขนละ 3 โมเดล รวม 6 โมเดล
  ทดสอบ      : data\\phase4 (User 5) เตรียมข้อมูลด้วย st_dataset แบบเดียวกับ evaluate_phase4.py
               ข้อมูลทดสอบไม่ถูกใช้เทรน / early stopping / เลือกโมเดล เด็ดขาด

ใช้โค้ดของจริงทั้งหมด (ไม่เขียนซ้ำ เพื่อให้เหมือนโมเดลที่ deploy):
  - train_st_cnn2d.load_dataset()  → เตรียมข้อมูล ST แบบเดียวกับตอนเทรนโมเดลจริง
  - train_st_cnn2d.build_model()   → สถาปัตยกรรม + optimizer เดียวกัน
  - evaluate_phase4.load_take() / subject_metrics() → วัดผลแบบเดียวกับ §5.1e-E

วิธีรัน (ที่ root ของ repo; ใช้เวลานาน — CPU อย่างเดียว ประมาณ 30-90 นาที ขึ้นกับเครื่อง [ยังไม่ได้วัด]):
    python pc_tools\\phase55a_shift_aug.py --st-data C:\\ds\\ST_VL53L8CX_handposture_dataset
        --data-dir data\\phase4

ผลลัพธ์: results_phase55a\\run_<เวลา>\\ report.txt, summary.json, per_model.csv, models\\*.keras
รันซ้ำด้วย seed เดิมควรได้ผลใกล้เดิม (TensorFlow บน CPU อาจต่างเล็กน้อยจากลำดับการคำนวณ — oneDNN)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase4 as ev  # noqa: E402

BG_D = (4000.0 - 295.0) / 196.0      # ค่าระยะฉากหลังหลัง normalize (โซนที่ใช้ไม่ได้)
BG_S = (0.0 - 281.0) / 452.0         # ค่าแสงฉากหลังหลัง normalize
FOCUS = ["Like", "Love", "Dislike"]
# เกณฑ์จาก pre-registration §5.1g — ห้ามแก้หลังเห็นผล
P4_LIKE_MIN = 30.0
P5_LOVE_MIN = 30.0
P6_DISLIKE_MIN = 70.0
P7_GAIN_MIN = 5.0
S1_DEPLOYED = 66.3
S1_TOL = 10.0


def find_file(name: str, explicit: Path | None) -> Path:
    """หาไฟล์ .py ของโปรเจกต์ (ใช้ path ที่ระบุ หรือค้นใน pc_tools / root ของ repo)"""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"file not found: {explicit}")
        return explicit
    here = Path(__file__).resolve().parent
    for cand in (here / name, here.parent / name):
        if cand.is_file():
            return cand
    found = [c for c in sorted(here.parent.rglob(name)) if ".git" not in c.parts]
    if found:
        return found[0]
    raise SystemExit(f"{name} not found — locate it with:  dir /s /b {name}")


def make_copies(X: np.ndarray, y: np.ndarray, k: int, max_shift: int, use_shift: bool,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """คัดลอกข้อมูลเทรน k ชุด แต่ละชุด: สุ่มกลับซ้าย-ขวา (p=0.5) และถ้า use_shift สุ่มเลื่อน dy,dx
    ทั้งสองแขนใช้ rng ลำดับเดียวกันสำหรับการกลับภาพ → ต่างกันแค่การเลื่อนอย่างเดียว"""
    rng_flip = np.random.default_rng(seed)
    rng_shift = np.random.default_rng(seed + 10_000)
    xs, ys = [], []
    for _ in range(k):
        xc = X.copy()
        flip = rng_flip.random(len(X)) < 0.5
        xc[flip] = xc[flip, :, ::-1, :]
        if use_shift:
            dys = rng_shift.integers(-max_shift, max_shift + 1, len(X))
            dxs = rng_shift.integers(-max_shift, max_shift + 1, len(X))
            xc = shift_each(xc, dys, dxs)
        xs.append(xc)
        ys.append(y)
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys)


def shift_each(X: np.ndarray, dys: np.ndarray, dxs: np.ndarray) -> np.ndarray:
    """เลื่อนแต่ละเฟรมตาม dy (+ = ลง), dx (+ = ขวา) ของตัวเอง ช่องที่ว่างเติมค่าฉากหลัง"""
    pad = int(max(np.abs(dys).max(initial=0), np.abs(dxs).max(initial=0)))
    if pad == 0:
        return X.copy()
    P = np.empty((len(X), 8 + 2 * pad, 8 + 2 * pad, 2), dtype=X.dtype)
    P[..., 0], P[..., 1] = BG_D, BG_S
    P[:, pad:pad + 8, pad:pad + 8, :] = X
    out = np.empty_like(X)
    for dy in np.unique(dys):
        for dx in np.unique(dxs):
            sel = (dys == dy) & (dxs == dx)
            if sel.any():
                r0, c0 = pad - dy, pad - dx
                out[sel] = P[sel, r0:r0 + 8, c0:c0 + 8, :]
    return out


def keras_trainer(tr_mod) -> Callable:
    """คืนฟังก์ชันเทรน 1 โมเดลด้วย build_model ของ train_st_cnn2d (ตั้งค่าเหมือน production)"""
    import tensorflow as tf

    class Progress(tf.keras.callbacks.Callback):
        """พิมพ์ความคืบหน้าบรรทัดเดียวทุก 5 epoch"""
        def __init__(self, label: str):
            super().__init__()
            self.label, self.t0 = label, time.time()

        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:
                logs = logs or {}
                print(f"\r  {self.label}: epoch {epoch + 1:4d}  val_loss {logs.get('val_loss', float('nan')):.4f}"
                      f"  ({time.time() - self.t0:5.0f} s)   ", end="", flush=True)

    def train(X_tr, y_tr, X_val, y_val, seed: int, label: str, save_to: Path):
        model = tr_mod.build_model(seed)        # set_random_seed อยู่ใน build_model แล้ว
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
        hist = model.fit(X_tr, y_tr, batch_size=32, epochs=1000, validation_data=(X_val, y_val),
                         shuffle=True, callbacks=[stop, Progress(label)], verbose=0)
        print()
        save_to.parent.mkdir(parents=True, exist_ok=True)
        model.save(save_to)
        return (lambda X: model.predict(X, batch_size=256, verbose=0)), len(hist.history["loss"]), tf.__version__
    return train


def summarise_arm(rows: list[dict]) -> dict:
    """ค่าเฉลี่ย ± SD ข้าม seed ของแต่ละตัววัด"""
    keys = ["frame_acc", "vote_acc"] + [f"recall_{c}" for c in ev.MODEL_CLASSES]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                  "values": v.tolist()}
    return out


def main(argv: list[str] | None = None, trainer_factory: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st-data", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help=r"data\phase4 (ชุดทดสอบ)")
    ap.add_argument("--user", default="User5")
    ap.add_argument("--train-script", type=Path, default=None, help="path ของ train_st_cnn2d.py (ปกติหาเอง)")
    ap.add_argument("--st-dataset", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results_phase55a"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--max-shift", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args(argv)
    if "extra" in args.data_dir.name.lower():
        raise SystemExit("test set must be data\\phase4, not phase4_extra")
    for p in (args.st_data, args.data_dir):
        if not p.is_dir():
            raise SystemExit(f"folder not found: {p}")
    registered = (args.seeds == [42, 43, 44] and args.copies == 5 and args.max_shift == 2 and args.val_frac == 0.1)

    tr_path = find_file("train_st_cnn2d.py", args.train_script)
    tr = ev.import_module_from(tr_path, "train_st_cnn2d")
    if list(tr.CLASSES) != ev.MODEL_CLASSES:
        raise SystemExit(f"class order in {tr_path} differs from the deployed model: {tr.CLASSES}")
    std = ev.import_module_from(ev.find_st_dataset(args.st_dataset), "st_dataset")

    print("Loading ST training data with train_st_cnn2d.load_dataset() ...")
    ds = tr.load_dataset(args.st_data)
    X_all, y_all = ds.X.astype(np.float32), ds.y
    print(f"  {len(X_all):,} frames, users {sorted(np.unique(ds.groups).tolist())}")
    rng = np.random.default_rng(42)                      # validation split เดียวกันทุกแขน/ทุก seed
    perm = rng.permutation(len(X_all))
    n_val = int(round(len(X_all) * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    print(f"Loading test set {args.data_dir} ({args.user}) ...")
    takes = [ev.load_take(f, std) for f in sorted(args.data_dir.glob("*/*/npz/*.npz"))]
    takes = [t for t in takes if t.user == args.user]
    if not takes:
        raise SystemExit(f"no takes for {args.user}")

    trainer = (trainer_factory or keras_trainer)(tr)
    out = args.out_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    rows: list[dict] = []
    t_start = time.time()
    plan = [(arm, s) for s in args.seeds for arm in ("baseline", "shift")]
    for i, (arm, seed) in enumerate(plan, 1):
        X_tr, y_tr = make_copies(X_all[tr_idx], y_all[tr_idx], args.copies, args.max_shift,
                                 use_shift=(arm == "shift"), seed=seed)
        label = f"[{i}/{len(plan)}] {arm:<8} seed {seed}"
        t0 = time.time()
        predict, epochs, tf_version = trainer(X_tr, y_tr, X_all[val_idx], y_all[val_idx], seed, label,
                                              out / "models" / f"{arm}_seed{seed}.keras")
        for t in takes:
            prob = np.asarray(predict(t.X)) if len(t.X) else np.zeros((0, 7))
            t.pred, t.conf = prob.argmax(axis=1), prob.max(axis=1)
        m = ev.subject_metrics(takes)
        row = {"arm": arm, "seed": seed, "epochs": epochs, "train_s": round(time.time() - t0, 1),
               "frame_acc": m["frame_acc"], "vote_acc": m["vote_acc"],
               **{f"recall_{c}": m["recall"][c] for c in ev.MODEL_CLASSES},
               "top_confusion": (f"{m['top_confusions'][0]['true']}->{m['top_confusions'][0]['pred']}"
                                 if m["top_confusions"] else "")}
        rows.append(row)
        done = time.time() - t_start
        print(f"  → {args.user} accuracy {m['frame_acc']:.1f} %  ·  Like {m['recall']['Like']:.1f}  "
              f"Love {m['recall']['Love']:.1f}  Dislike {m['recall']['Dislike']:.1f}  ·  "
              f"elapsed {done / 60:.0f} min, about {done / i * (len(plan) - i) / 60:.0f} min left")

    base = summarise_arm([r for r in rows if r["arm"] == "baseline"])
    shift = summarise_arm([r for r in rows if r["arm"] == "shift"])
    gain = shift["frame_acc"]["mean"] - base["frame_acc"]["mean"]
    checks = {
        "S1_baseline_near_deployed": abs(base["frame_acc"]["mean"] - S1_DEPLOYED) <= S1_TOL,
        "P4_like": shift["recall_Like"]["mean"] > P4_LIKE_MIN,
        "P5_love": shift["recall_Love"]["mean"] > P5_LOVE_MIN,
        "P6_dislike": shift["recall_Dislike"]["mean"] >= P6_DISLIKE_MIN,
        "P7_gain": gain >= P7_GAIN_MIN,
    }

    def ms(d: dict) -> str:
        return f"{d['mean']:5.1f} ± {d['sd']:4.1f}"
    L = [ev.LINE, f"  Phase 5.5a  ·  shift augmentation, trained on ST only, tested on {args.user}", ev.LINE,
         f"  Training   ST {len(tr_idx):,} frames × {args.copies} copies  ·  val {n_val:,} frames  ·  "
         f"seeds {args.seeds}  ·  max shift ±{args.max_shift}",
         f"  Test       {args.data_dir}  ·  {sum(len(t.X) for t in takes):,} frames  ·  never used for training",
         f"  Settings   {'as pre-registered (§5.1g)' if registered else 'CHANGED from pre-registration — exploratory only'}",
         "",
         f"  {'metric (mean ± SD over seeds)':<30}{'baseline':>16}{'shift':>16}",
         f"  {'per-frame accuracy':<30}{ms(base['frame_acc']):>16}{ms(shift['frame_acc']):>16}",
         f"  {'15-frame vote accuracy':<30}{ms(base['vote_acc']):>16}{ms(shift['vote_acc']):>16}"]
    for c in ev.MODEL_CLASSES:
        L.append(f"  {'recall ' + c:<30}{ms(base['recall_' + c]):>16}{ms(shift['recall_' + c]):>16}")
    L += ["",
          "  Pre-registered checks (§5.1g)",
          f"    S1  baseline {base['frame_acc']['mean']:.1f} % within ±{S1_TOL:.0f} of deployed {S1_DEPLOYED} %  →  "
          f"{'YES' if checks['S1_baseline_near_deployed'] else 'NO — explain before using the rest'}",
          f"    P4  shift Like recall {shift['recall_Like']['mean']:.1f} % > {P4_LIKE_MIN:.0f} %  →  "
          f"{'HOLDS' if checks['P4_like'] else 'DOES NOT HOLD'}",
          f"    P5  shift Love recall {shift['recall_Love']['mean']:.1f} % > {P5_LOVE_MIN:.0f} %  →  "
          f"{'HOLDS' if checks['P5_love'] else 'DOES NOT HOLD'}",
          f"    P6  shift Dislike recall {shift['recall_Dislike']['mean']:.1f} % ≥ {P6_DISLIKE_MIN:.0f} %  →  "
          f"{'HOLDS' if checks['P6_dislike'] else 'DOES NOT HOLD'}",
          f"    P7  gain {gain:+.1f} pp ≥ +{P7_GAIN_MIN:.0f} pp  →  {'HOLDS' if checks['P7_gain'] else 'DOES NOT HOLD'}",
          "",
          f"  n = 1 test subject; SD = training randomness only (3 seeds), not person-to-person spread.",
          ev.LINE]
    report = "\n".join(L)
    print("\n" + report)

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    with (out / "per_model.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {"run_at": datetime.now().isoformat(timespec="seconds"), "registered_settings": registered,
            "args": {k: str(v) for k, v in vars(args).items()}, "train_script": str(tr_path),
            "train_script_sha256": ev.sha256_file(tr_path), "n_train_frames": int(len(tr_idx)),
            "n_val_frames": n_val, "test_data_sha256": ev.data_checksum(
                [t.path for t in takes], args.data_dir), "git_commit": ev.git_commit(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "tensorflow": tf_version}
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "per_model": rows, "baseline": base, "shift": shift,
                   "gain_pp": gain, "checks": checks}, fh, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out.resolve()}")
    return {"rows": rows, "checks": checks, "out": out, "base": base, "shift": shift}


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ใช้เวลานานบน CPU — ปล่อยรันได้ ปิดหน้าต่าง cmd = หยุด ; โมเดลที่เสร็จแล้วถูกเซฟไว้ใน models\
#  2. ถ้า S1 ไม่ผ่าน (baseline ห่างจาก 66.3 % เกิน 10 pp) แปลว่าวิธีเทรนนี้ต่างจาก production
#     อย่างมีผล — ต้องอธิบายก่อนจะเชื่อการเทียบ baseline vs shift
#  3. การเลื่อนแบบ static (คัดลอก 5 ชุด) อ่อนกว่าการสุ่มใหม่ทุก epoch — ถ้าผลก้ำกึ่ง อาจเป็นเพราะตรงนี้
#  4. ผลกับ User 5 คนเดียว ไม่บอกเรื่องคนใหม่ — ต้องรอ 6-subject LOSO ใน Phase 5.5
#  5. ห้ามเลือก seed ที่ดีที่สุดมารายงาน — รายงานค่าเฉลี่ย ± SD ทั้ง 3 seed เสมอ
# ─────────────────────────────────────────────────────────────────────────
