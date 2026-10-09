"""train_phase2_fixed.py — เทรน Phase 2 ใหม่ โดยแก้บั๊ก validation split (PROJECT_FACTS §9 #51) จุดเดียว

บั๊กเดิม: train_st_cnn2d.py เรียก model.fit(validation_split=0.1) กับข้อมูลที่เรียงตามชื่อท่า A-Z
  Keras ตัด 10 % "ท้ายสุด" เป็น validation โดยไม่สุ่ม (ยืนยันจาก Keras 3.15.1 array_slicing.py)
  → validation เกือบทั้งหมดเป็นท่า Love และโมเดลได้เรียน Love น้อยมาก
การแก้: สลับลำดับข้อมูลเทรนแบบสุ่ม (seed เดียวกับการเทรนรอบนั้น) "ก่อน" เรียก fit — แก้แค่นี้จุดเดียว
  ที่เหลือใช้ฟังก์ชันจาก train_st_cnn2d.py ตัวเดิมทั้งหมด (import มาใช้ ไม่เขียนซ้ำ ไม่แก้ไฟล์เดิม)
  ไฟล์เดิมและ results_phase2_final\\ เก็บไว้เหมือนเดิม เป็นหลักฐานของบั๊ก

Pre-registration: PROJECT_FACTS §5.1h (เขียนก่อนรัน) — R1 ตรวจสัดส่วนท่าใน validation ว่าแก้ได้จริง

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\train_phase2_fixed.py --data-dir C:\\ds\\ST_VL53L8CX_handposture_dataset
ใช้เวลา: LOSO 4 คน × 3 seeds = 12 โมเดล + production 1 โมเดล ≈ 20-60 นาทีบน CPU [ประมาณ ยังไม่ได้วัด]

ผลลัพธ์: results_phase2_fixed\\phase2_results.json, production_model.keras, report.txt
แล้ววัดมือ User 5 ด้วย evaluate_phase4.py --model results_phase2_fixed\\production_model.keras
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase4 as ev  # noqa: E402  (ใช้ import_module_from / sha256 / git_commit / LINE)


def shuffled(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """สลับลำดับ X, y ด้วยลำดับเดียวกัน (seed กำหนด) — หัวใจของการแก้บั๊ก #51"""
    perm = np.random.default_rng(seed).permutation(len(X))
    return X[perm], y[perm]


def val_mix(y: np.ndarray, classes: list[str], frac: float = 0.1) -> dict[str, float]:
    """สัดส่วนท่าใน validation set ที่ Keras จะใช้ (= 10 % ท้ายสุด, เหมือน array_slicing.py)"""
    split_at = int(np.floor(len(y) * (1.0 - frac)))
    tail = y[split_at:]
    cnt = Counter(tail.tolist())
    return {c: round(cnt.get(i, 0) / len(tail) * 100, 1) for i, c in enumerate(classes)}


def make_fixed_train_and_eval(tr):
    """train_and_eval ตัวเดิมของ train_st_cnn2d ทุกบรรทัด + สลับลำดับก่อน fit"""
    tf = tr.tf

    def train_and_eval(X_train, y_train, X_test, y_test, label: str, seed: int,
                       max_epochs: int = 1000) -> dict:
        model = tr.build_model(seed)
        X_train_aug = tr.augment_flip(X_train, seed)
        X_fit, y_fit = shuffled(X_train_aug, y_train, seed)          # ← การแก้ (จุดเดียว)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                                      restore_best_weights=True)
        history = model.fit(X_fit, y_fit, batch_size=32, epochs=max_epochs, validation_split=0.1,
                            callbacks=[early_stop], verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        cm = tf.math.confusion_matrix(y_test, y_pred, num_classes=tr.N_CLASSES).numpy()
        print(f"    {label}: acc {float(test_acc) * 100:.2f} %  ({len(history.history['loss'])} epochs)")
        return {"label": label, "seed": seed, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                "epochs_ran": len(history.history["loss"]), "test_accuracy": float(test_acc),
                "test_loss": float(test_loss), "confusion_matrix": np.asarray(cm).tolist()}
    return train_and_eval


def fixed_production(tr, ds, seed: int, save_path: Path) -> dict:
    """run_production ตัวเดิม + สลับลำดับก่อน fit"""
    tf = tr.tf
    model = tr.build_model(seed)
    X_aug = tr.augment_flip(ds.X, seed)
    X_fit, y_fit = shuffled(X_aug, ds.y, seed)                       # ← การแก้ (จุดเดียว)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    history = model.fit(X_fit, y_fit, batch_size=32, epochs=1000, validation_split=0.1,
                        callbacks=[early_stop], verbose=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    return {"label": "production_model_fixed", "seed": seed, "n_train_total": int(len(ds.X)),
            "epochs_ran": len(history.history["loss"]),
            "final_train_accuracy": float(history.history["accuracy"][-1]), "saved_to": str(save_path)}


def per_class_recall(cm: list[list[int]], classes: list[str]) -> dict[str, float]:
    """recall ต่อท่าจาก confusion matrix"""
    a = np.asarray(cm, dtype=float)
    return {c: (float(a[i, i] / a[i].sum() * 100) if a[i].sum() else float("nan"))
            for i, c in enumerate(classes)}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="โฟลเดอร์ ST_VL53L8CX_handposture_dataset")
    ap.add_argument("--train-script", type=Path, default=None, help="path ของ train_st_cnn2d.py (ปกติหาเอง)")
    ap.add_argument("--out-dir", type=Path, default=Path("results_phase2_fixed"))
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--skip-loso", action="store_true", help="เทรนแค่ production (ไม่แนะนำ — R2 ต้องใช้ LOSO)")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    tr_path = args.train_script or next((p for p in (here / "train_st_cnn2d.py", here.parent / "train_st_cnn2d.py")
                                         if p.is_file()), None)
    if tr_path is None or not tr_path.is_file():
        raise SystemExit("train_st_cnn2d.py not found — use --train-script <path>")
    tr = ev.import_module_from(tr_path, "train_st_cnn2d")
    if list(tr.CLASSES) != ev.MODEL_CLASSES:
        raise SystemExit(f"class order differs from the deployed model: {tr.CLASSES}")
    seeds = [tr.SEED + i for i in range(args.n_seeds)]

    print(f"Loading {args.data_dir} with train_st_cnn2d.load_dataset() ...")
    ds = tr.load_dataset(args.data_dir)
    print(f"  {len(ds.X):,} frames, users {sorted(np.unique(ds.groups).tolist())}")

    # ---- R1: พิสูจน์ว่าการแก้ทำงาน — สัดส่วนท่าใน validation ก่อน/หลังแก้ ----
    X_aug = tr.augment_flip(ds.X, seeds[0])
    _, y_fix = shuffled(X_aug, ds.y, seeds[0])
    mix_orig, mix_fix = val_mix(ds.y, tr.CLASSES), val_mix(y_fix, tr.CLASSES)
    share = {c: round(float((ds.y == i).mean() * 100), 1) for i, c in enumerate(tr.CLASSES)}
    r1 = all(abs(mix_fix[c] - share[c]) <= 3.0 for c in tr.CLASSES)
    print("  validation mix (production)  original order → " +
          ", ".join(f"{c} {v:.0f}%" for c, v in mix_orig.items() if v > 0))
    print("                               fixed (shuffled) → " +
          ", ".join(f"{c} {v:.0f}%" for c, v in mix_fix.items()))
    print(f"  R1 (each class within ±3 pp of its data share): {'HOLDS' if r1 else 'FAILS — stop and check'}")
    if not r1:
        raise SystemExit("R1 failed: the shuffle is not producing a representative validation set")

    results: dict = {"fix": "shuffle before validation_split (PROJECT_FACTS §9 #51, §5.1h)",
                     "train_script": str(tr_path), "train_script_sha256": ev.sha256_file(tr_path),
                     "seeds": seeds, "classes": list(tr.CLASSES), "tensorflow_version": tr.tf.__version__,
                     "n_total_frames": int(len(ds.X)), "val_mix_original": mix_orig, "val_mix_fixed": mix_fix,
                     "R1_holds": r1, "git_commit": ev.git_commit(),
                     "run_at": datetime.now().isoformat(timespec="seconds")}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not args.skip_loso:
        print(f"\n=== subject-independent (leave-one-user-out), {len(seeds)} seeds per fold, FIXED split ===")
        tr.train_and_eval = make_fixed_train_and_eval(tr)   # run_subject_independent → train_multi_seed → ตัวนี้
        si = tr.run_subject_independent(ds, seeds=seeds, equalize=False)
        for r in si["fold_results"]:
            recs = [per_class_recall(run["confusion_matrix"], tr.CLASSES) for run in r["runs"]]
            r["mean_recall"] = {c: float(np.mean([x[c] for x in recs])) for c in tr.CLASSES}
        results["subject_independent"] = si
        print(f"  elapsed {(time.time() - t0) / 60:.0f} min")

    print("\n=== production model (all 4 users), FIXED split ===")
    prod_path = args.out_dir / "production_model.keras"
    results["production"] = fixed_production(tr, ds, seeds[0], prod_path)
    results["production"]["sha256"] = ev.sha256_file(prod_path)

    # ---- รายงาน ----
    L = [ev.LINE, "  Phase 2 re-run with the validation-split fix (§5.1h)", ev.LINE,
         f"  Data        {len(ds.X):,} ST frames, 4 users  ·  seeds {seeds}",
         f"  Fix check   R1 {'HOLDS' if r1 else 'FAILS'}: validation Love share {mix_orig['Love']:.0f} % → "
         f"{mix_fix['Love']:.0f} % (data share {share['Love']:.0f} %)"]
    if "subject_independent" in results:
        si = results["subject_independent"]
        L += ["", f"  {'held-out user':<15}{'original (§5.1d)':>18}{'fixed':>20}"]
        orig = {1: "68.91", 2: "89.71", 3: "78.31", 4: "81.97"}
        for r in sorted(si["fold_results"], key=lambda r: r["held_out_user"]):
            u = r["held_out_user"]
            L.append(f"  User{u:<11}{orig.get(u, '?') + ' %':>18}{r['mean_accuracy'] * 100:>12.2f} ± "
                     f"{r['sd_accuracy'] * 100:4.2f}")
        mm = si["mean_of_fold_means"] * 100
        L.append(f"  {'overall':<15}{'79.73 %':>18}{mm:>12.2f} ± {si['sd_of_fold_means'] * 100:4.2f}")
        L.append(f"  R2 (within ±5 pp of 79.73 %): {'HOLDS' if abs(mm - 79.73) <= 5 else 'DOES NOT HOLD'}"
                 f"  (difference {mm - 79.73:+.2f} pp)")
        L.append("  mean recall over folds:  " + "  ".join(
            f"{c} {np.mean([r['mean_recall'][c] for r in si['fold_results']]):.0f}" for c in tr.CLASSES))
    p = results["production"]
    L += ["", f"  Production  {p['epochs_ran']} epochs  ·  saved {p['saved_to']}",
          f"              sha256 {p['sha256'][:16]}…",
          "  Next: evaluate User 5 with this model (R3, R4, R5) using evaluate_phase4.py --model <path above>",
          f"  Elapsed {(time.time() - t0) / 60:.0f} min", ev.LINE]
    report = "\n".join(L)
    print("\n" + report)
    (args.out_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    with (args.out_dir / "phase2_results.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=float)
    return results


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ถ้า train_st_cnn2d.py ในเครื่องต่างจากตัวที่ใช้ตอน Phase 2 ผลจะเทียบกันไม่ได้ —
#     sha256 ของไฟล์ถูกบันทึกใน phase2_results.json
#  2. results_phase2_fixed\\ ถูกเขียนทับถ้ารันซ้ำ — ย้ายไฟล์เดิมออกก่อนถ้าต้องการเก็บ
#  3. production ใช้ seed เดียว (42) เหมือนต้นฉบับ — ความแปรปรวนจาก seed ดูจาก LOSO
#  4. ถ้า R1 ไม่ผ่าน สคริปต์จะหยุดก่อนเทรน (ไม่เสียเวลา)
# ─────────────────────────────────────────────────────────────────────────
