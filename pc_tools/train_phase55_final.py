"""train_phase55_final.py — Phase 5.5 ขั้นสุดท้าย: เทรนแขนที่ถูกเลือก (B) ด้วยข้อมูล "ทุกคน" แล้วเซฟโมเดล
ตาม deliverable rule ของ PROJECT_FACTS §5.1i / §5.1i-R

  สูตร   : เหมือน train_phase55.train_one ทุกอย่าง (build_model, กลับภาพซ้าย-ขวา, สลับก่อน
           validation_split=0.1, Adam 0.01, batch 32, early stopping patience 30, สูงสุด 1000 epochs)
           แขน B = ถ่วงน้ำหนักให้ทุกคนมีน้ำหนักรวมเท่ากัน (subject_weights)
  ข้อมูล : ST 4 คน + User 5 + User 6 (ชุดหลัก + รอบเพิ่ม) = ทุกเฟรมที่มี ไม่มีใครถูกกันไว้ทดสอบ
  seeds  : 42-46 → 5 โมเดล เซฟเป็น .keras พร้อม sha256

ทำไมไม่มีตัวเลขความแม่นยำ: ข้อมูลทุกคนถูกใช้เทรนหมดแล้ว จึงไม่มีคนที่ "ไม่เคยเห็น" เหลือให้ทดสอบ
ตัวเลขที่บอกว่าสูตรนี้ดีแค่ไหนกับคนใหม่คือผล LOSO ใน §5.1i-R ส่วนการเลือก 1 ใน 5 โมเดล (หรือใช้ทั้ง 5
เป็น ensemble) คือ Phase 5.5b ที่ต้องลงทะเบียนกติกาก่อน — สคริปต์นี้จึง "ไม่พิมพ์" val_loss บนจอ
(เก็บไว้ในไฟล์เท่านั้น) กันการเลือกโมเดลจากตัวเลขหลังเห็นผล

รันต่อจากที่ค้างได้: โมเดลที่เซฟแล้วและ sha256 ตรงกับ runs.jsonl จะถูกข้าม

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\train_phase55_final.py --st-data C:\\ds\\ST_VL53L8CX_handposture_dataset
        --ours data\\phase4 data\\phase4_extra
ทดสอบว่าระบบรันได้ (โมเดลห้ามเอาไปใช้):
    ... --smoke-test     (1 seed, 2 epochs, เขียนลงโฟลเดอร์ _SMOKE_TEST แยก)

ผลลัพธ์: results_phase55_final\\
    model_B_seed42.keras ... model_B_seed46.keras   โมเดล 5 ตัว
    runs.jsonl                                      บันทึกทีละโมเดล (epochs, val_loss, sha256)
    manifest.json                                   ข้อมูลไว้ทำซ้ำ: commit, sha256 ของสคริปต์/ข้อมูล, จำนวนเฟรม, ลำดับคลาส
    report.txt                                      สรุปบนจอ
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase4 as ev  # noqa: E402
import train_phase55 as t5  # noqa: E402

SELECTED_ARM = "B"                    # ผลของกติกาเลือกแขน §5.1i-R (20 ต.ค. 2026)


def train_final(tr, arm: str, X: np.ndarray, y: np.ndarray, users: np.ndarray, seed: int,
                max_epochs: int, model_path: Path) -> dict:
    """เทรน 1 โมเดลด้วยข้อมูลทั้งหมด แล้วเซฟลง model_path — สูตรเดียวกับ t5.train_one ทุกบรรทัด
    ต่างกันแค่ไม่มีชุดทดสอบ และเซฟโมเดล"""
    tf = tr.tf
    model = t5.build_gap_model(tr, seed) if arm == "C" else tr.build_model(seed)
    X_aug = tr.augment_flip(X, seed)
    perm = np.random.default_rng(seed).permutation(len(X_aug))      # แก้บั๊ก #51: สลับก่อนแบ่ง
    X_fit, y_fit = X_aug[perm], y[perm]
    w_fit = t5.subject_weights(users)[perm] if arm == "B" else None
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    hist = model.fit(X_fit, y_fit, sample_weight=w_fit, batch_size=32, epochs=max_epochs,
                     validation_split=0.1, callbacks=[stop], verbose=0)
    model.save(str(model_path))          # Keras 3 ต้องการชื่อไฟล์ลงท้าย .keras
    vl = hist.history.get("val_loss", [])
    best = int(np.argmin(vl)) if vl else -1
    return {"epochs": len(hist.history["loss"]), "best_epoch": best + 1,
            "best_val_loss": float(vl[best]) if vl else float("nan"),
            "val_acc_at_best": float(hist.history.get("val_accuracy", [float("nan")] * (best + 1))[best])
            if vl else float("nan"),
            "n_train": int(len(X)), "params": int(model.count_params())}


def lib_versions(tr) -> dict:
    """เวอร์ชัน TensorFlow / Keras ที่ใช้เทรน (ต้องรู้ตอนแปลงลงบอร์ด)"""
    out = {}
    try:
        out["tensorflow"] = tr.tf.__version__
    except AttributeError:
        out["tensorflow"] = "unknown"
    try:
        import keras  # Keras 3 แยกแพ็กเกจ
        out["keras"] = keras.__version__
    except (ImportError, AttributeError):
        out["keras"] = "unknown"
    return out


def data_checksums(dirs: list[Path]) -> dict:
    """sha256 รวมของไฟล์ .npz ในแต่ละโฟลเดอร์ — ข้อมูลเปลี่ยนแม้ไฟล์เดียว ค่าจะเปลี่ยน"""
    out = {}
    for d in dirs:
        files = sorted(Path(d).rglob("*.npz"))
        out[str(d)] = {"n_files": len(files), "sha256": ev.data_checksum(files, Path(d)) if files else "none"}
    return out


def read_done(runs_path: Path, out_dir: Path) -> dict[int, dict]:
    """seed ที่เสร็จแล้วจริง = มีใน runs.jsonl (ใช้บรรทัดล่าสุดของ seed นั้น) + ไฟล์โมเดลยังอยู่ + sha256 ตรง"""
    if not runs_path.exists():
        return {}
    last: dict[int, dict] = {}
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            last[int(r["seed"])] = r
    ok = {}
    for seed, r in last.items():
        f = out_dir / r["model_file"]
        if f.is_file() and ev.sha256_file(f) == r["model_sha256"]:
            ok[seed] = r
    return ok


def render(arm: str, runs: list[dict], smoke: bool) -> str:
    """ข้อความสรุปบนจอ — ตั้งใจไม่แสดง val_loss (ดู docstring)"""
    L = [ev.LINE, f"  Phase 5.5 final training  ·  arm {arm}  ·  all 6 subjects  ·  §5.1i-R", ev.LINE]
    if smoke:
        L.append("  *** SMOKE TEST — models are meaningless, do not use ***")
    L.append(f"  {'seed':<6}{'epochs':>8}{'params':>9}  model file / sha256 (first 16)")
    for r in sorted(runs, key=lambda x: x["seed"]):
        L.append(f"  {r['seed']:<6}{r['epochs']:>8}{r['params']:>9}  {r['model_file']}  {r['model_sha256'][:16]}")
    L += ["  No accuracy here: every subject is in training (accuracy for new people = §5.1i-R LOSO).",
          "  Choosing 1 of these models, or an ensemble, is Phase 5.5b — pre-register first.", ev.LINE]
    return "\n".join(L)


def main(argv: list[str] | None = None, trainer: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st-data", type=Path, required=True)
    ap.add_argument("--ours", type=Path, nargs="+", required=True, help=r"data\phase4 data\phase4_extra")
    ap.add_argument("--train-script", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results_phase55_final"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--arm", default=SELECTED_ARM, choices=t5.ARMS,
                    help="ค่าเริ่มต้น B ตามกติกาเลือกแขน — เปลี่ยนเฉพาะเมื่อมี pre-registration ใหม่")
    ap.add_argument("--smoke-test", action="store_true", help="1 seed, 2 epochs — ทดสอบว่ารันได้ โมเดลห้ามใช้")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    tr_path = args.train_script or next((p for p in (here / "train_st_cnn2d.py", here.parent / "train_st_cnn2d.py")
                                         if p.is_file()), None)
    if tr_path is None or not Path(tr_path).is_file():
        raise SystemExit("train_st_cnn2d.py not found — use --train-script <path>")
    tr = ev.import_module_from(Path(tr_path), "train_st_cnn2d")
    if list(tr.CLASSES) != ev.MODEL_CLASSES:
        raise SystemExit(f"class order differs: {tr.CLASSES}")
    if args.arm != SELECTED_ARM:
        print(f"  [WARNING] arm {args.arm} is not the selected arm ({SELECTED_ARM}) — needs its own pre-registration")

    seeds, max_epochs, out_dir = args.seeds, 1000, args.out_dir
    if args.smoke_test:
        seeds, max_epochs = seeds[:1], 2
        out_dir = out_dir.with_name(out_dir.name + "_SMOKE_TEST")

    print("Loading ST + our data with train_st_cnn2d.load_dataset() ...")
    data = t5.load_all(tr, args.st_data, args.ours)
    users_present = sorted(set(data["user"].tolist()))
    missing = [u for u in t5.OUR_USERS if u not in users_present]
    if missing and not args.smoke_test:
        raise SystemExit(f"our subjects missing: {missing} — final training needs both User5 and User6.")
    per_user = {}
    for u in users_present:
        n_all = int((data["user"] == u).sum())
        n_main = int(((data["user"] == u) & data["is_main"]).sum())
        per_user[f"User{u}"] = {"frames": n_all, "main": n_main, "extra": n_all - n_main}
        print(f"  User{u}: {n_all:,} frames" + (f" (main {n_main:,}, extra {n_all - n_main:,})"
                                                if u in t5.OUR_USERS else ""))
    print(f"  total: {len(data['y']):,} frames — all used for training (no held-out subject)")

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    done = read_done(runs_path, out_dir)
    plan = [s for s in seeds if s not in done]
    print(f"Plan: {len(plan)} models to train ({len(done)} already saved and verified in {out_dir})")

    fn = trainer or (lambda *a: train_final(tr, *a))
    t0 = time.time()
    for i, seed in enumerate(plan, 1):
        name = f"model_{args.arm}_seed{seed}.keras"
        r = fn(args.arm, data["X"], data["y"], data["user"], seed, max_epochs, out_dir / name)
        r.update({"arm": args.arm, "seed": seed, "model_file": name,
                  "model_sha256": ev.sha256_file(out_dir / name),
                  "at": datetime.now().isoformat(timespec="seconds")})
        with runs_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
        done[seed] = r
        el = time.time() - t0
        print(f"  [{i}/{len(plan)}] arm {args.arm}  seed {seed}: saved {name}  ({r['epochs']} ep)  ·  "
              f"about {el / i * (len(plan) - i) / 60:.0f} min left")

    runs = [done[s] for s in seeds if s in done]
    report = render(args.arm, runs, args.smoke_test)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"), "smoke_test": args.smoke_test,
        "arm": args.arm, "seeds": seeds, "max_epochs": max_epochs,
        "class_order": list(tr.CLASSES),           # ลำดับ output ของโมเดล — ใช้ตอนแปลงลงบอร์ด (§9 #38)
        "git_commit": ev.git_commit(),
        "train_st_cnn2d": {"path": str(tr_path), "sha256": ev.sha256_file(Path(tr_path))},
        "train_phase55": {"path": str(Path(t5.__file__)), "sha256": ev.sha256_file(Path(t5.__file__))},
        "this_script_sha256": ev.sha256_file(Path(__file__)),
        "libs": lib_versions(tr), "frames_per_user": per_user, "n_frames": int(len(data["y"])),
        "data_checksums": data_checksums([args.st_data] + list(args.ours)),
        "models": [{k: r[k] for k in ("seed", "model_file", "model_sha256", "epochs", "params")} for r in runs],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return {"out": out_dir, "runs": runs, "manifest": manifest}


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ถ้าแก้โค้ดหรือข้อมูลระหว่างทาง ให้ลบทั้งโฟลเดอร์ results_phase55_final แล้วเริ่มใหม่
#     ไม่งั้นโมเดลเก่ากับใหม่จะปนกัน (sha256 ตรวจได้แค่ว่าไฟล์ไม่ถูกแก้ ไม่ได้ตรวจว่าสูตรเหมือนเดิม)
#  2. TensorFlow บน CPU ไม่รับประกันผลซ้ำเป๊ะแม้ seed เดิม [จำมา] — โมเดลที่เทรนใหม่อาจต่างเล็กน้อย
#     ให้ถือว่าไฟล์ .keras + sha256 ที่เซฟไว้คือตัวจริง ไม่ใช่ "seed 42"
#  3. val_loss ในไฟล์มาจากเฟรมสุ่ม 10 % ของทุกคน (เฟรมติดกันมาก) จึงมองโลกในแง่ดี — ใช้หยุดเทรนเท่านั้น
#     ห้ามใช้เลือกโมเดลจนกว่า 5.5b จะลงทะเบียนว่าจะใช้หรือไม่
#  4. ห้ามเอาโมเดลพวกนี้ไปวัดกับ data\phase4 แล้วรายงานเป็นความแม่นยำ — ข้อมูลนั้นอยู่ในชุดเทรนแล้ว
#  5. smoke test เขียนลงโฟลเดอร์ _SMOKE_TEST แยก — ห้ามเอาโมเดลไปใช้
# ─────────────────────────────────────────────────────────────────────────
