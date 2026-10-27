"""train_phase55c_final.py — 5.5c ขั้นที่ (1): เทรนโมเดลตัวจริงสูตร R (re-centring) ด้วยข้อมูล "ทุกคน" แล้วเซฟ
ตามลำดับงานของ PROJECT_FACTS §5.1w / §5.1x / §5.1y (กติกา A15)

  สูตร   : เหมือนแขน R ใน train_phase55c.train_one ทุกบรรทัด ต่างแค่ "ไม่มีชุดทดสอบ" และ "เซฟโมเดล"
           = build_model → กลับภาพซ้าย-ขวา (augment_flip) → recentre() → สลับลำดับ → ถ่วงน้ำหนักให้ทุกคนเท่ากัน
             (subject_weights) → fit(validation_split=0.1, early stopping patience 30, สูงสุด 1000 epochs)
  ข้อมูล : ST 4 คน + User 5 + User 6 (data\\phase4 + data\\phase4_extra) — เหมือน E5 ทุกเฟรม (36,409)
  seeds  : 42–46 → 5 โมเดล → ขั้นถัดไปรวมเป็น E5R (คนละสคริปต์)

  ⚠️ โมเดลพวกนี้ "ต้องได้ข้อมูลที่ re-centre แล้วเท่านั้น" ทั้งบน PC และบนบอร์ด (MY_TOF_RECENTRE)
     ถ้าส่งเฟรมดิบเข้าไปตรงๆ ผลจะผิดโดยไม่มีอะไรเตือน — manifest.json บันทึก input_transform ไว้ให้สคริปต์อื่นตรวจ

  ไม่มีตัวเลขความแม่นยำ: ทุกคนอยู่ในชุดเทรน (ความแม่นยำกับคนใหม่ = LOSO §5.1x; กับคนที่รู้จัก = กติกา A15 §5.1y)
  val_loss ไม่แสดงบนจอ (เก็บในไฟล์เท่านั้น) กันการเลือกโมเดลจากตัวเลขหลังเห็นผล — เหมือน train_phase55_final.py

  รันต่อจากที่ค้างได้: โมเดลที่เซฟแล้วและ sha256 ตรงกับ runs.jsonl จะถูกข้าม

วิธีรัน (ที่ root ของ repo):
  1) ทดสอบระบบก่อน (โมเดลห้ามใช้, ~1 นาที, เขียนลง results_phase55c_final_SMOKE_TEST):
     python pc_tools\\train_phase55c_final.py --st-data C:\\ds\\ST_VL53L8CX_handposture_dataset --ours data\\phase4 data\\phase4_extra --smoke-test
  2) ของจริง (5 โมเดล):
     python pc_tools\\train_phase55c_final.py --st-data C:\\ds\\ST_VL53L8CX_handposture_dataset --ours data\\phase4 data\\phase4_extra

ผลลัพธ์: results_phase55c_final\\
    model_R_seed42.keras ... model_R_seed46.keras   โมเดล 5 ตัว
    runs.jsonl                                      บันทึกทีละโมเดล (epochs, val_loss, sha256, ตรวจ re-centre)
    manifest.json                                   ข้อมูลไว้ทำซ้ำ (commit, sha256 ของสคริปต์/ข้อมูล, input_transform)
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
import evaluate_phase4 as ev            # noqa: E402
import train_phase55 as t5              # noqa: E402
import train_phase55_final as base      # noqa: E402  (ใช้ read_done / lib_versions / data_checksums ตัวเดิม)
import recentre as rc                   # noqa: E402

ARM = "R"                               # แขนที่กติกา §5.1w เลือก (ผล §5.1x, 27 ต.ค. 2026)
# ชุดทดสอบที่ห้ามเข้าชุดเทรนเด็ดขาด (§5.1l, §5.1y) — ถ้าชื่อโฟลเดอร์ใน --ours มีคำพวกนี้ สคริปต์หยุดทันที
FORBIDDEN_IN_TRAINING = ("phase56", "phase57", "a5_love", "a6_", "a6sim", "dryrun", "rejected")


def recentre_stats(X: np.ndarray) -> dict:
    """ตรวจว่าข้อมูลที่จะป้อนโมเดล re-centre แล้วจริง: จุดศูนย์กลางมืออยู่ใน (3.0, 4.0] และเลื่อนซ้ำแล้วไม่ขยับ"""
    hand = X[..., 0] < rc.HAND_THRESH
    cnt = hand.sum(axis=(1, 2))
    has = cnt > 0
    r = (hand * np.arange(8)[None, :, None]).sum(axis=(1, 2))
    c = (hand * np.arange(8)[None, None, :]).sum(axis=(1, 2))
    cr = np.where(has, r / np.maximum(cnt, 1), np.nan)
    cc = np.where(has, c / np.maximum(cnt, 1), np.nan)
    inbox = has & (cr > 3.0) & (cr <= 4.0) & (cc > 3.0) & (cc <= 4.0)
    dr, dc, _ = rc.shifts(X)
    return {"frames": int(len(X)), "frames_with_hand": int(has.sum()),
            "centroid_in_box_frac": float(inbox.sum() / max(int(has.sum()), 1)),
            "second_pass_nonzero_shift": int(((dr != 0) | (dc != 0)).sum())}


def train_final_r(tr, X: np.ndarray, y: np.ndarray, users: np.ndarray, seed: int,
                  max_epochs: int, model_path: Path) -> dict:
    """เทรน 1 โมเดลสูตร R ด้วยข้อมูลทั้งหมด แล้วเซฟ — ลำดับเดียวกับ train_phase55c.train_one แขน R ทุกบรรทัด"""
    tf = tr.tf
    model = tr.build_model(seed)                       # set_random_seed อยู่ใน build_model
    X_aug = tr.augment_flip(X, seed)
    X_aug = rc.recentre(X_aug)                         # หลัง flip — เหมือน LOSO แขน R (§5.1w)
    check = recentre_stats(X_aug)
    if check["centroid_in_box_frac"] < 0.99:           # ไม่ควรเกิด: ถ้าเกิด แปลว่า recentre ไม่ได้ทำงานตามที่คิด
        raise SystemExit(f"re-centring check failed: only {check['centroid_in_box_frac']:.3%} of hand frames "
                         "have their centroid in (3, 4] — stop and investigate")
    perm = np.random.default_rng(seed).permutation(len(X_aug))      # แก้บั๊ก #51: สลับก่อนแบ่ง
    X_fit, y_fit = X_aug[perm], y[perm]
    w_fit = t5.subject_weights(users)[perm]                         # แขน B/R: ทุกคนน้ำหนักเท่ากัน
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    hist = model.fit(X_fit, y_fit, sample_weight=w_fit, batch_size=32, epochs=max_epochs,
                     validation_split=0.1, callbacks=[stop], verbose=0)
    model.save(str(model_path))                        # Keras 3 ต้องการชื่อไฟล์ลงท้าย .keras
    vl = hist.history.get("val_loss", [])
    best = int(np.argmin(vl)) if vl else -1
    va = hist.history.get("val_accuracy", [])
    return {"epochs": len(hist.history["loss"]), "best_epoch": best + 1,
            "best_val_loss": float(vl[best]) if vl else float("nan"),
            "val_acc_at_best": float(va[best]) if (vl and len(va) > best) else float("nan"),
            "n_train": int(len(X)), "params": int(model.count_params()), "recentre_check": check}


def render(runs: list[dict], smoke: bool) -> str:
    """ข้อความสรุปบนจอ — ตั้งใจไม่แสดง val_loss"""
    L = [ev.LINE, "  Phase 5.5c final training  ·  arm R (re-centring)  ·  all 6 subjects  ·  §5.1w/§5.1x", ev.LINE]
    if smoke:
        L.append("  *** SMOKE TEST — models are meaningless, do not use ***")
    L.append(f"  {'seed':<6}{'epochs':>8}{'params':>9}  {'centred':>8}  model file / sha256 (first 16)")
    for r in sorted(runs, key=lambda x: x["seed"]):
        cf = r.get("recentre_check", {}).get("centroid_in_box_frac", float("nan"))
        L.append(f"  {r['seed']:<6}{r['epochs']:>8}{r['params']:>9}  {cf * 100:>7.2f}%  "
                 f"{r['model_file']}  {r['model_sha256'][:16]}")
    L += ["  'centred' = share of training frames whose hand centroid lies in (3, 4] after recentre().",
          "  These models expect RE-CENTRED input (pc_tools\\recentre.py on PC, MY_TOF_RECENTRE on the board).",
          "  No accuracy here: every subject is in training (new people = LOSO §5.1x; known user = A15 §5.1y).", ev.LINE]
    return "\n".join(L)


def main(argv: list[str] | None = None, trainer: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st-data", type=Path, required=True)
    ap.add_argument("--ours", type=Path, nargs="+", required=True, help=r"data\phase4 data\phase4_extra")
    ap.add_argument("--train-script", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results_phase55c_final"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--smoke-test", action="store_true", help="1 seed, 2 epochs — ทดสอบว่ารันได้ โมเดลห้ามใช้")
    args = ap.parse_args(argv)

    for d in args.ours:                                 # กันชุดทดสอบหลุดเข้าชุดเทรน
        name = str(d).lower().replace("\\", "/")
        bad = [w for w in FORBIDDEN_IN_TRAINING if w in name]
        if bad:
            raise SystemExit(f"--ours {d}: looks like a test / rejected set ({bad}) — never train on it (§5.1l, §5.1y)")
    if sorted(str(d).lower().replace("\\", "/").rstrip("/").split("/")[-1] for d in args.ours) != ["phase4", "phase4_extra"] \
            and not args.smoke_test:
        raise SystemExit(r"--ours must be exactly data\phase4 data\phase4_extra (same data as E5 and the LOSO run)")

    here = Path(__file__).resolve().parent
    tr_path = args.train_script or next((p for p in (here / "train_st_cnn2d.py", here.parent / "train_st_cnn2d.py")
                                         if p.is_file()), None)
    if tr_path is None or not Path(tr_path).is_file():
        raise SystemExit("train_st_cnn2d.py not found — use --train-script <path>")
    tr = ev.import_module_from(Path(tr_path), "train_st_cnn2d")
    if list(tr.CLASSES) != ev.MODEL_CLASSES:
        raise SystemExit(f"class order differs: {tr.CLASSES}")

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
    pre = recentre_stats(rc.recentre(data["X"]))
    print(f"  re-centring check on all frames: centroid in (3, 4] for {pre['centroid_in_box_frac'] * 100:.2f} % "
          f"of {pre['frames_with_hand']:,} hand frames; second pass moves {pre['second_pass_nonzero_shift']} frames")

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    done = base.read_done(runs_path, out_dir)
    plan = [s for s in seeds if s not in done]
    print(f"Plan: {len(plan)} models to train ({len(done)} already saved and verified in {out_dir})")

    fn = trainer or (lambda *a: train_final_r(tr, *a))
    t0 = time.time()
    for i, seed in enumerate(plan, 1):
        name = f"model_{ARM}_seed{seed}.keras"
        r = fn(data["X"], data["y"], data["user"], seed, max_epochs, out_dir / name)
        r.update({"arm": ARM, "seed": seed, "model_file": name,
                  "model_sha256": ev.sha256_file(out_dir / name),
                  "at": datetime.now().isoformat(timespec="seconds")})
        with runs_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
        done[seed] = r
        el = time.time() - t0
        print(f"  [{i}/{len(plan)}] arm {ARM}  seed {seed}: saved {name}  ({r['epochs']} ep)  ·  "
              f"about {el / i * (len(plan) - i) / 60:.0f} min left")

    runs = [done[s] for s in seeds if s in done]
    report = render(runs, args.smoke_test)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    me = Path(__file__).resolve()
    manifest = {
        "run_at": datetime.now().isoformat(timespec="seconds"), "smoke_test": args.smoke_test,
        "arm": ARM, "seeds": seeds, "max_epochs": max_epochs,
        "class_order": list(tr.CLASSES),           # ลำดับ output ของโมเดล — ใช้ตอนแปลงลงบอร์ด (§9 #38)
        "input_transform": {"recentre": True, "applied_after": "st_dataset/train_st_cnn2d preprocessing (+ flip in training)",
                            "recentre_py": {"path": str(Path(rc.__file__)), "sha256": ev.sha256_file(Path(rc.__file__))},
                            "firmware_switch": "MY_TOF_RECENTRE"},
        "git_commit": ev.git_commit(),
        "train_st_cnn2d": {"path": str(tr_path), "sha256": ev.sha256_file(Path(tr_path))},
        "train_phase55": {"path": str(Path(t5.__file__)), "sha256": ev.sha256_file(Path(t5.__file__))},
        "train_phase55_final": {"path": str(Path(base.__file__)), "sha256": ev.sha256_file(Path(base.__file__))},
        "this_script": {"path": str(me), "sha256": ev.sha256_file(me)},
        "libs": base.lib_versions(tr), "frames_per_user": per_user, "n_frames": int(len(data["y"])),
        "recentre_check_all_frames": pre,
        "data_checksums": base.data_checksums([args.st_data] + list(args.ours)),
        "models": [{k: r[k] for k in ("seed", "model_file", "model_sha256", "epochs", "params")} for r in runs],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    return {"out": out_dir, "runs": runs, "manifest": manifest}


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ถ้าแก้โค้ดหรือข้อมูลระหว่างทาง ให้ลบทั้งโฟลเดอร์ results_phase55c_final แล้วเริ่มใหม่ (โมเดลเก่า/ใหม่จะปนกัน)
#  2. ห้ามเอาโมเดลพวกนี้ไปทายเฟรมที่ยังไม่ได้ re-centre — ผลจะผิดเงียบๆ (ดู manifest.json → input_transform)
#  3. ห้ามวัดความแม่นยำกับ data\phase4 / phase4_extra — อยู่ในชุดเทรนแล้ว
#  4. val_loss ในไฟล์ใช้หยุดเทรนเท่านั้น ห้ามใช้เลือกโมเดล (E5R ใช้ครบทั้ง 5 ตัว)
#  5. smoke test เขียนลงโฟลเดอร์ _SMOKE_TEST แยก — ห้ามเอาโมเดลไปใช้ และห้าม commit
#  6. ต้องไม่ commit ระหว่างรัน — git_commit ถูกบันทึกตอนจบ
# ─────────────────────────────────────────────────────────────────────────
