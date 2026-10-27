"""build_ensemble_r.py — 5.5c ขั้นที่ (1) ต่อ: รวมโมเดลสูตร R ทั้ง 5 ตัว (seed 42–46) เป็นโมเดลเดียว "E5R"
แล้วแปลงเป็น .tflite สำหรับ X-CUBE-AI — วิธีเดียวกับ E5 ทุกขั้น (ใช้ฟังก์ชันใน build_ensemble.py ตัวเดิม)
ต่างจาก E5 จุดเดียว: ข้อมูลที่ใช้ตรวจและชุดเฟรมสำหรับเทียบบอร์ด "re-centre แล้ว" (pc_tools\\recentre.py)
เพราะ E5R ถูกเทรนด้วยข้อมูลที่ re-centre — ป้อนเฟรมดิบเข้าไปจะได้ผลผิด (PROJECT_FACTS §5.1w, §5.1x)

  ขั้นที่สคริปต์ทำ (หยุดทันทีถ้าข้อไหนไม่ผ่าน):
  1. ตรวจไฟล์โมเดล 5 ตัวกับ sha256 ใน results_phase55c_final\\manifest.json + manifest ต้องเป็นแขน R,
     ไม่ใช่ smoke test, และบอกว่า input_transform = recentre
  2. สร้าง E5R = กราฟเดียว 5 สาขา + Average (build_ensemble.build_e5 ตัวเดียวกับที่สร้าง E5)
  3. ตรวจบนข้อมูล data\\phase4 ที่ re-centre แล้ว: ผลแต่ละสาขา == โมเดลต้นฉบับ และ E5R == ค่าเฉลี่ยของ 5 ตัว
  4. แปลง E5R เป็น .tflite float32 (ไม่ quantize) แล้วตรวจว่าทายท่าตรงกับ .keras ทีละเฟรม (batch 1)
  5. เซฟ 200 เฟรมตัวอย่าง (re-centre แล้ว, มีทุกท่า) + คำตอบของ PC ไว้ใช้ตรวจ X-CUBE-AI (validate on desktop)

  ⚠️ ไม่คำนวณความแม่นยำ — data\\phase4 อยู่ในชุดเทรน ใช้แค่เช็กว่าการรวม/แปลงไม่ทำให้ผลเพี้ยน
     ความแม่นยำกับคนที่รู้จัก = กติกา A15 (§5.1y) ใช้สคริปต์คนละตัว
  ไม่มีตัวสำรอง seed 42 แบบ E5: ถ้า E5R ขึ้นบอร์ดไม่ได้ E5 อยู่บนบอร์ดต่อ (§5.1y ข้อ 3)

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\build_ensemble_r.py --models-dir results_phase55c_final --check-data data\\phase4

ผลลัพธ์: results_phase55c_final\\deploy\\
    E5R.keras, E5R.tflite                      โมเดลรวม
    board_check_inputs_R_200.npy               200 เฟรม (N, 8, 8, 2) float32 ที่ re-centre แล้ว
    board_check_E5R_probs_200.npy              คำตอบของ E5R บน PC (N, 7)
    build_report.json, report.txt              ผลการตรวจทุกข้อ + sha256 ของไฟล์ที่สร้าง
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase4 as ev        # noqa: E402
import build_ensemble as be         # noqa: E402  (verify_models / build_e5 / to_tflite / tflite_predict / pick_board_samples)
import recentre as rc               # noqa: E402

NAME = "E5R"


def check_manifest_r(models_dir: Path) -> dict:
    """manifest ต้องมาจาก train_phase55c_final.py: แขน R, ไม่ใช่ smoke test, input = re-centred"""
    man = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    if man.get("arm") != "R":
        raise SystemExit(f"{models_dir}\\manifest.json is arm {man.get('arm')!r}, expected 'R' "
                         "(did you point at results_phase55_final by mistake?)")
    if not (man.get("input_transform") or {}).get("recentre"):
        raise SystemExit("manifest.json does not say input_transform.recentre = true — refusing to build E5R")
    return man


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=Path("results_phase55c_final"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--check-data", type=Path, required=True, help=r"data\phase4 — ใช้เช็กความเท่ากันเท่านั้น")
    ap.add_argument("--n-tflite-check", type=int, default=2000, help="จำนวนเฟรมที่เทียบ tflite กับ keras")
    ap.add_argument("--train-script", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help=r"ค่าเริ่มต้น <models-dir>\deploy")
    args = ap.parse_args(argv)

    if any(w in str(args.check_data).lower() for w in ("phase56", "phase57", "a5_love")):
        raise SystemExit("--check-data must be training data (data\\phase4) — test sets are not used here (§5.1y)")
    here = Path(__file__).resolve().parent
    tr_path = args.train_script or next((p for p in (here / "train_st_cnn2d.py", here.parent / "train_st_cnn2d.py")
                                         if p.is_file()), None)
    if tr_path is None:
        raise SystemExit("train_st_cnn2d.py not found — use --train-script <path>")
    tr = ev.import_module_from(Path(tr_path), "train_st_cnn2d")
    tf = tr.tf
    out_dir = args.out_dir or (args.models_dir / "deploy")
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: dict = {}

    print("[1] verifying model files against manifest.json ...")
    man = check_manifest_r(args.models_dir)
    files = be.verify_models(args.models_dir, args.seeds)       # sha256 + smoke_test + class order
    sources = {s: tf.keras.models.load_model(str(f), compile=False) for s, f in files.items()}
    checks["1_sha256"] = "PASS"
    checks["1_manifest_arm_R_recentre"] = "PASS"
    print(f"    PASS — {len(files)} files match manifest sha256; arm R, input = re-centred")

    print(f"[2] building {NAME} from seeds {args.seeds} ...")
    e5r, debug = be.build_e5(tf, sources)
    n_params = int(e5r.count_params())
    print(f"    {NAME} parameters: {n_params:,} (expected 22,555 = 5 × 4,511, same as E5)")

    print(f"[3] checking {NAME} against the 5 original models on RE-CENTRED {args.check_data} ...")
    ds = tr.load_dataset(args.check_data)
    X_raw = ds.X.astype(np.float32)
    X = rc.recentre(X_raw).astype(np.float32)                    # ★ จุดเดียวที่ต่างจาก E5
    moved = int((np.abs(X - X_raw).max(axis=(1, 2, 3)) > 0).sum())
    checks["3_frames_moved_by_recentre"] = moved
    print(f"    re-centring moved {moved:,} of {len(X):,} frames (the rest were already centred)")
    outs = debug.predict(X, batch_size=512, verbose=0)
    branch, avg = outs[:-1], outs[-1]
    worst_branch, orig = 0.0, []
    for (s, m), b in zip(sources.items(), branch):
        p = m.predict(X, batch_size=512, verbose=0)
        orig.append(p)
        worst_branch = max(worst_branch, float(np.abs(p - b).max()))
    worst_avg = float(np.abs(np.mean(orig, axis=0) - avg).max())
    checks["3_branch_max_abs_diff"] = worst_branch
    checks["3_mean_max_abs_diff"] = worst_avg
    if worst_branch > be.ATOL or worst_avg > be.ATOL:
        raise SystemExit(f"{NAME} does not reproduce the originals: branch {worst_branch:.2e}, mean {worst_avg:.2e} > {be.ATOL}")
    print(f"    PASS — {len(X):,} frames, max |diff| branch {worst_branch:.1e}, mean {worst_avg:.1e}")
    e5r.save(str(out_dir / f"{NAME}.keras"))

    print("[4] converting to .tflite (float32, no quantisation) and checking vs .keras, batch 1 ...")
    blob, how = be.to_tflite(tf, e5r)
    (out_dir / f"{NAME}.tflite").write_bytes(blob)
    checks["4_method"] = how
    rng = np.random.default_rng(be.SAMPLE_SEED)
    sub = np.sort(rng.choice(len(X), size=min(args.n_tflite_check, len(X)), replace=False))
    pk = e5r.predict(X[sub], batch_size=512, verbose=0)
    pt = be.tflite_predict(tf, blob, X[sub])
    agree = float((pk.argmax(1) == pt.argmax(1)).mean())
    diff = float(np.abs(pk - pt).max())
    checks["4_argmax_agree"] = agree
    checks["4_max_abs_diff"] = diff
    if agree < 1.0 or diff > be.ATOL:
        raise SystemExit(f"{NAME}: tflite differs from keras (agree {agree:.4f}, max |diff| {diff:.2e})")
    print(f"    PASS — {NAME}.tflite {len(blob) / 1024:.1f} KiB ({how.split(' (')[0]}), "
          f"{agree * 100:.0f} % same class on {len(sub)} frames, max |diff| {diff:.1e}")

    print(f"[5] saving {be.N_BOARD_CHECK} re-centred board-check frames + {NAME} answers ...")
    idx = be.pick_board_samples(ds.y, be.N_BOARD_CHECK, be.SAMPLE_SEED)
    np.save(out_dir / "board_check_inputs_R_200.npy", X[idx])
    np.save(out_dir / f"board_check_{NAME}_probs_200.npy", be.tflite_predict(tf, blob, X[idx]))
    checks["5_classes_in_sample"] = {c: int((ds.y[idx] == i).sum()) for i, c in enumerate(tr.CLASSES)}

    produced = sorted(p for p in out_dir.iterdir() if p.is_file() and p.suffix in (".keras", ".tflite", ".npy"))
    rep = {"run_at": datetime.now().isoformat(timespec="seconds"), "git_commit": ev.git_commit(),
           "name": NAME, "seeds": args.seeds, "params": n_params, "class_order": list(tr.CLASSES),
           "input_transform": man.get("input_transform"), "check_data": str(args.check_data),
           "n_check_frames": int(len(X)), "checks": checks,
           "source_models": {m["model_file"]: m["model_sha256"] for m in man["models"]},
           "files": {p.name: ev.sha256_file(p) for p in produced}}
    (out_dir / "build_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    L = [ev.LINE, f"  Phase 5.5c build  ·  {NAME} ensemble (re-centred input)  ·  §5.1w/§5.1x", ev.LINE,
         f"  {NAME} parameters     {n_params:,}",
         f"  {NAME} == mean of 5   PASS  (max |diff| {worst_avg:.1e} over {len(X):,} re-centred frames)",
         f"  {NAME}.tflite         PASS  ({len(blob) / 1024:.1f} KiB, same class on {agree * 100:.0f} % of {len(sub)} frames)",
         f"  board-check set      {be.N_BOARD_CHECK} re-centred frames saved (board_check_inputs_R_200.npy)",
         "  No accuracy computed: check data is in the training set.",
         f"  Output: {out_dir}", ev.LINE]
    text = "\n".join(L)
    print("\n" + text)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    return rep


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ชี้ --models-dir ไปที่ results_phase55_final (โมเดลของ E5) โดยไม่ตั้งใจ → สคริปต์หยุดเพราะ manifest ไม่ใช่แขน R
#  2. ไฟล์ board_check_inputs_R_200.npy คือข้อมูล "re-centre แล้ว" — ใช้กับ E5R เท่านั้น ห้ามใช้กับ E5
#     (ชื่อไฟล์ต่างจากของ E5 ตั้งใจไว้ กันสลับ)
#  3. ตัวแปลง TFLite: ใช้ build_ensemble.to_tflite ตัวเดียวกับที่แปลง E5 สำเร็จเมื่อ 20 ต.ค. (from_keras_model)
#  4. ห้ามเอาตัวเลขใดๆ จากสคริปต์นี้ไปรายงานเป็นความแม่นยำ — check-data อยู่ในชุดเทรน
#  5. ลำดับคลาสต้องเป็น Fist, FlatHand, Dislike, Like, Love, CrossHands, BreakTime (§9 #38) — ตรวจใน verify_models แล้ว
# ─────────────────────────────────────────────────────────────────────────
