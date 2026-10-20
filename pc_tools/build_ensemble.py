"""build_ensemble.py — Phase 5.5b ขั้น 2.1–2.2: รวมโมเดลสูตร B ทั้ง 5 ตัว (seed 42–46) เป็นโมเดลเดียว "E5"
ที่เอาความมั่นใจ (softmax) ของ 5 ตัวมาเฉลี่ยกัน แล้วแปลงเป็น .tflite สำหรับ X-CUBE-AI — ตาม PROJECT_FACTS §5.1l

  ขั้นที่สคริปต์ทำ (หยุดทันทีถ้าข้อไหนไม่ผ่าน):
  1. ตรวจว่าไฟล์โมเดลทั้ง 5 ตรงกับ sha256 ใน results_phase55_final\\manifest.json (ไม่ได้ถูกแก้/สลับ)
  2. สร้าง E5 เป็นกราฟเดียว: 5 สาขาโครงเดียวกับ train_st_cnn2d.build_model (ตัด Dropout ออก เพราะตอนทาย
     Dropout ไม่ทำอะไรอยู่แล้ว) คัดลอกน้ำหนักจากแต่ละโมเดล แล้วเฉลี่ยด้วยชั้น Average
  3. ตรวจ: ผลของแต่ละสาขา == ผลของโมเดลต้นฉบับ และ ผลของ E5 == ค่าเฉลี่ยของ 5 ตัว (ทุกเฟรมใน --check-data)
  4. แปลง E5 และ seed 42 (ตัวสำรองตาม §5.1l) เป็น .tflite แบบ float32 (ไม่ quantize เหมือน Phase 3)
  5. ตรวจ: .tflite ทายท่าตรงกับ .keras ทีละเฟรม (batch 1 เหมือนบนบอร์ด)
  6. เซฟตัวอย่าง 200 เฟรม (สุ่มแบบมีทุกท่า) + คำตอบที่ถูกต้องของ PC ไว้เทียบกับบอร์ดในขั้น 2.4

  ⚠️ สคริปต์นี้ "ไม่คำนวณความแม่นยำ" — data\\phase4 อยู่ในชุดเทรนแล้ว ใช้ได้แค่เช็กว่าการรวม/แปลงไม่ทำให้ผลเพี้ยน

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\build_ensemble.py --models-dir results_phase55_final --check-data data\\phase4

ผลลัพธ์: results_phase55_final\\deploy\\
    E5.keras, E5.tflite              โมเดลรวม (ตัวหลัก)
    seed42.tflite                    ตัวสำรอง
    board_check_inputs_200.npy       ข้อมูล 200 เฟรม (N, 8, 8, 2) float32 สำหรับเทียบบอร์ด
    board_check_E5_probs_200.npy     คำตอบของ E5 บน PC (N, 7)   + ไฟล์ของ seed 42 แบบเดียวกัน
    build_report.json, report.txt    ผลการตรวจทุกข้อ + sha256 ของไฟล์ที่สร้าง
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_phase4 as ev  # noqa: E402

ATOL = 1e-5                  # ความต่างสูงสุดที่ยอมได้ระหว่าง keras กับของที่รวม/แปลงแล้ว (float32)
N_BOARD_CHECK = 200          # จำนวนเฟรมเทียบบอร์ด ตาม §5.1l
SAMPLE_SEED = 2026           # seed สำหรับสุ่มเฟรมตัวอย่าง (ไม่เกี่ยวกับ seed ของโมเดล)


def verify_models(models_dir: Path, seeds: list[int]) -> dict[int, Path]:
    """ข้อ 1: ไฟล์โมเดลต้องมีครบและ sha256 ตรงกับ manifest.json ที่สคริปต์เทรนเขียนไว้"""
    man = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    if man.get("smoke_test"):
        raise SystemExit("manifest says smoke_test = true — these models must not be used")
    want = {int(m["seed"]): m for m in man["models"]}
    out = {}
    for s in seeds:
        if s not in want:
            raise SystemExit(f"seed {s} not in manifest.json")
        f = models_dir / want[s]["model_file"]
        if not f.is_file():
            raise SystemExit(f"missing {f}")
        got = ev.sha256_file(f)
        if got != want[s]["model_sha256"]:
            raise SystemExit(f"sha256 mismatch for {f.name}: file {got[:16]} vs manifest {want[s]['model_sha256'][:16]}")
        out[s] = f
    if man.get("class_order") != ev.MODEL_CLASSES:
        raise SystemExit(f"class order in manifest differs: {man.get('class_order')}")
    return out


def weight_layers(model) -> list:
    """ชั้นที่มีน้ำหนักของโมเดลต้นฉบับ ตามลำดับ: Conv2D, Dense(32), Dense(7)"""
    ls = [l for l in model.layers if l.get_weights()]
    kinds = [type(l).__name__ for l in ls]
    if kinds != ["Conv2D", "Dense", "Dense"]:
        raise SystemExit(f"unexpected architecture {kinds} — expected Conv2D, Dense, Dense (train_st_cnn2d.build_model)")
    return ls


def build_e5(tf, sources: dict[int, object]):
    """ข้อ 2: สร้าง E5 = กราฟเดียว 5 สาขา + Average คืน (E5, debug model ที่ให้ผลทุกสาขา)"""
    L = tf.keras.layers
    inp = L.Input(shape=(8, 8, 2), name="input")
    branch_out = []
    for s, m in sources.items():
        conv_w, d1_w, d2_w = (l.get_weights() for l in weight_layers(m))
        src_conv = weight_layers(m)[0]
        conv = L.Conv2D(8, 3, padding=src_conv.padding, name=f"s{s}_conv")
        d1 = L.Dense(32, activation="relu", name=f"s{s}_dense32")
        d2 = L.Dense(len(ev.MODEL_CLASSES), activation="softmax", name=f"s{s}_softmax")
        x = conv(inp)
        x = L.Activation("relu", name=f"s{s}_relu")(x)
        x = L.MaxPooling2D(2, name=f"s{s}_pool")(x)
        x = L.Flatten(name=f"s{s}_flat")(x)
        x = d1(x)
        y = d2(x)
        conv.set_weights(conv_w)
        d1.set_weights(d1_w)
        d2.set_weights(d2_w)
        branch_out.append(y)
    avg = L.Average(name="ensemble_mean")(branch_out)
    e5 = tf.keras.Model(inp, avg, name="E5")
    debug = tf.keras.Model(inp, branch_out + [avg], name="E5_debug")
    return e5, debug


def to_tflite(tf, model) -> tuple[bytes, str]:
    """ข้อ 4: แปลงเป็น .tflite float32 — ลอง from_keras_model ก่อน ไม่ได้ค่อยผ่าน SavedModel"""
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        return conv.convert(), "from_keras_model"
    except Exception as exc:  # noqa: BLE001 — ตัวแปลงของ TF โยน error ได้หลายชนิด จึงจับกว้างแล้วลองอีกทาง
        first = f"{type(exc).__name__}: {exc}"
    with tempfile.TemporaryDirectory() as d:
        try:
            model.export(d)                     # Keras 3: เขียน SavedModel สำหรับ inference
            conv = tf.lite.TFLiteConverter.from_saved_model(d)
            return conv.convert(), f"export+from_saved_model (from_keras_model failed: {first[:200]})"
        except Exception as exc2:  # noqa: BLE001
            raise SystemExit(f"tflite conversion failed both ways:\n  1) {first}\n  2) {type(exc2).__name__}: {exc2}")


def tflite_predict(tf, blob: bytes, X: np.ndarray) -> np.ndarray:
    """ข้อ 5: รัน .tflite ทีละเฟรม (batch 1 เหมือนบนบอร์ด)"""
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    i_in, i_out = it.get_input_details()[0], it.get_output_details()[0]
    if tuple(i_in["shape"]) != (1, 8, 8, 2):
        raise SystemExit(f"tflite input shape {i_in['shape']} != (1, 8, 8, 2)")
    out = np.zeros((len(X), len(ev.MODEL_CLASSES)), np.float32)
    for k in range(len(X)):
        it.set_tensor(i_in["index"], X[k:k + 1].astype(np.float32))
        it.invoke()
        out[k] = it.get_tensor(i_out["index"])[0]
    return out


def pick_board_samples(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """ข้อ 6: เลือก n เฟรม ให้ทุกท่ามีจำนวนใกล้กัน (สุ่มด้วย seed คงที่)"""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per = int(np.ceil(n / len(classes)))
    idx = np.concatenate([rng.choice(np.where(y == c)[0], size=min(per, int((y == c).sum())), replace=False)
                          for c in classes])
    return np.sort(rng.permutation(idx)[:n])


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=Path("results_phase55_final"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--fallback-seed", type=int, default=42, help="ตัวสำรองตาม §5.1l")
    ap.add_argument("--check-data", type=Path, required=True, help=r"data\phase4 — ใช้เช็กความเท่ากันเท่านั้น")
    ap.add_argument("--n-tflite-check", type=int, default=2000, help="จำนวนเฟรมที่เทียบ tflite กับ keras")
    ap.add_argument("--train-script", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help=r"ค่าเริ่มต้น <models-dir>\deploy")
    args = ap.parse_args(argv)

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
    files = verify_models(args.models_dir, args.seeds)
    sources = {s: tf.keras.models.load_model(str(f), compile=False) for s, f in files.items()}
    checks["1_sha256"] = "PASS"
    print("    PASS — 5 files match manifest sha256")

    print(f"[2] building E5 from seeds {args.seeds} ...")
    e5, debug = build_e5(tf, sources)
    n_params = int(e5.count_params())
    print(f"    E5 parameters: {n_params:,} (= {len(args.seeds)} × {n_params // len(args.seeds):,})")

    print(f"[3] checking E5 against the 5 original models on {args.check_data} ...")
    ds = tr.load_dataset(args.check_data)
    X = ds.X.astype(np.float32)
    outs = debug.predict(X, batch_size=512, verbose=0)
    branch, avg = outs[:-1], outs[-1]
    worst_branch = 0.0
    orig = []
    for (s, m), b in zip(sources.items(), branch):
        p = m.predict(X, batch_size=512, verbose=0)
        orig.append(p)
        worst_branch = max(worst_branch, float(np.abs(p - b).max()))
    mean5 = np.mean(orig, axis=0)
    worst_avg = float(np.abs(mean5 - avg).max())
    checks["3_branch_max_abs_diff"] = worst_branch
    checks["3_mean_max_abs_diff"] = worst_avg
    if worst_branch > ATOL or worst_avg > ATOL:
        raise SystemExit(f"E5 does not reproduce the originals: branch {worst_branch:.2e}, mean {worst_avg:.2e} > {ATOL}")
    print(f"    PASS — {len(X):,} frames, max |diff| branch {worst_branch:.1e}, mean {worst_avg:.1e}")
    e5.save(str(out_dir / "E5.keras"))

    print("[4] converting to .tflite (float32, no quantisation) ...")
    blobs = {}
    for name, model in (("E5", e5), (f"seed{args.fallback_seed}", sources[args.fallback_seed])):
        blob, how = to_tflite(tf, model)
        (out_dir / f"{name}.tflite").write_bytes(blob)
        blobs[name] = (blob, model)
        checks[f"4_{name}_method"] = how
        print(f"    {name}.tflite  {len(blob) / 1024:.1f} KiB  ({how.split(' (')[0]})")

    print(f"[5] checking .tflite vs .keras, batch 1, {args.n_tflite_check} frames ...")
    rng = np.random.default_rng(SAMPLE_SEED)
    sub = np.sort(rng.choice(len(X), size=min(args.n_tflite_check, len(X)), replace=False))
    for name, (blob, model) in blobs.items():
        pk = model.predict(X[sub], batch_size=512, verbose=0)
        pt = tflite_predict(tf, blob, X[sub])
        agree = float((pk.argmax(1) == pt.argmax(1)).mean())
        diff = float(np.abs(pk - pt).max())
        checks[f"5_{name}_argmax_agree"] = agree
        checks[f"5_{name}_max_abs_diff"] = diff
        if agree < 1.0 or diff > ATOL:
            raise SystemExit(f"{name}: tflite differs from keras (agree {agree:.4f}, max |diff| {diff:.2e})")
        print(f"    PASS — {name}: {agree * 100:.0f} % same class, max |diff| {diff:.1e}")

    print(f"[6] saving {N_BOARD_CHECK} board-check frames ...")
    idx = pick_board_samples(ds.y, N_BOARD_CHECK, SAMPLE_SEED)
    np.save(out_dir / "board_check_inputs_200.npy", X[idx])
    for name, (blob, _) in blobs.items():
        np.save(out_dir / f"board_check_{name}_probs_200.npy", tflite_predict(tf, blob, X[idx]))
    checks["6_classes_in_sample"] = {c: int((ds.y[idx] == i).sum()) for i, c in enumerate(tr.CLASSES)}

    produced = sorted(p for p in out_dir.iterdir() if p.is_file() and p.suffix in (".keras", ".tflite", ".npy"))
    rep = {"run_at": datetime.now().isoformat(timespec="seconds"), "git_commit": ev.git_commit(),
           "seeds": args.seeds, "fallback_seed": args.fallback_seed, "e5_params": n_params,
           "class_order": list(tr.CLASSES), "check_data": str(args.check_data), "n_check_frames": int(len(X)),
           "checks": checks, "files": {p.name: ev.sha256_file(p) for p in produced}}
    (out_dir / "build_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    L = [ev.LINE, "  Phase 5.5b build  ·  E5 ensemble + fallback seed 42  ·  §5.1l", ev.LINE,
         f"  E5 parameters      {n_params:,}",
         f"  E5 == mean of 5    PASS  (max |diff| {worst_avg:.1e} over {len(X):,} frames)"]
    for name in blobs:
        L.append(f"  {name + '.tflite':<18} PASS  ({len(blobs[name][0]) / 1024:.1f} KiB, same class on "
                 f"{checks[f'5_{name}_argmax_agree'] * 100:.0f} % of {len(sub)} frames)")
    L += [f"  board-check set    {N_BOARD_CHECK} frames saved", "  No accuracy computed: check data is in the training set.",
          f"  Output: {out_dir}", ev.LINE]
    text = "\n".join(L)
    print("\n" + text)
    (out_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    return rep


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ตัวแปลง TFLite ของ TF 2.21 กับ Keras 3 — ถ้า from_keras_model ใช้ไม่ได้ สคริปต์จะลองผ่าน SavedModel
#     ให้อัตโนมัติ และบันทึกวิธีที่ใช้ไว้ใน build_report.json [จำมา — ยังไม่ได้ลองบนเครื่อง Chin]
#  2. tf.lite.Interpreter อาจขึ้นคำเตือนว่าจะย้ายไป LiteRT — เป็นแค่คำเตือน [จำมา]
#  3. ATOL 1e-5: ถ้าแปลงแล้วต่างเล็กน้อยกว่านี้แต่ทายท่าเดียวกัน 100 % จะไม่ผ่าน — ตั้งเข้มไว้ก่อน ถ้าไม่ผ่าน
#     ให้ดูตัวเลขก่อน อย่าคลายเกณฑ์โดยไม่บันทึกเหตุผล
#  4. ห้ามเอาตัวเลขใดๆ จากสคริปต์นี้ไปรายงานเป็นความแม่นยำ — check-data อยู่ในชุดเทรน
#  5. ลำดับคลาสต้องเป็น Fist, FlatHand, Dislike, Like, Love, CrossHands, BreakTime ใน firmware ด้วย (§9 #38)
# ─────────────────────────────────────────────────────────────────────────
