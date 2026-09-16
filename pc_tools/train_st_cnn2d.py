"""Phase 2 — train the ST st_cnn2d_handposture architecture on the public
ST_VL53L8CX_handposture_dataset, comparing a random split (reproducing ST's own
method, TRAP #8) against a subject-independent (leave-one-user-out) split.

โหลดข้อมูลตามฟอร์แมตที่ตรวจสอบไว้ใน PROJECT_FACTS.md §5.1 / §5.1b:
    *.npz per log, keys: start_tstmp, end_tstmp, zone_data, glob_data,
                         zone_head, glob_head
    zone_data shape = (4, 64, N_frames)
    zone_head = ['target_status', 'valid', 'signal_per_spad', 'distance_mm']
    glob_head = ['.GestureGT']
    folder name contains the user id, e.g. "...__User2__..."

Preprocessing (from PROJECT_FACTS §5.1, read out of ST's data_loader.py):
    Max_distance 400 mm, Min_distance 100 mm, Background_distance 120 mm
    valid_status = [5, 9]; invalid zone -> distance=4000, signal=0
    normalisation: (distance-295)/196, (signal-281)/452
    augmentation (train only): random horizontal flip

Architecture (read directly from ST's .keras file, PROJECT_FACTS §5.1):
    Input (8, 8, 2)
    Conv2D 8 filters 3x3 -> Activation -> MaxPooling2D 2x2 -> Dropout 0.2
    Flatten -> Dense 32 relu -> Dense N_CLASSES softmax
    (N_CLASSES = 7 here, not 8: the "None" class is dropped, TRAP #18)

Hyperparameters (from ST's data_loader.py, PROJECT_FACTS §5.1):
    batch 32, epochs up to 1000 (with early stopping added here since our
    public dataset is smaller than ST's internal one), Adam lr 0.01, seed 42

How to run:
    python train_st_cnn2d.py --data-dir /path/to/ST_VL53L8CX_handposture_dataset \
        --split both --out-dir results_phase2

Note this script does NOT reproduce ST's 98.47% — TRAP #16/#17 established that
ST trained on an internal dataset we do not have. This script trains on the
public 11,443-frame dataset only, which is what contribution #4 is about.

จุดที่อาจพลาด (แจ้งไว้ล่วงหน้า):
  - โฟลเดอร์แต่ละ log ต้องมีคำว่า "User<เลข>" อยู่ในชื่อ ไม่งั้น parse user id ไม่ได้
  - ถ้า TensorFlow เวอร์ชันต่างจากที่ทดสอบ (2.21.0) ผลตัวเลขอาจขยับเล็กน้อยแม้ seed เดียวกัน
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ต้องติดตั้ง tensorflow ก่อน: pip install tensorflow --break-system-packages"
    ) from exc

from sklearn.model_selection import GroupKFold, train_test_split

SEED = 42
CLASSES = [
    "Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime",
]  # "None" deliberately excluded — TRAP #18
N_CLASSES = len(CLASSES)

MAX_DISTANCE = 400.0
MIN_DISTANCE = 100.0
BACKGROUND_DISTANCE = 120.0
VALID_STATUS = {5, 9}
DIST_MEAN, DIST_STD = 295.0, 196.0
SIG_MEAN, SIG_STD = 281.0, 452.0


@dataclass
class LoadedDataset:
    """ผลลัพธ์จากการโหลดข้อมูลทั้งหมด พร้อม user id ต่อเฟรมสำหรับ split แบบ subject-independent"""

    X: np.ndarray  # (N, 8, 8, 2) float32, already normalised
    y: np.ndarray  # (N,) int, index into CLASSES
    groups: np.ndarray  # (N,) int, user id per frame
    dropped_none: int = 0
    dropped_out_of_range: int = 0
    files_read: int = 0


def _extract_user_id(folder_name: str) -> int:
    """ดึงหมายเลข user จากชื่อโฟลเดอร์ เช่น '...__User2__...' -> 2"""
    m = re.search(r"User(\d+)", folder_name)
    if not m:
        raise ValueError(
            f"หาหมายเลข user ไม่เจอในชื่อโฟลเดอร์ '{folder_name}' "
            "— ต้องมีคำว่า User<เลข> อยู่ในชื่อ"
        )
    return int(m.group(1))


def _preprocess_frame(distance_mm: np.ndarray, signal: np.ndarray,
                       status: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """แปลง 1 เฟรมดิบ (แต่ละอาร์เรย์ยาว 64) ให้เป็น (8,8,2) ตาม preprocessing ของ ST

    คืนค่า None ถ้าเฟรมนี้ต้องถูกทิ้ง (ระยะเกิน Max_distance ทั้งเฟรม)
    """
    nearest = distance_mm[valid.astype(bool) & np.isin(status, list(VALID_STATUS))]
    if nearest.size and nearest.min() > MAX_DISTANCE:
        return None  # ทั้งเฟรมอยู่ไกลเกินไป — ทิ้งตาม TRAP #16 (5/11448 ทิ้งแบบนี้)

    d = distance_mm.copy().astype(np.float32)
    s = signal.copy().astype(np.float32)
    bad = ~(np.isin(status, list(VALID_STATUS)) & valid.astype(bool))
    # โซนที่ไกลกว่าจุดที่ใกล้ที่สุด + Background_distance ก็ถือว่า background
    if nearest.size:
        bad |= d > (nearest.min() + BACKGROUND_DISTANCE)
    d[bad] = 4000.0
    s[bad] = 0.0

    d_norm = (d - DIST_MEAN) / DIST_STD
    s_norm = (s - SIG_MEAN) / SIG_STD
    return np.stack([d_norm.reshape(8, 8), s_norm.reshape(8, 8)], axis=-1)


def load_dataset(data_dir: Path) -> LoadedDataset:
    npz_files = sorted(data_dir.rglob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"ไม่เจอไฟล์ .npz เลยใน {data_dir} — เช็ค path ให้ตรงกับที่แตกซิปไว้"
        )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    g_list: list[int] = []
    dropped_none = 0
    dropped_range = 0

    for fp in npz_files:
        user_id = _extract_user_id(str(fp))
        # ชื่อคลาสอยู่ในชื่อโฟลเดอร์ระดับ dataset_dir/<ClassName>/log__.../npz/*.npz
        # ยืนยันจากการรันจริง 15 Sep 2026: glob_data เป็นตัวเลขรหัส ("27.0") ไม่ใช่
        # ข้อความชื่อคลาส ดังนั้นใช้ชื่อโฟลเดอร์แทน ซึ่งเชื่อถือได้กว่า
        class_folder = fp.parents[2].name  # npz -> log__... -> <ClassName>
        if class_folder not in CLASSES:
            with np.load(fp, allow_pickle=True) as z:
                n_frames_skip = z["zone_data"].shape[-1]
            dropped_none += n_frames_skip
            continue
        class_idx = CLASSES.index(class_folder)

        with np.load(fp, allow_pickle=True) as z:
            zone_data = z["zone_data"]  # (4, 64, N_frames)

        n_frames = zone_data.shape[-1]
        for f in range(n_frames):
            status = zone_data[0, :, f]
            valid = zone_data[1, :, f]
            signal = zone_data[2, :, f]
            distance = zone_data[3, :, f]
            frame = _preprocess_frame(distance, signal, status, valid)
            if frame is None:
                dropped_range += 1
                continue
            X_list.append(frame)
            y_list.append(class_idx)
            g_list.append(user_id)

    return LoadedDataset(
        X=np.stack(X_list).astype(np.float32),
        y=np.array(y_list, dtype=np.int64),
        groups=np.array(g_list, dtype=np.int64),
        dropped_none=dropped_none,
        dropped_out_of_range=dropped_range,
        files_read=len(npz_files),
    )


def build_model(seed: int) -> tf.keras.Model:
    """สถาปัตยกรรมเดียวกับ st_cnn2d_handposture เป๊ะ ยกเว้นเปลี่ยน output เป็น 7 คลาส"""
    tf.keras.utils.set_random_seed(seed)
    model = models.Sequential([
        layers.Input(shape=(8, 8, 2)),
        layers.Conv2D(8, 3, padding="same"),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def augment_flip(X: np.ndarray, seed: int) -> np.ndarray:
    """random horizontal flip ตามที่ ST ทำ — ใช้กับ train set เท่านั้น"""
    rng = np.random.default_rng(seed)
    flipped = X.copy()
    mask = rng.random(len(X)) < 0.5
    flipped[mask] = flipped[mask, :, ::-1, :]
    return flipped


def train_and_eval(X_train, y_train, X_test, y_test, label: str,
                    seed: int, max_epochs: int = 1000) -> dict:
    model = build_model(seed)
    X_train_aug = augment_flip(X_train, seed)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )
    history = model.fit(
        X_train_aug, y_train,
        batch_size=32, epochs=max_epochs,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = tf.math.confusion_matrix(y_test, y_pred, num_classes=N_CLASSES).numpy()
    return {
        "label": label,
        "seed": seed,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "epochs_ran": len(history.history["loss"]),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
    }


def train_multi_seed(X_train, y_train, X_test, y_test, label: str,
                      seeds: list[int]) -> dict:
    """เทรนซ้ำหลาย seed แล้วสรุป mean/SD — แยก 'ผลจากความสุ่มตอนเทรน'
    ออกจาก 'ผลจากข้อมูล/user จริง' ตามที่ตกลงกันไว้ 15 Sep 2026"""
    runs = [
        train_and_eval(X_train, y_train, X_test, y_test,
                        label=f"{label}_seed{s}", seed=s)
        for s in seeds
    ]
    accs = [r["test_accuracy"] for r in runs]
    return {
        "label": label,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "runs": runs,
        "mean_accuracy": float(np.mean(accs)),
        "sd_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "min_accuracy": float(np.min(accs)),
        "max_accuracy": float(np.max(accs)),
        # confusion matrix ของ run แรกไว้ดูตัวอย่าง (ไม่ใช่ค่าเฉลี่ย เพราะรวม matrix ข้าม seed ตรงๆ ไม่มีความหมาย)
        "sample_confusion_matrix": runs[0]["confusion_matrix"],
    }


def run_random_split(ds: LoadedDataset, seeds: list[int]) -> dict:
    """เลียนแบบวิธี ST เอง (TRAP #8): shuffle แล้วแบ่ง ไม่แยกตาม user
    split ทำครั้งเดียวด้วย SEED คงที่ (คนละเรื่องกับ seed ของการเทรน) เพื่อให้
    ทุก seed เทรน/ทดสอบบนข้อมูลชุดเดียวกันเป๊ะ ต่างกันแค่การสุ่มตอนเทรน"""
    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=0.2, random_state=SEED, stratify=ds.y
    )
    return train_multi_seed(X_tr, y_tr, X_te, y_te, label="random_split", seeds=seeds)


def equalize_train_size(X_train, y_train, target_n: int, seed: int = SEED):
    """สุ่มตัดข้อมูล train ลงมาให้เหลือ target_n เฟรม แบบ stratified ตามสัดส่วนคลาสเดิม
    ใช้แยกผลของ "ข้อมูล user ต่างกัน" ออกจากผลของ "training set ขนาดต่างกัน"
    """
    if len(X_train) <= target_n:
        return X_train, y_train  # fold นี้เล็กกว่าเป้าหมายอยู่แล้ว ไม่ต้องตัด
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(y_train == c)[0] for c in range(N_CLASSES)]
    frac = target_n / len(X_train)
    keep_idx = []
    for idx in idx_by_class:
        n_keep = max(1, round(len(idx) * frac))
        keep_idx.append(rng.choice(idx, size=min(n_keep, len(idx)), replace=False))
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)
    return X_train[keep_idx], y_train[keep_idx]


def run_subject_independent(ds: LoadedDataset, seeds: list[int],
                             equalize: bool = False) -> dict:
    """leave-one-user-out ทุก user (มี 4 user ใน public dataset)
    แต่ละ fold เทรนซ้ำหลาย seed แล้วเฉลี่ย ก่อนจะเฉลี่ยข้าม fold อีกที
    (สองชั้น: ชั้นในตัดผลของความสุ่มตอนเทรนออก ชั้นนอกคือผลต่างระหว่าง user จริง)

    equalize=True: ตัด training set ของทุก fold ให้เท่ากับ fold ที่เล็กที่สุด
    (stratified ตามคลาส) เพื่อแยกผลของ "user ต่างกัน" ออกจากผลของ
    "จำนวนข้อมูลฝึกต่างกัน" — ดู PROJECT_FACTS §9 correction log วันที่ 15 Sep
    """
    gkf = GroupKFold(n_splits=len(np.unique(ds.groups)))
    splits = list(gkf.split(ds.X, ds.y, groups=ds.groups))

    min_train_n = min(len(tr_idx) for tr_idx, _ in splits) if equalize else None
    if equalize:
        print(f"  [equalize] จะตัด training set ทุก fold ให้เหลือ {min_train_n} เฟรม")

    fold_results = []
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        held_out_user = int(np.unique(ds.groups[te_idx])[0])
        X_tr, y_tr = ds.X[tr_idx], ds.y[tr_idx]
        if equalize:
            X_tr, y_tr = equalize_train_size(X_tr, y_tr, min_train_n)
        res = train_multi_seed(
            X_tr, y_tr, ds.X[te_idx], ds.y[te_idx],
            label=f"fold{fold_i}_heldout_user{held_out_user}", seeds=seeds,
        )
        res["held_out_user"] = held_out_user
        res["n_train_before_equalize"] = int(len(tr_idx))
        res["n_train_used"] = int(len(X_tr))
        fold_results.append(res)

    fold_means = [r["mean_accuracy"] for r in fold_results]
    return {
        "label": "subject_independent_summary",
        "equalized": equalize,
        "n_folds": len(fold_results),
        "n_seeds_per_fold": len(seeds),
        "fold_results": fold_results,
        # ค่าเฉลี่ยข้าม fold ของ "ค่าเฉลี่ยข้าม seed ในแต่ละ fold" — สองชั้นตามที่ตั้งใจ
        "mean_of_fold_means": float(np.mean(fold_means)),
        "sd_of_fold_means": float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else 0.0,
    }


def run_production(ds: LoadedDataset, seed: int, save_path: Path) -> dict:
    """เทรนโมเดล 'ตัวจริง' จากข้อมูลทั้งหมด (ไม่มี held-out test set) แล้วบันทึกไฟล์
    .keras ไว้ใช้งานจริง — คนละจุดประสงค์กับ run_random_split/run_subject_independent
    ที่มีไว้วัดผล ไม่ใช่ไว้ deploy

    เตือน: เพราะไม่มี test set แยก จึงไม่มีตัวเลข accuracy รายงานตรงนี้
    ต้องอ้างอิงตัวเลขจาก subject-independent (79.73%) เป็นตัวประมาณคร่าวๆ
    ของสิ่งที่จะเจอกับ 'คนที่โมเดลไม่เคยเห็น' (เช่นมือคุณเอง)
    """
    model = build_model(seed)
    X_aug = augment_flip(ds.X, seed)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True
    )
    history = model.fit(
        X_aug, ds.y, batch_size=32, epochs=1000,
        validation_split=0.1, callbacks=[early_stop], verbose=0,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    return {
        "label": "production_model", "seed": seed,
        "n_train_total": int(len(ds.X)), "epochs_ran": len(history.history["loss"]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "saved_to": str(save_path),
        "note": "ไม่มี held-out test — ดูตัวเลข accuracy จริงจาก subject-independent "
                "run แทน (ประมาณการณ์ performance กับคนที่ไม่เคยเห็นมาก่อน)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, type=Path,
                     help="path ไปยังโฟลเดอร์ที่แตกซิป ST_VL53L8CX_handposture_dataset แล้ว")
    ap.add_argument("--split", choices=["random", "subject", "both", "production"],
                     default="both")
    ap.add_argument("--out-dir", type=Path, default=Path("results_phase2"))
    ap.add_argument("--equalize-train", action="store_true",
                     help="ตัด training set ทุก fold ของ subject-independent ให้เท่ากับ "
                          "fold ที่เล็กที่สุด (แยกผล 'user ต่างกัน' ออกจาก 'ข้อมูลน้อยกว่า')")
    ap.add_argument("--n-seeds", type=int, default=3,
                     help="จำนวนรอบเทรนซ้ำต่อ 1 การตั้งค่า (split/fold) ด้วย seed ต่างกัน "
                          "เพื่อแยกความแปรปรวนจากการสุ่มตอนเทรน ออกจากผลต่างของข้อมูลจริง "
                          "(ค่าเริ่มต้น 3 — น้อยสุดที่พอคำนวณ SD ได้อย่างมีความหมาย)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [SEED + i for i in range(args.n_seeds)]
    print(f"จะเทรนซ้ำ {args.n_seeds} รอบต่อการตั้งค่า ด้วย seeds = {seeds}")

    print(f"กำลังโหลดข้อมูลจาก {args.data_dir} ...")
    ds = load_dataset(args.data_dir)
    print(f"  อ่าน {ds.files_read} ไฟล์, ได้ {len(ds.X)} เฟรมที่ใช้ได้")
    print(f"  ทิ้งเพราะเป็นคลาส None/อื่นๆ: {ds.dropped_none}")
    print(f"  ทิ้งเพราะไกลเกิน Max_distance ทั้งเฟรม: {ds.dropped_out_of_range}")
    print(f"  จำนวน user: {sorted(np.unique(ds.groups).tolist())}")
    for i, c in enumerate(CLASSES):
        print(f"    {c:12s}: {int(np.sum(ds.y == i))} เฟรม")

    results: dict = {
        "base_seed": SEED, "n_seeds": args.n_seeds, "seeds_used": seeds,
        "classes": CLASSES, "tensorflow_version": tf.__version__,
        "n_total_frames": int(len(ds.X)), "dropped_none": ds.dropped_none,
        "dropped_out_of_range": ds.dropped_out_of_range,
    }

    if args.split == "production":
        print("\n=== เทรนโมเดล production (ข้อมูลครบ 4 คน, ไม่ split) ===")
        model_path = args.out_dir / "production_model.keras"
        results["production"] = run_production(ds, seed=seeds[0], save_path=model_path)
        print(f"  train accuracy สุดท้าย = {results['production']['final_train_accuracy']:.4f}")
        print(f"  บันทึกโมเดลไว้ที่ {model_path}")

    if args.split in ("random", "both"):
        print(f"\n=== เทรนแบบ random split ({args.n_seeds} seeds, เลียนแบบวิธี ST, TRAP #8) ===")
        results["random_split"] = run_random_split(ds, seeds=seeds)
        rs = results["random_split"]
        acc_list = ", ".join(f"{r['test_accuracy']:.4f}" for r in rs["runs"])
        print(f"  accuracy แต่ละ seed = [{acc_list}]")
        print(f"  mean = {rs['mean_accuracy']:.4f} ± {rs['sd_accuracy']:.4f}")

    if args.split in ("subject", "both"):
        print(f"\n=== เทรนแบบ subject-independent ({args.n_seeds} seeds/fold, leave-one-user-out) ===")
        results["subject_independent"] = run_subject_independent(
            ds, seeds=seeds, equalize=args.equalize_train
        )
        si = results["subject_independent"]
        for r in si["fold_results"]:
            print(f"  user{r['held_out_user']}: mean={r['mean_accuracy']:.4f} "
                  f"± {r['sd_accuracy']:.4f}  (range {r['min_accuracy']:.4f}-{r['max_accuracy']:.4f}, "
                  f"n_train={r['n_train_used']})")
        print(f"  ภาพรวมข้าม fold: {si['mean_of_fold_means']:.4f} "
              f"± {si['sd_of_fold_means']:.4f} (n={si['n_folds']} folds)")

    out_path = args.out_dir / "phase2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nบันทึกผลไว้ที่ {out_path}")


if __name__ == "__main__":
    main()
