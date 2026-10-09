"""evaluate_phase4.py — Phase 4 (H8): โมเดลที่เทรนจากข้อมูล ST อย่างเดียว
ทายข้อมูลที่เก็บจากบอร์ดเรา (VL53L8CA) ได้แม่นแค่ไหน — ทดสอบอย่างเดียว ไม่เทรน

ทำตาม pre-registration ใน PROJECT_FACTS §5.1e (ลงทะเบียน 21 ก.ย. 2026 ก่อนเก็บข้อมูล):
  - โมเดล: results_phase2_final\\production_model.keras (§5.1d-F) — บันทึก sha256 ทุกครั้ง
  - preprocessing: st_dataset._preprocess_session() (กฎของ ST, ทิ้งเฟรมที่มืออยู่นอก 100-400 มม.)
  - ป้ายท่าจับคู่ด้วย "ชื่อ" ไม่ใช่เลข index (§9 #38) — ป้ายจริงมาจากชื่อโฟลเดอร์ (เจตนา, A6)
    และตรวจซ้ำกับรหัสใน glob_data ของไฟล์
  - หน่วยวิเคราะห์ = คน (subject) ไม่ใช่เฟรม (เฟรมในรอบเดียวกันสัมพันธ์กันสูง)
  - ตัววัด: per-frame accuracy, 15-frame majority vote accuracy (แบบเดียวกับ live_predict.py),
    per-class recall, confusion matrix, top-5 confusion pairs, P1/P2/P3/H9 และ decision rule
  - ใช้กับ data\\phase4 (ชุดวัดผล) เท่านั้น — data\\phase4_extra เป็นชุดเทรน (A7) สคริปต์จะไม่ยอม
  - n = 1-2 คน = pilot ผลเป็นเชิงพรรณนา ห้ามเขียนว่า "significant" (§5.1e, A1)

การตรวจความถูกต้องในตัว (ชัวร์ว่าตัวเลขไม่ผิดเพราะเตรียมข้อมูลต่างจากตอนเทรน):
  - preprocess ทุกเฟรมซ้ำด้วยโค้ดชุดที่ใช้ตอนเทรนโมเดลจริง (คัดลอกจาก
    train_st_cnn2d._preprocess_frame ทุกบรรทัด) แล้วเทียบกับ st_dataset ว่าได้ค่าเท่ากัน
    (ต่างได้แค่ระดับปัดเศษ float32 ≤ FLOAT_TOL เพราะสองโค้ดแปลงชนิดตัวเลขคนละจังหวะ)
    และนับเฟรมที่สองวิธีเก็บ/ทิ้งต่างกัน (ต่างได้แค่กฎ 100 มม. ด้านล่าง ซึ่งข้อมูลเราไม่ถึง)
  - ตรวจ manifest_collect.csv: จำนวนไฟล์ตรงกับแถว, missing_frames รวม, out_of_window รวม

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\evaluate_phase4.py --data-dir data\\phase4 --model results_phase2_final\\production_model.keras

ผลลัพธ์ (โฟลเดอร์ใหม่ทุกครั้ง ไม่ทับของเดิม):
    results_phase4\\eval_<เวลา>\\summary.json        ตัวเลขทั้งหมด + ข้อมูล reproducibility
    results_phase4\\eval_<เวลา>\\per_frame.csv       ทุกเฟรม: ไฟล์, เฟรม, ท่าจริง, ท่าที่ทาย, ความมั่นใจ
                                                    (ใช้ทำ error analysis ต่อ)
    results_phase4\\eval_<เวลา>\\per_take.csv        ความแม่นต่อรอบ
    results_phase4\\eval_<เวลา>\\confusion_User<N>.png  (ถ้ามี matplotlib)
    results_phase4\\eval_<เวลา>\\report.txt          สำเนาข้อความที่พิมพ์บนจอ

ไม่มีการสุ่มในสคริปต์นี้ (ทดสอบอย่างเดียว) — รันซ้ำได้ผลเท่าเดิมทุกครั้งกับโมเดลและข้อมูลเดิม
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np

# ลำดับ output ของโมเดล = ลำดับใน train_st_cnn2d.py (ตรวจแล้ว 21 ก.ย. 2026, §5.1e pre-flight 3)
MODEL_CLASSES = ["Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime"]
VOTE_WINDOW = 15                     # เท่ากับค่าเริ่มต้นของ live_predict.py
CHANCE = 1 / len(MODEL_CLASSES)      # 14.3 %
ST_LOSO = {"User1": 68.91, "User3": 78.31, "User4": 81.97, "User2": 89.71}  # §5.1d, mean 79.73
P1_BAND = (60.0, 85.0)               # §5.1e P1
HIGH_GROUP = ["Love", "CrossHands", "BreakTime"]          # P2
LOW_GROUP = ["Fist", "FlatHand", "Dislike", "Like"]       # P2 / H9
STOP_BELOW = 60.0                    # decision rule: ต่ำกว่านี้ = หยุดหาสาเหตุ
ST_LOWEST = 68.91                    # decision rule: ถึงค่านี้ = อยู่ในช่วงของ ST

# ค่าคงที่ของโค้ดตอนเทรน (train_st_cnn2d.py) — ใช้เฉพาะการตรวจซ้ำ
TR_MAX_DISTANCE = 400.0
TR_BACKGROUND_DISTANCE = 120.0
TR_VALID_STATUS = {5, 9}
TR_DIST_MEAN, TR_DIST_STD = 295.0, 196.0
TR_SIG_MEAN, TR_SIG_STD = 281.0, 452.0

LINE = "═" * 66
# โค้ดตอนเทรนแปลงเป็น float32 ก่อน normalize ส่วน st_dataset normalize ก่อนแล้วค่อยแปลง
# ต่างกันระดับปัดเศษ (~3e-7 หน่วย normalize ≈ 0.00006 มม.) — เกินค่านี้ถือว่าเตรียมข้อมูลต่างกันจริง
FLOAT_TOL = 1e-5
TAKE_RE = re.compile(r"take(\d+)")
USER_RE = re.compile(r"__User(\d+)__")


# ---------------------------------------------------------------- โหลด st_dataset
def find_st_dataset(explicit: Path | None) -> Path:
    """หา st_dataset.py: ใช้ path ที่ระบุ หรือค้นในโฟลเดอร์สคริปต์ / root ของ repo"""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"--st-dataset: file not found: {explicit}")
        return explicit
    here = Path(__file__).resolve().parent
    for cand in (here / "st_dataset.py", here.parent / "st_dataset.py"):
        if cand.is_file():
            return cand
    for cand in sorted(here.parent.rglob("st_dataset.py")):
        if ".git" not in cand.parts:
            return cand
    raise SystemExit("st_dataset.py not found — locate it with:  dir /s /b st_dataset.py\n"
                     "then add  --st-dataset <that path>")


def import_module_from(path: Path, name: str) -> ModuleType:
    """import ไฟล์ .py จาก path ตรงๆ (ไม่ต้องอยู่ใน sys.path)"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    # ต้องลงทะเบียนใน sys.modules ก่อนรัน ไม่งั้น @dataclass ในไฟล์ที่ใช้
    # "from __future__ import annotations" จะหาโมดูลตัวเองไม่เจอแล้วพัง (บั๊ก 9 ต.ค. 2026)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- ข้อมูลต่อรอบ
@dataclass
class TakeData:
    """ข้อมูล 1 รอบหลัง preprocess พร้อมข้อมูลประกอบไว้วิเคราะห์ต่อ"""
    path: Path
    user: str
    cls: str
    take: int
    X: np.ndarray                 # (n, 8, 8, 2) จาก st_dataset
    raw_index: np.ndarray         # ตำแหน่งเฟรมในไฟล์ดิบของแต่ละแถวใน X
    nearest_mm: np.ndarray        # ระยะมือของแต่ละแถวใน X
    n_raw: int
    crosscheck: dict = field(default_factory=dict)
    pred: np.ndarray | None = None
    conf: np.ndarray | None = None


def train_style_frame(distance_mm: np.ndarray, signal: np.ndarray,
                      status: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """คัดลอกจาก train_st_cnn2d._preprocess_frame ทุกบรรทัด (โค้ดที่ใช้เทรนโมเดลจริง)
    ใช้ตรวจว่า st_dataset เตรียมข้อมูลเหมือนตอนเทรน — ห้ามแก้ตรงนี้ถ้าไม่ได้แก้ตัวต้นฉบับ"""
    nearest = distance_mm[valid.astype(bool) & np.isin(status, list(TR_VALID_STATUS))]
    if nearest.size and nearest.min() > TR_MAX_DISTANCE:
        return None
    d = distance_mm.copy().astype(np.float32)
    s = signal.copy().astype(np.float32)
    bad = ~(np.isin(status, list(TR_VALID_STATUS)) & valid.astype(bool))
    if nearest.size:
        bad |= d > (nearest.min() + TR_BACKGROUND_DISTANCE)
    d[bad] = 4000.0
    s[bad] = 0.0
    d_norm = (d - TR_DIST_MEAN) / TR_DIST_STD
    s_norm = (s - TR_SIG_MEAN) / TR_SIG_STD
    return np.stack([d_norm.reshape(8, 8), s_norm.reshape(8, 8)], axis=-1)


def load_take(path: Path, std: ModuleType) -> TakeData:
    """โหลด 1 ไฟล์ .npz: preprocess ด้วย st_dataset + ตรวจซ้ำด้วยโค้ดตอนเทรน + ตรวจป้าย"""
    cls = path.parents[2].name
    if cls not in MODEL_CLASSES:
        raise SystemExit(f"unknown posture folder '{cls}' for {path}")
    m_user = USER_RE.search(str(path))
    m_take = TAKE_RE.search(path.parents[1].name)
    if m_user is None or m_take is None:
        raise SystemExit(f"cannot read User/take from folder name: {path.parents[1].name}")
    user = f"User{m_user.group(1)}"

    with np.load(path, allow_pickle=True) as z:
        zone = z["zone_data"]
        glob = z["glob_data"]
    codes = set(np.unique(glob[~np.isnan(glob)]).astype(int).tolist())
    expected = std.HAND_POSTURE_DICT[cls]
    if codes != {expected}:
        raise SystemExit(f"label mismatch in {path}: folder says {cls} (code {expected}), "
                         f"glob_data has {sorted(codes)}")

    X, y_raw = std._preprocess_session(zone, glob, std.MAX_DISTANCE_MM, std.MIN_DISTANCE_MM,
                                       std.BACKGROUND_DISTANCE_MM)
    # หาว่าแถวใน X มาจากเฟรมไหน (ทำตามเงื่อนไขเดียวกับ st_dataset แล้วนับให้ตรงกัน)
    nan_mask = np.isnan(zone).any(axis=(0, 1)) | np.isnan(glob).any(axis=0)
    status, dist = zone[std.CH_TARGET_STATUS], zone[std.CH_DISTANCE]
    valid = np.isin(status, std.VALID_STATUS)
    hand = np.where(valid, dist, std.DEFAULT_DISTANCE).min(axis=0)
    keep = (~nan_mask) & (hand >= std.MIN_DISTANCE_MM) & (hand <= std.MAX_DISTANCE_MM)
    raw_index = np.nonzero(keep)[0]
    if len(raw_index) != len(X):
        raise SystemExit(f"internal check failed: frame bookkeeping mismatch in {path}")

    # ตรวจซ้ำกับโค้ดตอนเทรน
    n_raw = zone.shape[2]
    tr_frames: dict[int, np.ndarray] = {}
    for f in range(n_raw):
        if nan_mask[f]:
            continue
        fr = train_style_frame(zone[3, :, f], zone[2, :, f], zone[0, :, f], zone[1, :, f])
        if fr is not None:
            tr_frames[f] = fr
    common = [i for i, f in enumerate(raw_index) if f in tr_frames]
    max_diff = 0.0
    if common:
        a = X[common]
        b = np.stack([tr_frames[int(raw_index[i])] for i in common])
        max_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    only_train = sorted(set(tr_frames) - set(raw_index.tolist()))
    only_st = sorted(set(raw_index.tolist()) - set(tr_frames))
    return TakeData(path=path, user=user, cls=cls, take=int(m_take.group(1)), X=X.astype(np.float32),
                    raw_index=raw_index, nearest_mm=hand[raw_index], n_raw=n_raw,
                    crosscheck={"max_abs_diff": max_diff, "kept_only_by_training_code": len(only_train),
                                "kept_only_by_st_dataset": len(only_st)})


# ---------------------------------------------------------------- ตัววัด
def vote_sequence(pred: np.ndarray, window: int = VOTE_WINDOW) -> np.ndarray:
    """majority vote แบบหน้าต่างเลื่อน เหมือน live_predict.py ทุกอย่าง
    (deque ยาว window, Counter.most_common(1) — เสมอกันเลือกท่าที่เจอก่อนในหน้าต่าง)
    เริ่มนับเมื่อหน้าต่างเต็ม และเริ่มใหม่ทุกรอบ (ไม่ลากข้ามรอบ)"""
    buf: deque[int] = deque(maxlen=window)
    out: list[int] = []
    for p in pred:
        buf.append(int(p))
        if len(buf) == window:
            out.append(Counter(buf).most_common(1)[0][0])
    return np.array(out, dtype=int)


def subject_metrics(takes: list[TakeData]) -> dict:
    """ตัววัดทั้งหมดของผู้ทดลอง 1 คน"""
    k = len(MODEL_CLASSES)
    cm = np.zeros((k, k), dtype=int)
    vote_correct = vote_total = 0
    per_take = []
    for t in takes:
        ti = MODEL_CLASSES.index(t.cls)
        for p in t.pred:
            cm[ti, int(p)] += 1
        v = vote_sequence(t.pred)
        vote_correct += int((v == ti).sum())
        vote_total += len(v)
        n = len(t.pred)
        per_take.append({
            "file": t.path.name, "class": t.cls, "take": t.take, "n_raw": t.n_raw, "n_eval": n,
            "frame_acc": float((t.pred == ti).mean() * 100) if n else float("nan"),
            "vote_acc": float((v == ti).mean() * 100) if len(v) else float("nan"),
            "median_nearest_mm": float(np.median(t.nearest_mm)) if n else float("nan"),
            "most_common_pred": MODEL_CLASSES[Counter(t.pred.tolist()).most_common(1)[0][0]] if n else "",
        })
    n_eval = int(cm.sum())
    frame_acc = float(np.trace(cm) / n_eval * 100) if n_eval else float("nan")
    recall = {c: (float(cm[i, i] / cm[i].sum() * 100) if cm[i].sum() else float("nan"))
              for i, c in enumerate(MODEL_CLASSES)}
    precision = {c: (float(cm[i, i] / cm[:, i].sum() * 100) if cm[:, i].sum() else float("nan"))
                 for i, c in enumerate(MODEL_CLASSES)}
    pairs = [(MODEL_CLASSES[i], MODEL_CLASSES[j], int(cm[i, j]))
             for i in range(k) for j in range(k) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda x: -x[2])
    take_accs = [r["frame_acc"] for r in per_take if not np.isnan(r["frame_acc"])]
    return {
        "n_takes": len(takes), "n_raw_frames": int(sum(t.n_raw for t in takes)),
        "n_eval_frames": n_eval, "frame_acc": frame_acc,
        "vote_acc": float(vote_correct / vote_total * 100) if vote_total else float("nan"),
        "vote_windows": vote_total,
        "take_acc_mean": float(np.mean(take_accs)) if take_accs else float("nan"),
        "take_acc_sd": float(np.std(take_accs, ddof=1)) if len(take_accs) > 1 else float("nan"),
        "take_acc_min": float(min(take_accs)) if take_accs else float("nan"),
        "take_acc_max": float(max(take_accs)) if take_accs else float("nan"),
        "recall": recall, "precision": precision, "confusion_matrix": cm.tolist(),
        "top_confusions": [{"true": a, "pred": b, "frames": n} for a, b, n in pairs[:5]],
        "per_take": per_take,
    }


def preregistered_checks(m: dict) -> dict:
    """เทียบกับคำทำนายที่ลงทะเบียนไว้ล่วงหน้า (§5.1e) — รายงานตามจริง ไม่ปรับเกณฑ์ทีหลัง"""
    acc = m["frame_acc"]
    rec = m["recall"]
    high = float(np.nanmean([rec[c] for c in HIGH_GROUP]))
    low = float(np.nanmean([rec[c] for c in LOW_GROUP]))
    lowest4 = sorted(MODEL_CLASSES, key=lambda c: (rec[c], c))[:4]
    if acc < P1_BAND[0]:
        p1 = "BELOW the 60-85 % band"
    elif acc > P1_BAND[1]:
        p1 = "ABOVE the 60-85 % band"
    else:
        p1 = "INSIDE the 60-85 % band"
    if acc < STOP_BELOW:
        rule = "< 60 %: stop and diagnose (status, distance, orientation, sensor settings) before Phase 5.5"
    elif acc >= ST_LOWEST:
        rule = ">= 68.91 % (ST's lowest user): consistent with ST's per-user spread"
    else:
        rule = "60-68.9 %: check preprocessing and orientation first"
    return {
        "P1": {"frame_acc": acc, "verdict": p1},
        "P2": {"high_group_mean_recall": high, "low_group_mean_recall": low,
               "holds": bool(high > low)},
        "P3": {"vote_acc": m["vote_acc"], "frame_acc": acc, "holds": bool(m["vote_acc"] >= acc)},
        "H9": {"four_lowest": lowest4, "holds": set(lowest4) == set(LOW_GROUP)},
        "decision_rule": rule,
    }


# ---------------------------------------------------------------- reproducibility
def sha256_file(path: Path) -> str:
    """sha256 ของไฟล์ (อ่านทีละก้อน)"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_checksum(files: list[Path], root: Path) -> str:
    """sha256 รวมของข้อมูลทุกไฟล์ (เรียงตาม path) — ข้อมูลเปลี่ยนแม้ไฟล์เดียว ค่านี้เปลี่ยน"""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda p: str(p.relative_to(root))):
        h.update(str(f.relative_to(root)).replace("\\", "/").encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def git_commit() -> str:
    """commit ปัจจุบันของ repo (ถ้าหาไม่ได้คืน 'unknown')"""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def read_manifest(data_dir: Path) -> dict:
    """สรุป manifest_collect.csv (ถ้ามี) ไว้เทียบกับไฟล์ที่เจอจริง"""
    path = data_dir / "manifest_collect.csv"
    if not path.exists():
        return {"found": False}
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    def total(col: str) -> int:
        return int(sum(float(r.get(col) or 0) for r in rows))
    return {"found": True, "rows": len(rows), "missing_frames_total": total("missing_frames"),
            "out_of_window_total": total("out_of_window_frames"),
            "users": sorted({r.get("user", "") for r in rows}),
            "files": {Path(r["file"].replace("\\", "/")).name for r in rows if r.get("file")}}


# ---------------------------------------------------------------- แสดงผล
def bar(pct: float, width: int = 20) -> str:
    """แถบเปอร์เซ็นต์"""
    if np.isnan(pct):
        return "·" * width
    n = int(round(pct / 100 * width))
    return "█" * n + "░" * (width - n)


def fmt(x: float) -> str:
    return "  n/a" if np.isnan(x) else f"{x:5.1f}"


def render_report(meta: dict, results: dict[str, dict]) -> str:
    """ข้อความสรุปบนจอ (ภาษาอังกฤษ ให้แคปใส่รายงานได้สวย)"""
    L: list[str] = [LINE, "  Phase 4 Evaluation (H8)  ·  ST-trained model on our VL53L8CA board", LINE]
    L.append(f"  Model          {meta['model_path']}")
    L.append(f"                 sha256 {meta['model_sha256'][:16]}…  ·  {meta['n_model_classes']} classes")
    L.append(f"  Data           {meta['data_dir']}  ·  {meta['n_files']} takes  ·  "
             f"{', '.join(meta['users'])}")
    cc = meta["crosscheck"]
    if cc["max_abs_diff"] == 0:
        same = "identical"
    elif cc["max_abs_diff"] <= FLOAT_TOL:
        same = f"same within float32 rounding (max {cc['max_abs_diff']:.1e})"
    else:
        same = f"DIFFERENT (max {cc['max_abs_diff']:.2e}) — investigate before trusting results"
    L.append(f"  Preprocessing  st_dataset.py (ST rules, 100–400 mm)  ·  vs training code: {same}")
    L.append(f"                 from {meta['st_dataset_path']}")
    if cc["kept_only_by_training_code"] or cc["kept_only_by_st_dataset"]:
        L.append(f"                 frames kept by one rule only: training {cc['kept_only_by_training_code']}"
                 f" / st_dataset {cc['kept_only_by_st_dataset']}")
    mf = meta["manifest"]
    if mf.get("found"):
        L.append(f"  Manifest       {mf['rows']} rows  ·  missing frames {mf['missing_frames_total']}"
                 f"  ·  brief out-of-window {mf['out_of_window_total']}  ·  files match: "
                 f"{'yes' if mf['files_match'] else 'NO'}")
    L.append(f"  Baselines      chance {CHANCE * 100:.1f} %  ·  ST leave-one-user-out "
             f"{min(ST_LOSO.values()):.1f}–{max(ST_LOSO.values()):.1f} % (mean 79.7 %)")
    for user, m in results.items():
        c = m["checks"]
        L.append("")
        L.append(f"════ {user} " + "═" * max(4, 61 - len(user)))
        L.append(f"  Frames evaluated     {m['n_eval_frames']:,} of {m['n_raw_frames']:,} "
                 f"({m['n_takes']} takes; others removed by the 100–400 mm rule)")
        L.append(f"  Per-frame accuracy   {m['frame_acc']:5.1f} %  {bar(m['frame_acc'])}")
        L.append(f"  15-frame vote acc.   {m['vote_acc']:5.1f} %  {bar(m['vote_acc'])}")
        L.append(f"  Per-take accuracy    mean {m['take_acc_mean']:.1f} %  ·  SD {m['take_acc_sd']:.1f}  ·  "
                 f"range {m['take_acc_min']:.1f}–{m['take_acc_max']:.1f}  (n = {m['n_takes']})")
        L.append("  Per-class recall")
        for cls in MODEL_CLASSES:
            takes = [r for r in m["per_take"] if r["class"] == cls]
            tk = "  ".join(f"t{r['take']} {fmt(r['frame_acc'])}" for r in sorted(takes, key=lambda r: r["take"]))
            L.append(f"    {cls:<12}{fmt(m['recall'][cls])} %  {bar(m['recall'][cls], 16)}   {tk}")
        if m["top_confusions"]:
            L.append("  Top confusions (true → predicted, frames)")
            for t in m["top_confusions"]:
                L.append(f"    {t['true']:<11} → {t['pred']:<11} {t['frames']:>5}")
        L.append("  Pre-registered checks (§5.1e)")
        L.append(f"    P1  frame accuracy {c['P1']['frame_acc']:.1f} %  →  {c['P1']['verdict']}")
        L.append(f"    P2  recall Love/CrossHands/BreakTime {c['P2']['high_group_mean_recall']:.1f} %  vs  "
                 f"Fist/FlatHand/Dislike/Like {c['P2']['low_group_mean_recall']:.1f} %  →  "
                 f"{'HOLDS' if c['P2']['holds'] else 'DOES NOT HOLD'}")
        L.append(f"    P3  vote {c['P3']['vote_acc']:.1f} %  ≥  frame {c['P3']['frame_acc']:.1f} %  →  "
                 f"{'HOLDS' if c['P3']['holds'] else 'DOES NOT HOLD'}")
        L.append(f"    H9  four lowest recall: {', '.join(c['H9']['four_lowest'])}  →  "
                 f"{'HOLDS' if c['H9']['holds'] else 'DOES NOT HOLD'}")
        L.append(f"    Decision rule: {c['decision_rule']}")
    L.append("")
    L.append(f"  Pilot: n = {len(results)} subject(s) — descriptive only, no significance test (§5.1e-A1).")
    L.append(LINE)
    return "\n".join(L)


def save_confusion_png(path: Path, user: str, m: dict) -> bool:
    """รูป confusion matrix (% ต่อแถว + จำนวนเฟรม) — ข้ามถ้าไม่มี matplotlib"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    cm = np.array(m["confusion_matrix"], dtype=float)
    row = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0) * 100
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(MODEL_CLASSES)), MODEL_CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(MODEL_CLASSES)), MODEL_CLASSES)
    ax.set_xlabel("Predicted posture")
    ax.set_ylabel("True posture (intended)")
    ax.set_title(f"{user} — per-frame, accuracy {m['frame_acc']:.1f} % (n = {m['n_eval_frames']:,} frames)")
    for i in range(len(MODEL_CLASSES)):
        for j in range(len(MODEL_CLASSES)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{pct[i, j]:.0f}%\n({int(cm[i, j])})", ha="center", va="center",
                        fontsize=7, color="white" if pct[i, j] > 55 else "black")
    fig.colorbar(im, ax=ax, label="% of true-class frames")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main
def load_keras_predictor(model_path: Path) -> tuple[Callable[[np.ndarray], np.ndarray], int, str]:
    """โหลดโมเดล .keras คืน (ฟังก์ชันทาย, จำนวนคลาส, เวอร์ชัน TF)"""
    try:
        import tensorflow as tf
    except ImportError:
        raise SystemExit("tensorflow is required (same environment as live_predict.py)")
    model = tf.keras.models.load_model(model_path)
    n_out = int(model.output_shape[-1])

    def predict(X: np.ndarray) -> np.ndarray:
        return model.predict(X, batch_size=256, verbose=0)
    return predict, n_out, tf.__version__


def main(argv: list[str] | None = None,
         predictor_factory: Callable[[Path], tuple[Callable[[np.ndarray], np.ndarray], int, str]]
         = load_keras_predictor) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help=r"เช่น data\phase4")
    ap.add_argument("--model", type=Path, required=True,
                    help=r"เช่น results_phase2_final\production_model.keras (ตัวที่ deploy, §5.1d-F)")
    ap.add_argument("--st-dataset", type=Path, default=None, help="path ของ st_dataset.py (ปกติหาเอง)")
    ap.add_argument("--out-dir", type=Path, default=Path("results_phase4"))
    ap.add_argument("--users", nargs="*", default=None, help="เช่น User5 (ค่าเริ่มต้น = ทุกคนที่เจอ)")
    ap.add_argument("--allow-extra", action="store_true",
                    help="ยอมให้ประเมิน phase4_extra (ปกติห้าม: ชุดนั้นใช้เทรน, A7)")
    args = ap.parse_args(argv)

    if "extra" in args.data_dir.name.lower() and not args.allow_extra:
        raise SystemExit("data\\phase4_extra is training data (A7) — evaluate data\\phase4 instead")
    if not args.data_dir.is_dir():
        raise SystemExit(f"data folder not found: {args.data_dir}")
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")

    st_path = find_st_dataset(args.st_dataset)
    std = import_module_from(st_path, "st_dataset")
    files = sorted(p for p in args.data_dir.glob("*/*/npz/*.npz"))
    if not files:
        raise SystemExit(f"no .npz files under {args.data_dir}\\<Posture>\\log__*\\npz\\")

    print(f"Loading {len(files)} takes and checking labels/preprocessing ...")
    takes = [load_take(f, std) for f in files]
    if args.users:
        takes = [t for t in takes if t.user in set(args.users)]
        if not takes:
            raise SystemExit(f"no takes for users {args.users}")

    print(f"Loading model {args.model} ...")
    predict, n_out, tf_version = predictor_factory(args.model)
    if n_out != len(MODEL_CLASSES):
        raise SystemExit(f"model has {n_out} outputs, expected {len(MODEL_CLASSES)} — wrong model file?")
    for t in takes:
        if len(t.X):
            prob = np.asarray(predict(t.X))
            t.pred, t.conf = prob.argmax(axis=1), prob.max(axis=1)
        else:
            t.pred, t.conf = np.empty(0, dtype=int), np.empty(0)

    users = sorted({t.user for t in takes}, key=lambda u: int(u[4:]))
    results: dict[str, dict] = {}
    for u in users:
        m = subject_metrics([t for t in takes if t.user == u])
        m["checks"] = preregistered_checks(m)
        results[u] = m

    manifest = read_manifest(args.data_dir)
    if manifest.get("found"):
        manifest["files_match"] = manifest.pop("files") == {f.name for f in files}
    cc_all = {"max_abs_diff": max(t.crosscheck["max_abs_diff"] for t in takes),
              "kept_only_by_training_code": sum(t.crosscheck["kept_only_by_training_code"] for t in takes),
              "kept_only_by_st_dataset": sum(t.crosscheck["kept_only_by_st_dataset"] for t in takes)}
    meta = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model), "model_sha256": sha256_file(args.model), "n_model_classes": n_out,
        "data_dir": str(args.data_dir), "n_files": len(files),
        "data_sha256": data_checksum(files, args.data_dir), "users": users,
        "crosscheck": cc_all, "manifest": manifest, "git_commit": git_commit(),
        "python": sys.version.split()[0], "numpy": np.__version__, "tensorflow": tf_version,
        "platform": platform.platform(), "vote_window": VOTE_WINDOW, "class_order": MODEL_CLASSES,
        "st_dataset_filter_mm": [std.MIN_DISTANCE_MM, std.MAX_DISTANCE_MM],
        "st_dataset_path": str(st_path), "st_dataset_sha256": sha256_file(st_path),
    }

    report = render_report(meta, results)
    print("\n" + report)

    out = args.out_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    with (out / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": results}, fh, ensure_ascii=False, indent=2)
    with (out / "per_take.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["user", "class", "take", "file", "n_raw", "n_eval", "frame_acc", "vote_acc",
                    "median_nearest_mm", "most_common_pred"])
        for u, m in results.items():
            for r in m["per_take"]:
                w.writerow([u, r["class"], r["take"], r["file"], r["n_raw"], r["n_eval"],
                            f"{r['frame_acc']:.2f}", f"{r['vote_acc']:.2f}",
                            f"{r['median_nearest_mm']:.1f}", r["most_common_pred"]])
    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["user", "class", "take", "file", "raw_frame", "nearest_mm", "pred", "confidence", "correct"])
        for t in takes:
            for i in range(len(t.pred)):
                pred = MODEL_CLASSES[int(t.pred[i])]
                w.writerow([t.user, t.cls, t.take, t.path.name, int(t.raw_index[i]),
                            f"{t.nearest_mm[i]:.0f}", pred, f"{t.conf[i]:.4f}", int(pred == t.cls)])
    pngs = [u for u, m in results.items() if save_confusion_png(out / f"confusion_{u}.png", u, m)]
    print(f"\nSaved to {out.resolve()}")
    print("  report.txt, summary.json, per_take.csv, per_frame.csv"
          + (", " + ", ".join(f"confusion_{u}.png" for u in pngs) if pngs else
             "  (no PNG: matplotlib not installed)"))
    return {"meta": meta, "results": results, "out": out}


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ใช้โมเดลผิดไฟล์ → sha256 ใน summary.json จะต่างจากรอบอื่น ; ตัวที่ deploy คือ
#     results_phase2_final\production_model.keras (§5.1d-F) ไม่ใช่ pc_tools\results_phase2\...
#  2. st_dataset.py ที่หาเจอไม่ใช่ตัวที่ตรวจแล้ว (มีหลายสำเนา) → ใส่ --st-dataset ระบุเอง
#  3. 15-frame vote เริ่มใหม่ทุกรอบ (ไม่ลากข้ามรอบ) ต่างจาก live_predict.py ที่ไม่ล้าง buffer —
#     แบบนี้วัด "ท่าเดียวต่อเนื่อง" ได้ตรงกว่า และไม่มีผลจากท่าก่อนหน้าปน
#  4. เฟรมในรอบเดียวกันไม่อิสระต่อกัน — อย่าคิด CI จากจำนวนเฟรม ใช้ "คน" เป็นหน่วย
#  5. ผลนี้เป็น pilot (n ≤ 2) — ห้ามสรุปว่าโมเดล "ใช้ได้/ไม่ได้" กับบอร์ดเราจากตัวเลขนี้อย่างเดียว
# ─────────────────────────────────────────────────────────────────────────
