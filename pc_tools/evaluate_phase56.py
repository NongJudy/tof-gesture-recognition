"""evaluate_phase56.py — Phase 5.6: วัดความแม่นยำบนชุดทดสอบวันใหม่ data\\phase56 (PROJECT_FACTS §5.1l + §5.1o)

ชุดทดสอบนี้ห้ามใช้ฝึก / เลือกโมเดล / ปรับค่าใดๆ ตลอดไป (§5.1l) — สคริปต์นี้แค่อ่านและวัด

ทำอะไร (ทุกข้อจดไว้ล่วงหน้าแล้วก่อนมีข้อมูล):
  1. PC: ทายด้วย 7 โมเดลบนข้อมูลชุดเดียวกัน (เตรียมข้อมูลด้วย st_dataset._preprocess_session ตัวเดียวกับตอนฝึก)
       ฐาน 1 = ST อย่างเดียว (results_phase2_fixed\\production_model.keras)
       ฐาน 2 = โมเดลเดี่ยวสูตร B 5 ตัว (seed 42–46, ตรวจ sha256 กับ manifest.json)
       E5    = ตัวที่อยู่บนบอร์ด (results_phase55_final\\deploy\\E5.keras)
     รายงาน: ความแม่นยำรายเฟรม, หลังโหวต 15 เฟรม (กติกาเดียวกับเฟิร์มแวร์), recall รายท่า, ความสับสน, รายรอบ
  2. บอร์ด (§5.1o): ความแม่นยำจากผลทายของบอร์ดเอง (ไฟล์ *_board.csv ที่ collect_own_data.py v6 บันทึก)
  3. บอร์ด vs PC (E5) ทุกเฟรม: ท่าตรงกันไหม, ตัดสินในช่วง/นอกช่วงตรงกันไหม, ความน่าจะเป็นต่างกันเท่าไร
  4. เวลาต่อเฟรมบนบอร์ดทุกเฟรม (บรรทัด Q): เฉลี่ย, SD, มัธยฐาน, p99, สูงสุด, จำนวนเฟรมที่เกิน 33 ms
  5. ตำแหน่งมือ (centroid แบบ §5.1j) เทียบกับชุดหลัก data\\phase4 ของคนเดียวกัน และช่วงของ ST
  6. ตรวจคำทำนาย B1–B7 (§5.1l, §5.1o)

ข้อจำกัด: n = 1–2 คน (คนที่โมเดลเคยเห็น) → รายงานเชิงพรรณนา ไม่มีค่า p; เฟรมในรอบเดียวกันไม่อิสระต่อกัน

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\evaluate_phase56.py
ตัวเลือก: --data-dir data\\phase56 --ref-dir data\\phase4 --models-dir results_phase55_final
          --e5 results_phase55_final\\deploy\\E5.keras --baseline results_phase2_fixed\\production_model.keras
ผลลัพธ์: results_phase56\\eval_<วันเวลา>\\ report.txt, summary.json, per_frame.csv, confusion_E5_<user>.png
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

CLASSES = ["Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime"]  # ลำดับของ train_st_cnn2d.py
VOTE_WINDOW = 15            # เท่ากับ MY_AI_VOTE_WINDOW ในเฟิร์มแวร์
BUDGET_US = 33000           # เกณฑ์ §5.1l
# ช่วงแถวของ centroid มือของ ST 4 คน (§5.1e-E) ใช้บอกว่ามือเราอยู่นอกช่วงของ ST ไหม
ST_ROW_RANGE = {"Fist": (4.54, 5.59), "FlatHand": (3.89, 4.52), "Dislike": (3.82, 4.54), "Like": (4.64, 5.29),
                "Love": (4.29, 4.96), "CrossHands": (4.50, 4.70), "BreakTime": (3.47, 4.27)}
USER_RE = re.compile(r"__(User\d+)__")
TAKE_RE = re.compile(r"__take(\d+)")


def import_st_dataset():
    """โหลด st_dataset.py ที่อยู่ข้างสคริปต์นี้ (pc_tools) — ตัวเดียวกับที่ใช้ฝึก"""
    here = Path(__file__).resolve().parent
    for d in (here, here.parent):
        if (d / "st_dataset.py").is_file():
            sys.path.insert(0, str(d))
            import st_dataset  # noqa: E402
            return st_dataset
    raise SystemExit("st_dataset.py not found next to this script")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def in_range_mask(zone: np.ndarray, st) -> np.ndarray:
    """เฟรมไหน "มีมือในช่วง 100–400 มม." — สูตรเดียวกับ st_dataset._preprocess_session ข้อ 2–4"""
    valid = np.isin(zone[st.CH_TARGET_STATUS], st.VALID_STATUS)
    hand = np.where(valid, zone[st.CH_DISTANCE], st.DEFAULT_DISTANCE).min(axis=0)
    return (hand >= st.MIN_DISTANCE_MM) & (hand <= st.MAX_DISTANCE_MM)


def hand_masks(zone: np.ndarray, st) -> np.ndarray:
    """โซนที่นับเป็นมือ (หลังตัดฉากหลัง) ของเฟรมที่อยู่ในช่วง → (n, 8, 8) ใช้หา centroid แบบ §5.1j"""
    status, dist = zone[st.CH_TARGET_STATUS], zone[st.CH_DISTANCE]
    valid = np.isin(status, st.VALID_STATUS)
    dmin = np.where(valid, dist, st.DEFAULT_DISTANCE).min(axis=0)
    inr = (dmin >= st.MIN_DISTANCE_MM) & (dmin <= st.MAX_DISTANCE_MM)
    valid &= dist <= dmin[None, :] + st.BACKGROUND_DISTANCE_MM
    return valid[:, inr].T.reshape(-1, 8, 8)


def centroid(masks: list[np.ndarray]) -> tuple[float, float]:
    """(แถว, คอลัมน์) ของจุดศูนย์กลางมือเฉลี่ย — วิธีเดียวกับ r3work/centroids.py (§5.1j)"""
    m = np.concatenate(masks).astype(float).mean(axis=0)
    r, c = np.mgrid[0:8, 0:8]
    return float((m * r).sum() / m.sum()), float((m * c).sum() / m.sum())


def read_board_csv(path: Path, n: int) -> dict:
    """อ่าน <take>_board.csv (v6) → board_cls (n,) [-9 = ไม่มีข้อมูล], board_p (n,7), times (n,5) [nan = ไม่มี]"""
    cls = np.full(n, -9, dtype=int)
    probs = np.full((n, len(CLASSES)), np.nan)
    times = np.full((n, 5), np.nan)
    if not path.is_file():
        return {"cls": cls, "p": probs, "t": times, "present": False}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != n:
        raise SystemExit(f"{path.name}: {len(rows)} rows but the .npz has {n} frames")
    for i, r in enumerate(rows):
        if r["board_class"] != "":
            cls[i] = int(r["board_class"])
        if r["board_p0"] != "":
            probs[i] = [float(r[f"board_p{k}"]) for k in range(len(CLASSES))]
        if r["total_us"] != "":
            times[i] = [float(r[k]) for k in ("sensor_us", "preprocess_us", "inference_us", "decision_us", "total_us")]
    return {"cls": cls, "p": probs, "t": times, "present": True}


def load_takes(data_dir: Path, st, with_board: bool = True) -> list[dict]:
    """อ่านทุกรอบใน data_dir: ข้อมูลที่เตรียมแล้ว (เฉพาะเฟรมในช่วง) + หน้ากากเฟรมในช่วง + ผลของบอร์ด"""
    takes = []
    for f in sorted(data_dir.glob("*/*/npz/*.npz")):
        cls = f.parts[-4]
        if cls not in CLASSES:
            continue
        with np.load(f, allow_pickle=True) as d:
            zone = d["zone_data"].astype(np.float64)
        n = zone.shape[2]
        mask = in_range_mask(zone, st)
        X, _ = st._preprocess_session(zone, np.zeros((1, n)), st.MAX_DISTANCE_MM, st.MIN_DISTANCE_MM,
                                      st.BACKGROUND_DISTANCE_MM)
        if len(X) != int(mask.sum()):
            raise SystemExit(f"{f.name}: in-range mask ({mask.sum()}) != st_dataset output ({len(X)})")
        t = {"file": f, "user": USER_RE.search(f.name).group(1), "cls": cls, "label": CLASSES.index(cls),
             "take": int(TAKE_RE.search(f.name).group(1)), "n": n, "mask": mask, "X": X.astype(np.float32),
             "hand": hand_masks(zone, st)}
        if with_board:
            t["board"] = read_board_csv(f.with_name(f.stem + "_board.csv"), n)
        takes.append(t)
    if not takes:
        raise SystemExit(f"no .npz files under {data_dir}")
    return takes


def firmware_vote(seq: np.ndarray, window: int = VOTE_WINDOW) -> np.ndarray:
    """โหวตแบบเดียวกับ my_tof_send_prediction(): หน้าต่าง window เฟรมล่าสุด, -1 = "none" ได้คะแนนเริ่ม,
    ท่าจะชนะต้องได้คะแนน "มากกว่า" (เสมอกัน → none หรือท่าที่ดัชนีน้อยกว่า) ; คืนค่าเฉพาะ t >= window-1"""
    out = []
    for t in range(window - 1, len(seq)):
        w = seq[t - window + 1: t + 1]
        best, best_n = -1, int((w < 0).sum())
        for c in range(len(CLASSES)):
            k = int((w == c).sum())
            if k > best_n:
                best, best_n = c, k
        out.append(best)
    return np.array(out, dtype=int)


def score(takes: list[dict], preds: dict[int, np.ndarray], user: str) -> dict:
    """preds[i] = ท่าที่ทาย (ยาว n, -1 = ไม่ทาย) ของรอบ i → เมตริกของคนนี้"""
    y_all, p_all, vy, vp, per_take = [], [], [], [], []
    for i, t in enumerate(takes):
        if t["user"] != user:
            continue
        seq = preds[i]
        m = t["mask"]
        y_all.append(np.full(int(m.sum()), t["label"]))
        p_all.append(seq[m])
        v = firmware_vote(seq)
        vy.append(np.full(len(v), t["label"]))
        vp.append(v)
        per_take.append({"cls": t["cls"], "take": t["take"], "acc": float((seq[m] == t["label"]).mean() * 100)})
    y, p = np.concatenate(y_all), np.concatenate(p_all)
    cm = np.zeros((7, 7), dtype=int)
    for a, b in zip(y, p):
        if b >= 0:
            cm[a, b] += 1
    recall = {c: float(cm[k, k] / max(1, (y == k).sum()) * 100) for k, c in enumerate(CLASSES)}
    conf = sorted(((int(cm[a, b]), CLASSES[a], CLASSES[b]) for a in range(7) for b in range(7) if a != b and cm[a, b]),
                  reverse=True)[:5]
    return {"n_frames": int(len(y)), "acc": float((y == p).mean() * 100),
            "vote_acc": float((np.concatenate(vy) == np.concatenate(vp)).mean() * 100),
            "recall": recall, "cm": cm.tolist(), "top_confusions": conf, "per_take": per_take}


def predict_takes(takes: list[dict], predict: Callable[[np.ndarray], np.ndarray]) -> tuple[dict, dict]:
    """ทายทุกรอบในครั้งเดียว → (ท่ายาว n ต่อรอบ [-1 นอกช่วง], ความน่าจะเป็นของเฟรมในช่วง)"""
    X = np.concatenate([t["X"] for t in takes])
    P = np.asarray(predict(X)) if len(X) else np.zeros((0, 7))
    cls_out, prob_out, k = {}, {}, 0
    for i, t in enumerate(takes):
        m = int(t["mask"].sum())
        pp = P[k:k + m]
        k += m
        seq = np.full(t["n"], -1, dtype=int)
        seq[t["mask"]] = pp.argmax(axis=1)
        cls_out[i], prob_out[i] = seq, pp
    return cls_out, prob_out


def board_vs_pc(takes: list[dict], e5_cls: dict, e5_prob: dict) -> dict:
    """เทียบผลบอร์ดกับ E5 บน PC ทุกเฟรม (§5.1o B5)"""
    n_both = n_agree = gate_mis = missing = 0
    diffs = []
    first_mis = []
    for i, t in enumerate(takes):
        b = t["board"]
        if not b["present"]:
            missing += t["n"]
            continue
        pc_in = t["mask"]
        prob_rows = {idx: r for idx, r in zip(np.nonzero(pc_in)[0], e5_prob[i])}
        for j in range(t["n"]):
            bc = b["cls"][j]
            if bc == -9:
                missing += 1
                continue
            if (bc >= 0) != bool(pc_in[j]):
                gate_mis += 1
                if len(first_mis) < 10:
                    first_mis.append(f"{t['file'].name}#{j} gate")
                continue
            if bc < 0:
                continue
            n_both += 1
            if bc == e5_cls[i][j]:
                n_agree += 1
            elif len(first_mis) < 10:
                first_mis.append(f"{t['file'].name}#{j} class board {bc} pc {e5_cls[i][j]}")
            if not np.isnan(b["p"][j]).any():
                diffs.append(float(np.abs(b["p"][j] - prob_rows[j]).max()))
    return {"compared": n_both, "agree": n_agree, "gate_mismatch": gate_mis, "missing_rows": missing,
            "max_abs_diff": max(diffs) if diffs else float("nan"),
            "median_abs_diff": float(np.median(diffs)) if diffs else float("nan"), "first_mismatches": first_mis}


def timing_stats(takes: list[dict]) -> dict:
    """เวลาต่อเฟรมบนบอร์ดจากบรรทัด Q (§5.1o B7)"""
    T = np.concatenate([t["board"]["t"] for t in takes if t["board"]["present"]]) if takes else np.zeros((0, 5))
    T = T[~np.isnan(T).any(axis=1)]
    out = {"n": int(len(T))}
    for k, name in enumerate(["sensor_us", "preprocess_us", "inference_us", "decision_us", "total_us"]):
        if len(T):
            col = T[:, k]
            out[name] = {"mean": float(col.mean()), "sd": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
                         "median": float(np.median(col)), "p99": float(np.percentile(col, 99)),
                         "min": float(col.min()), "max": float(col.max())}
    out["over_budget"] = int((T[:, 4] > BUDGET_US).sum()) if len(T) else 0
    return out


def save_confusion_png(cm: list[list[int]], title: str, path: Path) -> None:
    """รูป confusion matrix (นับเฟรม + % ต่อแถว) สำหรับรายงาน"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.array(cm, dtype=float)
    pct = a / np.maximum(1, a.sum(axis=1, keepdims=True)) * 100
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(7):
        for j in range(7):
            if a[i, j]:
                ax.text(j, i, f"{int(a[i, j])}\n{pct[i, j]:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if pct[i, j] > 60 else "black")
    ax.set_xticks(range(7), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Performed (label)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None, load_model: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/phase56"))
    ap.add_argument("--ref-dir", type=Path, default=Path("data/phase4"), help="ชุดหลัก Phase 4 ไว้เทียบตำแหน่งมือ")
    ap.add_argument("--models-dir", type=Path, default=Path("results_phase55_final"))
    ap.add_argument("--e5", type=Path, default=Path("results_phase55_final/deploy/E5.keras"))
    ap.add_argument("--baseline", type=Path, default=Path("results_phase2_fixed/production_model.keras"))
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    st = import_st_dataset()
    # ---- โมเดล: ตรวจไฟล์และ sha256 ก่อนใช้ ----
    man = json.loads((args.models_dir / "manifest.json").read_text(encoding="utf-8"))
    if man.get("class_order") not in (None, CLASSES):
        raise SystemExit(f"class order in manifest differs: {man.get('class_order')}")
    singles = []
    for m in sorted(man["models"], key=lambda m: int(m["seed"])):
        f = args.models_dir / m["model_file"]
        if sha256_file(f) != m["model_sha256"]:
            raise SystemExit(f"sha256 mismatch: {f}")
        singles.append((f"seed{m['seed']}", f))
    for f in (args.e5, args.baseline):
        if not f.is_file():
            raise SystemExit(f"model not found: {f}")
    if load_model is None:
        import tensorflow as tf
        def load_model(path: Path):  # noqa: E306
            mdl = tf.keras.models.load_model(str(path), compile=False)
            if mdl.output_shape[-1] != len(CLASSES):
                raise SystemExit(f"{path}: {mdl.output_shape[-1]} outputs, expected 7")
            return lambda X: mdl.predict(X, batch_size=512, verbose=0)
    models = [("baseline_ST_only", args.baseline)] + singles + [("E5", args.e5)]
    model_sha = {name: sha256_file(p)[:16] for name, p in models}

    print(f"Loading {args.data_dir} ...")
    takes = load_takes(args.data_dir, st)
    users = sorted({t["user"] for t in takes})
    print(f"  {len(takes)} takes, users {users}, frames {sum(t['n'] for t in takes)}")

    res: dict = {"models": {}, "users": users}
    all_cls: dict[str, dict] = {}
    e5_cls = e5_prob = None
    for name, path in models:
        print(f"Predicting with {name} ...")
        cls, prob = predict_takes(takes, load_model(path))
        all_cls[name] = cls
        res["models"][name] = {u: score(takes, cls, u) for u in users}
        if name == "E5":
            e5_cls, e5_prob = cls, prob

    board_present = all(t["board"]["present"] for t in takes)
    board_cls = {i: np.where(t["board"]["cls"] == -9, -1, t["board"]["cls"]) for i, t in enumerate(takes)}
    res["board"] = {u: score(takes, board_cls, u) for u in users} if board_present else None
    res["board_vs_pc"] = board_vs_pc(takes, e5_cls, e5_prob)
    res["timing"] = timing_stats(takes)

    # ---- ตำแหน่งมือ: phase56 เทียบชุดหลัก phase4 ของคนเดียวกัน ----
    ref = []
    if args.ref_dir.is_dir():
        ref = [t for t in load_takes(args.ref_dir, st, with_board=False) if t["user"] in users]
    pos = {}
    for u in users:
        for c in CLASSES:
            new = [t["hand"] for t in takes if t["user"] == u and t["cls"] == c and len(t["hand"])]
            old = [t["hand"] for t in ref if t["user"] == u and t["cls"] == c and len(t["hand"])]
            pos[f"{u}|{c}"] = {"phase56": centroid(new) if new else None, "phase4": centroid(old) if old else None}
    res["position"] = pos

    # ---- คำทำนาย B1–B7 ----
    single_names = [n for n, _ in singles]
    checks = {}
    if "User5" in users:
        like = [res["models"][n]["User5"]["recall"]["Like"] for n in single_names]
        checks["B1 User5 Like recall, mean of 5 singles > 50 %"] = (float(np.mean(like)), float(np.mean(like)) > 50)
        e5u5 = res["models"]["E5"]["User5"]["acc"]
        b1u5 = res["models"]["baseline_ST_only"]["User5"]["acc"]
        checks["B4 User5 E5 - ST-only baseline >= 10 pp"] = (e5u5 - b1u5, e5u5 - b1u5 >= 10)
    for u in users:
        e5 = res["models"]["E5"][u]["acc"]
        mean5 = float(np.mean([res["models"][n][u]["acc"] for n in single_names]))
        checks[f"B2 {u} E5 per-frame >= 85 %"] = (e5, e5 >= 85)
        checks[f"B3 {u} E5 >= mean of 5 singles ({mean5:.1f})"] = (e5 - mean5, e5 >= mean5)
    bp = res["board_vs_pc"]
    checks["B5 board vs PC class agreement = 100 %, gate mismatch 0"] = (
        f"{bp['agree']}/{bp['compared']}, gate {bp['gate_mismatch']}",
        bp["compared"] > 0 and bp["agree"] == bp["compared"] and bp["gate_mismatch"] == 0)
    if res["board"]:
        same = all(abs(res["board"][u]["acc"] - res["models"]["E5"][u]["acc"]) < 1e-9 for u in users)
        checks["B6 on-board accuracy == PC E5 accuracy"] = ("equal" if same else "differs", same)
    tm = res["timing"]
    if tm["n"]:
        tot = tm["total_us"]
        ok = 29200 <= tot["mean"] <= 29400 and tot["max"] <= 30000 and tm["over_budget"] == 0
        checks["B7 total mean 29.3±0.1 ms, max <= 30.0 ms, 0 frames > 33 ms"] = (
            f"mean {tot['mean']:.0f} us, max {tot['max']:.0f} us, >33ms {tm['over_budget']}", ok)
    res["checks"] = {k: {"value": v, "holds": bool(h)} for k, (v, h) in checks.items()}

    # ---- เขียนผล ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out_dir or Path("results_phase56") / f"eval_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    write_per_frame(out / "per_frame.csv", takes, [n for n, _ in models], all_cls, board_cls)
    for u in users:
        save_confusion_png(res["models"]["E5"][u]["cm"],
                           f"E5 on data\\phase56, {u} (n = {res['models']['E5'][u]['n_frames']} frames)",
                           out / f"confusion_E5_{u}.png")
    res["meta"] = {"run_at": stamp, "data_dir": str(args.data_dir), "model_sha256_prefix": model_sha,
                   "n_takes": len(takes), "n_frames": int(sum(t["n"] for t in takes))}
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    text = report_text(res, single_names, out)
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return res


def write_per_frame(path: Path, takes: list[dict], names: list[str], all_cls: dict, board_cls: dict) -> None:
    """per_frame.csv: ท่าที่แต่ละโมเดลทายทุกเฟรม (-1 = นอกช่วง) + ของบอร์ด ใช้ทำ error analysis ต่อ"""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["user", "class", "take", "frame_idx", "pc_in_range"] + names + ["board"])
        for i, t in enumerate(takes):
            for j in range(t["n"]):
                w.writerow([t["user"], t["cls"], t["take"], j, int(t["mask"][j])]
                           + [int(all_cls[n][i][j]) for n in names] + [int(board_cls[i][j])])


def report_text(res: dict, single_names: list[str], out: Path) -> str:
    """รายงานอ่านง่าย (ภาษาอังกฤษ เพราะฟอนต์ terminal ไม่มีอักษรไทย)"""
    L = ["=" * 78, "  Phase 5.6 — new-day test set data\\phase56 (§5.1l + §5.1o)  ·  known users, descriptive", "=" * 78,
         f"  takes {res['meta']['n_takes']}, frames {res['meta']['n_frames']}, users {', '.join(res['users'])}", ""]
    for u in res["users"]:
        L.append(f"  ── {u} ──  per-frame accuracy % / 15-frame vote accuracy %")
        rows = [("ST-only baseline", res["models"]["baseline_ST_only"][u])]
        accs = [res["models"][n][u]["acc"] for n in single_names]
        votes = [res["models"][n][u]["vote_acc"] for n in single_names]
        L.append(f"    {'ST-only baseline':<22} {rows[0][1]['acc']:6.1f} / {rows[0][1]['vote_acc']:6.1f}")
        L.append(f"    {'5 singles mean ± SD':<22} {np.mean(accs):6.1f} ± {np.std(accs, ddof=1):.1f} / "
                 f"{np.mean(votes):6.1f} ± {np.std(votes, ddof=1):.1f}   (range {min(accs):.1f}–{max(accs):.1f})")
        e5 = res["models"]["E5"][u]
        L.append(f"    {'E5 (PC)':<22} {e5['acc']:6.1f} / {e5['vote_acc']:6.1f}   n = {e5['n_frames']} frames")
        if res["board"]:
            b = res["board"][u]
            L.append(f"    {'E5 (ON THE BOARD)':<22} {b['acc']:6.1f} / {b['vote_acc']:6.1f}")
        L.append("    recall %       " + "  ".join(f"{c[:6]:>6}" for c in CLASSES))
        for name, lab in (("baseline_ST_only", "ST-only"), ("E5", "E5")):
            r = res["models"][name][u]["recall"]
            L.append(f"    {lab:<14} " + "  ".join(f"{r[c]:6.1f}" for c in CLASSES))
        mr = {c: np.mean([res["models"][n][u]["recall"][c] for n in single_names]) for c in CLASSES}
        L.append(f"    {'singles mean':<14} " + "  ".join(f"{mr[c]:6.1f}" for c in CLASSES))
        L.append("    E5 top confusions: " + (", ".join(f"{a}→{b} {n}" for n, a, b in e5["top_confusions"]) or "none"))
        L.append("    E5 per take: " + ", ".join(f"{p['cls'][:5]}{p['take']} {p['acc']:.0f}" for p in e5["per_take"]))
        L.append("")
    bp = res["board_vs_pc"]
    L += ["  ── Board vs PC (E5), every collected frame ──",
          f"    compared {bp['compared']}, class agree {bp['agree']}/{bp['compared']}, gate mismatch {bp['gate_mismatch']}, "
          f"rows without board data {bp['missing_rows']}",
          f"    prob |diff| max {bp['max_abs_diff']:.2e}, median {bp['median_abs_diff']:.2e}"]
    if bp["first_mismatches"]:
        L.append("    first mismatches: " + "; ".join(bp["first_mismatches"]))
    tm = res["timing"]
    L += ["", f"  ── On-board per-frame work, every predicted frame (Q lines), n = {tm['n']} ──"]
    for k in ("sensor_us", "preprocess_us", "inference_us", "decision_us", "total_us"):
        if k in tm:
            s = tm[k]
            L.append(f"    {k:<14} mean {s['mean']:8.1f}  sd {s['sd']:6.1f}  median {s['median']:8.0f}  "
                     f"p99 {s['p99']:8.0f}  min {s['min']:8.0f}  max {s['max']:8.0f}")
    L.append(f"    frames over {BUDGET_US} us: {tm['over_budget']}")
    L += ["", "  ── Hand centroid row | col (phase56 vs phase4 main set; ST row range) ──"]
    for key, v in res["position"].items():
        u, c = key.split("|")
        n56 = "  n/a" if v["phase56"] is None else f"{v['phase56'][0]:.2f} | {v['phase56'][1]:.2f}"
        n4 = "  n/a" if v["phase4"] is None else f"{v['phase4'][0]:.2f} | {v['phase4'][1]:.2f}"
        lo, hi = ST_ROW_RANGE[c]
        L.append(f"    {u} {c:<11} phase56 {n56}   phase4 {n4}   ST rows {lo:.2f}–{hi:.2f}")
    L += ["", "  ── Pre-registered predictions ──"]
    for k, v in res["checks"].items():
        val = v["value"]
        val = f"{val:.1f}" if isinstance(val, float) else str(val)
        L.append(f"    {'HOLDS ' if v['holds'] else 'FAILS '} {k}   [{val}]")
    L += ["", "  Not accuracy for new people (that stays LOSO 85.3 %, §5.1i-R). n = subjects listed above.",
          f"  output: {out}", "=" * 78]
    return "\n".join(L)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ไฟล์ *_board.csv ไม่มี (เก็บด้วย v5) → ส่วนบอร์ดจะว่าง ส่วน PC ยังทำงาน
#  2. จำนวนแถวใน *_board.csv ไม่เท่าจำนวนเฟรมใน .npz → หยุดทันที (ไฟล์ไม่ตรงกัน ห้ามเดา)
#  3. sha256 ของโมเดลเดี่ยวไม่ตรง manifest.json → หยุด (โมเดลถูกแก้/สลับ)
#  4. ลำดับคลาสต้องเป็น Fist, FlatHand, Dislike, Like, Love, CrossHands, BreakTime ทั้งในโมเดลและเฟิร์มแวร์ (§9 #38)
#  5. ชุดนี้ห้ามใช้ปรับอะไรทั้งสิ้น — ถ้าเห็นผลแล้วอยากแก้โมเดล ต้องเขียนแผนใหม่และวัดด้วย LOSO ก่อน (§5.1l)
# ─────────────────────────────────────────────────────────────────────────
