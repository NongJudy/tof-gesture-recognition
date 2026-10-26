"""a6_sim.py — A6-sim: เลื่อนมือในคอมพิวเตอร์ 1–2 โซน แล้วดูว่าโมเดลยังทายถูกไหม (PROJECT_FACTS §5.1t, amendment A13)

กติกาทั้งหมดจดไว้ใน §5.1t ก่อนรัน — สคริปต์นี้แค่อ่าน เลื่อน ทาย และนับ ไม่ปรับอะไร
  1. อ่านเฟรมดิบ 8×8 (distance, status, signal) ของชุดทดสอบวันใหม่ (ไม่ใช่ data\\phase4 เพราะ E5 ฝึกด้วยชุดนั้น)
  2. เลื่อนทั้งภาพทีละโซน: none, row±1, col±1 (หลัก) และ row±2, col±2 (รอง)
       ช่องที่ว่างจากการเลื่อน = status ใช้ไม่ได้ (เหมือนไม่มีวัตถุ) ; ช่องที่เลื่อนหลุดตาราง = หายไป
  3. เตรียมข้อมูลด้วย st_dataset._preprocess_session ตัวเดียวกับตอนฝึก แล้วให้โมเดลทายบน PC
     (บอร์ด = PC ยืนยันแล้ว > 9,000 เฟรม §5.1n–u)
  4. รายงานต่อ ชุดข้อมูล × โมเดล × การเลื่อน: ความแม่นยำรายเฟรม, หลังโหวต 15 เฟรม (กติกาเฟิร์มแวร์), recall รายท่า,
     เฟรมที่ "ถูก→ผิด / ผิด→ถูก" เทียบกับไม่เลื่อน (จับคู่เฟรมเดียวกัน), สัดส่วนช่องมือที่หลุดตาราง
  5. ตรวจคำทำนาย Q1–Q6 และกติกาตัดสิน 5.5c (§5.1s/§5.1t) อัตโนมัติ
  ข้อจำกัด: การเลื่อนในคอมไม่จำลองมุมมองที่เปลี่ยนและแสงที่อ่อนลงที่ขอบภาพ ; n = 1 คน ; เฟรมในรอบเดียวกันไม่อิสระ → เชิงพรรณนา

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\a6_sim.py
ตัวเลือก: --data-dirs data\\phase56 data\\phase56_stand data\\a5_love  --models-dir results_phase55_final
          --e5 results_phase55_final\\deploy\\E5.keras --baseline results_phase2_fixed\\production_model.keras
          --extra-model <path.keras>  (โมเดลใหม่ในอนาคต เช่น 5.5c — รายงานเท่านั้น ห้ามใช้เลือกโมเดล)
ผลลัพธ์: results_a6sim\\eval_<วันเวลา>\\ report.txt, summary.json, table.csv, a6sim_E5.png
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from evaluate_phase56 import (CLASSES, firmware_vote, import_st_dataset, in_range_mask, load_takes, sha256_file)

SHIFTS = {"none": (0, 0), "row+1": (1, 0), "row-1": (-1, 0), "col+1": (0, 1), "col-1": (0, -1),
          "row+2": (2, 0), "row-2": (-2, 0), "col+2": (0, 2), "col-2": (0, -2)}
PRIMARY = ["row+1", "row-1", "col+1", "col-1"]
SECONDARY = ["row+2", "row-2", "col+2", "col-2"]
MAIN_SETS = ("phase56", "phase56_stand")          # ชุดที่ใช้กับ Q1–Q3, Q5, Q6 และกติกาตัดสิน
DECISION_DROP_PP = 10.0


def shift_zone(zone: np.ndarray, dr: int, dc: int, st) -> np.ndarray:
    """เลื่อนเฟรมดิบ (ช่อง, 64, n) ไป dr แถว dc คอลัมน์ — ช่องว่าง = status 0 (ใช้ไม่ได้), ระยะ = DEFAULT"""
    n_ch, _, n = zone.shape
    z = zone.reshape(n_ch, 8, 8, n)
    out = np.zeros_like(z)
    out[st.CH_DISTANCE] = st.DEFAULT_DISTANCE       # status 0 อยู่แล้ว → preprocessing มองว่าว่าง
    rs = slice(max(dr, 0), 8 + min(dr, 0))          # แถวปลายทาง
    rd = slice(max(-dr, 0), 8 + min(-dr, 0))        # แถวต้นทาง
    cs = slice(max(dc, 0), 8 + min(dc, 0))
    cd = slice(max(-dc, 0), 8 + min(-dc, 0))
    out[:, rs, cs, :] = z[:, rd, cd, :]
    return out.reshape(n_ch, 64, n)


def lost_fraction(hand: np.ndarray, dr: int, dc: int) -> float:
    """สัดส่วนช่องมือ (เฟรมในช่วง) ที่เลื่อนแล้วหลุดตาราง"""
    if len(hand) == 0:
        return 0.0
    total = hand.sum()
    keep = hand[:, max(-dr, 0):8 + min(-dr, 0), max(-dc, 0):8 + min(-dc, 0)].sum()
    return float(1 - keep / total) if total else 0.0


def prepare(takes: list[dict], st) -> dict[str, list[dict]]:
    """สร้างข้อมูลที่เลื่อนแล้วของทุกรอบ: shift → [{X, mask}] ตามลำดับรอบ"""
    out = {}
    for name, (dr, dc) in SHIFTS.items():
        rows = []
        for t in takes:
            z = shift_zone(t["zone"], dr, dc, st)
            mask = in_range_mask(z, st)
            X, _ = st._preprocess_session(z, np.zeros((1, t["n"])), st.MAX_DISTANCE_MM, st.MIN_DISTANCE_MM,
                                          st.BACKGROUND_DISTANCE_MM)
            if len(X) != int(mask.sum()):
                raise SystemExit(f"{t['file'].name} {name}: mask {mask.sum()} != st_dataset {len(X)}")
            rows.append({"X": X.astype(np.float32), "mask": mask, "lost": lost_fraction(t["hand"], dr, dc)})
        out[name] = rows
    return out


def predict_all(prep: dict[str, list[dict]], takes: list[dict], predict: Callable) -> dict[str, list[np.ndarray]]:
    """ทายทุกการเลื่อนในครั้งเดียว → shift → [ลำดับท่ายาว n ต่อรอบ, -1 = นอกช่วง]"""
    X = np.concatenate([r["X"] for name in SHIFTS for r in prep[name]])
    P = np.asarray(predict(X)) if len(X) else np.zeros((0, len(CLASSES)))
    pred = P.argmax(axis=1)
    out, k = {}, 0
    for name in SHIFTS:
        seqs = []
        for t, r in zip(takes, prep[name]):
            m = int(r["mask"].sum())
            seq = np.full(t["n"], -1, dtype=int)
            seq[r["mask"]] = pred[k:k + m]
            k += m
            seqs.append(seq)
        out[name] = seqs
    return out


def metrics(takes: list[dict], seqs: list[np.ndarray], base: list[np.ndarray]) -> dict:
    """ความแม่นยำ + recall + จำนวนเฟรมถูก→ผิด/ผิด→ถูกเทียบกับไม่เลื่อน"""
    y, p, vy, vp = [], [], [], []
    c2w = w2c = paired = dropped = 0
    rec = {c: [0, 0] for c in CLASSES}
    for t, s, b in zip(takes, seqs, base):
        m = s >= 0
        y.append(np.full(int(m.sum()), t["label"]))
        p.append(s[m])
        rec[t["cls"]][0] += int((s[m] == t["label"]).sum())
        rec[t["cls"]][1] += int(m.sum())
        v = firmware_vote(s)
        vy.append(np.full(len(v), t["label"]))
        vp.append(v)
        both = (s >= 0) & (b >= 0)
        paired += int(both.sum())
        dropped += int(((b >= 0) & (s < 0)).sum())
        c2w += int(((b == t["label"]) & (s != t["label"]) & both).sum())
        w2c += int(((b != t["label"]) & (s == t["label"]) & both).sum())
    y, p = np.concatenate(y), np.concatenate(p)
    return {"n": int(len(y)), "acc": float((y == p).mean() * 100) if len(y) else float("nan"),
            "vote_acc": float((np.concatenate(vy) == np.concatenate(vp)).mean() * 100) if len(vy) else float("nan"),
            "recall": {c: (v[0] / v[1] * 100 if v[1] else float("nan")) for c, v in rec.items()},
            "correct_to_wrong": c2w, "wrong_to_correct": w2c, "paired": paired, "dropped_out_of_range": dropped}


def check_predictions(res: dict) -> list[tuple[str, bool | None, str]]:
    """Q1–Q6 (§5.1t) — คืน (ชื่อ, ผ่าน/ไม่ผ่าน/None=ประเมินไม่ได้, ตัวเลข)"""
    out = []
    sets = [d for d in res["datasets"] if res["datasets"][d]["name"] in MAIN_SETS]

    def acc(d, model, sh):
        return res["datasets"][d]["models"][model][sh]["acc"]

    for d in sets:
        nm = res["datasets"][d]["name"]
        prim = [acc(d, "E5", s) for s in PRIMARY]
        drop = acc(d, "E5", "none") - float(np.mean(prim))
        out.append((f"Q1 {nm}: E5 mean of ±1 shifts >= 5 pp below unshifted", drop >= 5, f"drop {drop:.1f} pp"))
        worst = PRIMARY[int(np.argmin(prim))]
        out.append((f"Q2 {nm}: row-1 is the worst ±1 shift", worst == "row-1",
                    "worst " + worst + " (" + ", ".join(f"{s} {a:.1f}" for s, a in zip(PRIMARY, prim)) + ")"))
        none_rec = res["datasets"][d]["models"]["E5"]["none"]["recall"]
        drops = {c: none_rec[c] - float(np.mean([res["datasets"][d]["models"]["E5"][s]["recall"][c] for s in PRIMARY]))
                 for c in CLASSES if not np.isnan(none_rec[c])}
        top = max(drops, key=drops.get)
        out.append((f"Q3 {nm}: largest mean ±1 recall drop is Like or Dislike", top in ("Like", "Dislike"),
                    f"largest {top} {drops[top]:.1f} pp"))
    love = [d for d in res["datasets"] if res["datasets"][d]["name"] == "a5_love"]
    for d in love:
        r = [res["datasets"][d]["models"]["E5"][s]["recall"]["Love"] for s in ("row+1", "row-1")]
        out.append(("Q4 a5_love: E5 Love >= 90 % at row+1 and row-1", min(r) >= 90, f"row+1 {r[0]:.1f}, row-1 {r[1]:.1f}"))
    for d in sets:
        nm = res["datasets"][d]["name"]
        bad = [m for m in res["datasets"][d]["models"] if not m.startswith("extra:")
               and np.mean([acc(d, m, s) for s in SECONDARY]) >= np.mean([acc(d, m, s) for s in PRIMARY])]
        out.append((f"Q5 {nm}: ±2 worse than ±1 for every model", not bad, "exceptions: " + (", ".join(bad) or "none")))
        singles = [m for m in res["datasets"][d]["models"] if m.startswith("seed")]
        e5d = acc(d, "E5", "none") - np.mean([acc(d, "E5", s) for s in PRIMARY])
        sd = float(np.mean([acc(d, m, "none") - np.mean([acc(d, m, s) for s in PRIMARY]) for m in singles])) \
            if singles else float("nan")
        out.append((f"Q6 {nm}: E5 drop <= singles' mean drop", None if np.isnan(sd) else bool(e5d <= sd),
                    f"E5 {e5d:.1f} pp, singles {sd:.1f} pp"))
    return out


def decision(res: dict) -> tuple[bool, list[str]]:
    """กติกาตัดสิน 5.5c: การเลื่อน ±1 ใดทำให้ E5 ตก >= 10 pp ในชุดหลักชุดใด → ทำ 5.5c"""
    hits = []
    for d, ds in res["datasets"].items():
        if ds["name"] not in MAIN_SETS:
            continue
        base = ds["models"]["E5"]["none"]["acc"]
        for s in PRIMARY:
            dr = base - ds["models"]["E5"][s]["acc"]
            if dr >= DECISION_DROP_PP:
                hits.append(f"{ds['name']} {s} −{dr:.1f} pp")
    return bool(hits), hits


def save_figure(res: dict, path: Path) -> None:
    """ภาพ heatmap ท่า × การเลื่อน (E5, recall %) ต่อชุดข้อมูลหลัก"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = [d for d in res["datasets"] if res["datasets"][d]["name"] in MAIN_SETS]
    if not sets:
        return
    cols = ["none"] + PRIMARY + SECONDARY
    fig, axes = plt.subplots(1, len(sets), figsize=(6.4 * len(sets), 4.6), dpi=150, squeeze=False)
    for ax, d in zip(axes[0], sets):
        e5 = res["datasets"][d]["models"]["E5"]
        a = np.array([[e5[s]["recall"][c] for s in cols] for c in CLASSES])
        ax.imshow(a, cmap="RdYlGn", vmin=0, vmax=100)
        for i in range(len(CLASSES)):
            for j in range(len(cols)):
                ax.text(j, i, f"{a[i, j]:.0f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
        ax.set_title(f"E5 recall % under simulated shift — {res['datasets'][d]['name']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None, load_model: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dirs", type=Path, nargs="+",
                    default=[Path("data/phase56"), Path("data/phase56_stand"), Path("data/a5_love")])
    ap.add_argument("--models-dir", type=Path, default=Path("results_phase55_final"))
    ap.add_argument("--e5", type=Path, default=Path("results_phase55_final/deploy/E5.keras"))
    ap.add_argument("--baseline", type=Path, default=Path("results_phase2_fixed/production_model.keras"))
    ap.add_argument("--extra-model", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    st = import_st_dataset()
    man = json.loads((args.models_dir / "manifest.json").read_text(encoding="utf-8"))
    if man.get("class_order") not in (None, CLASSES):
        raise SystemExit(f"class order in manifest differs: {man.get('class_order')}")
    models = [("baseline_ST_only", args.baseline)]
    for m in sorted(man["models"], key=lambda m: int(m["seed"])):
        f = args.models_dir / m["model_file"]
        if sha256_file(f) != m["model_sha256"]:
            raise SystemExit(f"sha256 mismatch: {f}")
        models.append((f"seed{m['seed']}", f))
    models.append(("E5", args.e5))
    models += [(f"extra:{p.name}", p) for p in args.extra_model]
    for _, p in models:
        if not p.is_file():
            raise SystemExit(f"model not found: {p}")
    if load_model is None:
        import tensorflow as tf

        def load_model(path: Path):  # noqa: E306
            mdl = tf.keras.models.load_model(str(path), compile=False)
            return lambda X: mdl.predict(X, batch_size=1024, verbose=0)

    data = {}
    for dd in args.data_dirs:
        print(f"Loading {dd} ...")
        takes = load_takes(dd, st, with_board=False)
        for t in takes:
            with np.load(t["file"], allow_pickle=True) as z:
                t["zone"] = z["zone_data"].astype(np.float64)
        users = sorted({t["user"] for t in takes})
        if len(users) != 1:
            raise SystemExit(f"{dd}: one subject per data dir please, found {users}")
        data[str(dd)] = {"name": dd.name, "takes": takes, "prep": prepare(takes, st), "user": users[0]}
        print(f"  {len(takes)} takes, {sum(t['n'] for t in takes)} frames, {users[0]}")

    res = {"generated": datetime.now().isoformat(timespec="seconds"), "shifts": SHIFTS,
           "model_sha256": {n: sha256_file(p)[:16] for n, p in models}, "datasets": {}}
    for k, d in data.items():
        res["datasets"][k] = {"name": d["name"], "user": d["user"], "n_takes": len(d["takes"]),
                              "lost_fraction": {s: float(np.mean([r["lost"] for r in d["prep"][s]])) for s in SHIFTS},
                              "models": {}}
    for name, path in models:
        print(f"Predicting with {name} ...")
        predict = load_model(path)
        for k, d in data.items():
            seqs = predict_all(d["prep"], d["takes"], predict)
            res["datasets"][k]["models"][name] = {s: metrics(d["takes"], seqs[s], seqs["none"]) for s in SHIFTS}

    res["predictions"] = [{"name": n, "holds": h, "value": v} for n, h, v in check_predictions(res)]
    trig, hits = decision(res)
    res["decision"] = {"trigger_5_5c": trig, "hits": hits}

    out = args.out_dir or Path("results_a6sim") / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "shift", "acc", "vote_acc", "correct_to_wrong", "wrong_to_correct",
                    "dropped_out_of_range"] + [f"recall_{c}" for c in CLASSES])
        for ds in res["datasets"].values():
            for m, by in ds["models"].items():
                for s, r in by.items():
                    w.writerow([ds["name"], m, s, f"{r['acc']:.2f}", f"{r['vote_acc']:.2f}", r["correct_to_wrong"],
                                r["wrong_to_correct"], r["dropped_out_of_range"]]
                               + [f"{r['recall'][c]:.2f}" for c in CLASSES])
    save_figure(res, out / "a6sim_E5.png")
    text = report_text(res)
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"saved → {out}")
    return res


def report_text(res: dict) -> str:
    cols = ["none"] + PRIMARY + SECONDARY
    L = ["A6-sim — simulated whole-zone hand shift, predicted on the PC (board = PC)  ·  PROJECT_FACTS §5.1t (A13)",
         f"generated {res['generated']}", ""]
    for ds in res["datasets"].values():
        L.append(f"══ {ds['name']}  ({ds['user']}, {ds['n_takes']} takes) ══")
        L.append(f"{'model':<17}" + "".join(f"{s:>7}" for s in cols) + "   (per-frame accuracy %)")
        mods = ds["models"]
        singles = [m for m in mods if m.startswith("seed")]
        for m in ["baseline_ST_only", "E5"] + [m for m in mods if m.startswith("extra:")]:
            L.append(f"{m[:17]:<17}" + "".join(f"{mods[m][s]['acc']:>7.1f}" for s in cols))
        if singles:
            L.append(f"{'singles mean':<17}" + "".join(f"{np.mean([mods[m][s]['acc'] for m in singles]):>7.1f}"
                                                       for s in cols))
            L.append(f"{'singles SD':<17}" + "".join(f"{np.std([mods[m][s]['acc'] for m in singles], ddof=1):>7.1f}"
                                                     for s in cols))
        L.append(f"{'E5 vote':<17}" + "".join(f"{mods['E5'][s]['vote_acc']:>7.1f}" for s in cols))
        L.append(f"{'hand lost %':<17}" + "".join(f"{ds['lost_fraction'][s] * 100:>7.1f}" for s in cols))
        L.append("E5 recall % by posture:")
        for c in CLASSES:
            if not np.isnan(mods["E5"]["none"]["recall"][c]):
                L.append(f"  {c:<15}" + "".join(f"{mods['E5'][s]['recall'][c]:>7.1f}" for s in cols))
        L.append("E5 frames correct→wrong / wrong→correct vs unshifted: "
                 + ", ".join(f"{s} {mods['E5'][s]['correct_to_wrong']}/{mods['E5'][s]['wrong_to_correct']}"
                             for s in cols[1:]))
        L.append("")
    L.append("Pre-registered predictions (§5.1t):")
    for p in res["predictions"]:
        tag = {True: "HOLDS", False: "FAILS", None: "N/A  "}[p["holds"]]
        L.append(f"  {tag}  {p['name']}   [{p['value']}]")
    d = res["decision"]
    L.append("")
    L.append("Decision rule (any ±1 shift lowers E5 by >= 10 pp on phase56 or phase56_stand → pre-register 5.5c): "
             + ("TRIGGERED — " + "; ".join(d["hits"]) if d["trigger_5_5c"] else "not triggered"))
    L.append("limits: simulated shift (no perspective / edge-illumination change); n = 1 subject; frames correlated → "
             "descriptive only")
    return "\n".join(L)


if __name__ == "__main__":
    main()
