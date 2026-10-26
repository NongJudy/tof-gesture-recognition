"""a6_eval.py — A6: เลื่อนมือ 1 โซนแล้วบอร์ดยังทายถูกไหม ทุก 7 ท่า (PROJECT_FACTS §5.1s)

กติกาทั้งหมดจดไว้ใน §5.1s ก่อนเก็บข้อมูล — สคริปต์นี้แค่อ่านและนับ ไม่ปรับอะไร
  แผน: data\\a6_shift\\a6_plan_User<n>.json (a6_plan.py) — take k ของแต่ละท่า = ตำแหน่ง + เป้า (แถว, คอลัมน์)
  ต่อ 1 รอบ (เหมือน A5): ใช้ผลทายของ "บอร์ด" ทุกเฟรมที่มือในช่วง
       ท่าที่บอร์ดทายบ่อยที่สุด = ผลของรอบ ; ได้ ≥ 80 % → ผล = ท่านั้น, ไม่ถึง → "unstable"
       ถูก = ผลตรงกับท่าที่ทำ
  ตำแหน่งจริง: centroid มือของรอบ (§5.1j) → ระยะเลื่อนจริงจากตำแหน่งปกติ (Δแถว, Δคอลัมน์) และห่างเป้าเท่าไร
  สรุป: ท่า × ตำแหน่ง (x/2), ต่อตำแหน่ง (x/14), ต่อท่า (x/10), ปกติ vs เลื่อน, สัดส่วนเฟรมที่ถูก
  --extra-model: ให้โมเดลอื่นทายข้อมูลชุดเดียวกันบน PC ด้วยกติกาเดียวกัน (ไว้เทียบก่อน/หลัง 5.5c — รายงานเท่านั้น
                 ห้ามใช้ A6 เลือกโมเดล §5.1l)
  ข้อจำกัด: n = 1 คน, 2 รอบต่อช่อง → เชิงพรรณนา

วิธีรัน (ที่ root ของ repo):
    python pc_tools\\a6_eval.py
ตัวเลือก: --data-dir data\\a6_shift --plan data\\a6_shift\\a6_plan_User5.json --e5 results_phase55_final\\deploy\\E5.keras
          --no-pc-check --extra-model <path.keras> [--extra-model ...]
ผลลัพธ์: results_a6\\eval_<วันเวลา>\\ report.txt, summary.json, per_take.csv, per_frame.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from evaluate_phase56 import (CLASSES, board_vs_pc, centroid, import_st_dataset, load_takes, predict_takes, sha256_file,
                              timing_stats)

STABLE_SHARE = 0.80                                          # เหมือน A5 (§5.1q)
POSITIONS = ["natural", "row+1", "row-1", "col+1", "col-1"]


def judge(seq: np.ndarray, label: int) -> dict:
    """ตัดสิน 1 รอบจากผลทายรายเฟรม (ค่า ≥ 0 = เฟรมในช่วง)"""
    seq = seq[seq >= 0]
    if len(seq) == 0:
        return {"n_pred": 0, "top": None, "share": 0.0, "correct_share": 0.0, "result": "no-data", "correct": False,
                "counts": {}}
    cnt = Counter(int(c) for c in seq)
    top, top_n = cnt.most_common(1)[0]
    share = top_n / len(seq)
    result = CLASSES[top] if share >= STABLE_SHARE else "unstable"
    return {"n_pred": int(len(seq)), "top": CLASSES[top], "share": share, "correct_share": cnt.get(label, 0) / len(seq),
            "result": result, "correct": result == CLASSES[label],
            "counts": {CLASSES[k]: v for k, v in sorted(cnt.items())}}


def tables(rows: list[dict], key: str) -> dict:
    """นับถูก/ทั้งหมด ต่อ (ท่า, ตำแหน่ง), ต่อตำแหน่ง, ต่อท่า, ปกติ vs เลื่อน — key = คอลัมน์ผลที่ใช้ (board/pc/extra)"""
    def frac(sub):
        return {"correct": sum(r[key]["correct"] for r in sub), "n": len(sub),
                "mean_correct_share": float(np.mean([r[key]["correct_share"] for r in sub])) if sub else float("nan")}
    out = {"cell": {}, "position": {}, "posture": {}}
    for cls in CLASSES:
        for pos in POSITIONS:
            out["cell"][f"{cls}|{pos}"] = frac([r for r in rows if r["cls"] == cls and r["position"] == pos])
        out["posture"][cls] = frac([r for r in rows if r["cls"] == cls])
    for pos in POSITIONS:
        out["position"][pos] = frac([r for r in rows if r["position"] == pos])
    out["natural"] = frac([r for r in rows if r["position"] == "natural"])
    out["shifted"] = frac([r for r in rows if r["position"] != "natural"])
    out["all"] = frac(rows)
    return out


def main(argv: list[str] | None = None, load_model=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/a6_shift"))
    ap.add_argument("--plan", type=Path, default=None, help="ค่าเริ่มต้น: <data-dir>/a6_plan_User<n>.json")
    ap.add_argument("--e5", type=Path, default=Path("results_phase55_final/deploy/E5.keras"))
    ap.add_argument("--no-pc-check", action="store_true")
    ap.add_argument("--extra-model", type=Path, action="append", default=[],
                    help="โมเดลอื่นที่จะทายข้อมูลชุดเดียวกันบน PC (รายงานเท่านั้น)")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    for mp in ([] if args.no_pc_check else [args.e5]) + args.extra_model:
        if not mp.is_file():
            raise SystemExit(f"{mp} not found" + ("" if mp in args.extra_model else " (or use --no-pc-check)"))
    st = import_st_dataset()
    takes = load_takes(args.data_dir, st)
    users = sorted({t["user"] for t in takes})
    if len(users) != 1:
        raise SystemExit(f"one subject per run please, found {users} — use separate --data-dir")
    plan_path = args.plan or args.data_dir / f"a6_plan_{users[0]}.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    seen = Counter((t["cls"], t["take"]) for t in takes)
    dup = [k for k, v in seen.items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate (posture, take): {dup} — keep one file per take")
    takes.sort(key=lambda t: (CLASSES.index(t["cls"]), t["take"]))

    if load_model is None and (not args.no_pc_check or args.extra_model):
        import tensorflow as tf  # โหลดเฉพาะตอนต้องใช้
        load_model = tf.keras.models.load_model
    pc_cls = pc = None
    if not args.no_pc_check:
        model = load_model(str(args.e5))
        pc_cls, pc_prob = predict_takes(takes, lambda X: model.predict(X, verbose=0))
        pc = board_vs_pc(takes, pc_cls, pc_prob)
        pc["e5_sha256"] = sha256_file(args.e5)
    extra_cls = {}
    for mp in args.extra_model:
        m = load_model(str(mp))
        extra_cls[mp.name], _ = predict_takes(takes, lambda X, m=m: m.predict(X, verbose=0))

    rows = []
    for i, t in enumerate(takes):
        if not t["board"]["present"]:
            raise SystemExit(f"{t['file'].name}: _board.csv missing — collect with collect_own_data.py v6+")
        p = plan["classes"][t["cls"]]
        tk = p["takes"][t["take"] - 1]
        r, c = centroid([t["hand"]]) if len(t["hand"]) else (float("nan"), float("nan"))
        row = {"cls": t["cls"], "take": t["take"], "position": tk["position"], "file": t["file"].name, "n": t["n"],
               "n_in_range": int(t["mask"].sum()), "target_row": tk["target_row"], "target_col": tk["target_col"],
               "row": r, "col": c, "d_row": r - p["natural_row"], "d_col": c - p["natural_col"],
               "miss_row": r - tk["target_row"], "miss_col": c - tk["target_col"],
               "board": judge(t["board"]["cls"], t["label"])}
        if pc_cls is not None:
            row["pc"] = judge(pc_cls[i], t["label"])
        for name, cl in extra_cls.items():
            row[f"extra:{name}"] = judge(cl[i], t["label"])
        rows.append(row)

    res = {"generated": datetime.now().isoformat(timespec="seconds"), "data_dir": str(args.data_dir),
           "plan": str(plan_path), "plan_sha256": sha256_file(plan_path), "user": users[0],
           "stable_share": STABLE_SHARE, "n_takes": len(rows),
           "missing": [f"{c}|{k}" for c in CLASSES for k in range(1, len(plan["classes"][c]["takes"]) + 1)
                       if (c, k) not in seen],
           "board": tables(rows, "board"), "board_vs_pc": pc, "timing": timing_stats(takes),
           "placement": {pos: {"d_row_mean": float(np.mean([r["d_row"] for r in rows if r["position"] == pos])),
                               "d_col_mean": float(np.mean([r["d_col"] for r in rows if r["position"] == pos])),
                               "abs_miss_max": float(max(max(abs(r["miss_row"]), abs(r["miss_col"]))
                                                         for r in rows if r["position"] == pos))}
                         for pos in POSITIONS if any(r["position"] == pos for r in rows)},
           "extra": {name: tables(rows, f"extra:{name}") for name in extra_cls}, "takes": rows}

    out = args.out_dir or Path("results_a6") / f"eval_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "per_take.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["posture", "take", "position", "target_row", "target_col", "row", "col", "d_row", "d_col",
                    "board_result", "board_correct", "board_correct_share", "n_in_range", "file"])
        for r in rows:
            w.writerow([r["cls"], r["take"], r["position"], r["target_row"], r["target_col"], f"{r['row']:.3f}",
                        f"{r['col']:.3f}", f"{r['d_row']:.3f}", f"{r['d_col']:.3f}", r["board"]["result"],
                        int(r["board"]["correct"]), f"{r['board']['correct_share']:.4f}", r["n_in_range"], r["file"]])
    with open(out / "per_frame.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["posture", "take", "position", "frame_idx", "in_range", "board_class"])
        for t, r in zip(takes, rows):
            for i in range(t["n"]):
                bc = int(t["board"]["cls"][i])
                w.writerow([t["cls"], t["take"], r["position"], i, int(t["mask"][i]), CLASSES[bc] if bc >= 0 else bc])
    text = report_text(res)
    (out / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"saved → {out}")
    return res


def grid(tab: dict, title: str) -> list[str]:
    """ตารางท่า × ตำแหน่ง (ถูก/ทั้งหมด)"""
    L = [title, f"{'posture':<11}" + "".join(f"{p:>9}" for p in POSITIONS) + f"{'total':>9}"]
    for cls in CLASSES:
        cells = [tab["cell"][f"{cls}|{p}"] for p in POSITIONS]
        L.append(f"{cls:<11}" + "".join(f"{c['correct']:>6}/{c['n']:<2}" for c in cells)
                 + f"{tab['posture'][cls]['correct']:>6}/{tab['posture'][cls]['n']:<2}")
    L.append(f"{'total':<11}" + "".join(f"{tab['position'][p]['correct']:>6}/{tab['position'][p]['n']:<2}"
                                        for p in POSITIONS) + f"{tab['all']['correct']:>6}/{tab['all']['n']:<2}")
    L.append(f"{'frames ok %':<11}" + "".join(f"{tab['position'][p]['mean_correct_share']:>9.1%}" for p in POSITIONS)
             + f"{tab['all']['mean_correct_share']:>9.1%}")
    n, s = tab["natural"], tab["shifted"]
    L.append(f"natural {n['correct']}/{n['n']} takes ({n['mean_correct_share']:.1%} frames)   shifted "
             f"{s['correct']}/{s['n']} takes ({s['mean_correct_share']:.1%} frames)")
    return L


def report_text(res: dict) -> str:
    L = ["A6 — 1-zone hand shift, all 7 postures, predicted ON THE BOARD (E5)  ·  PROJECT_FACTS §5.1s",
         f"data: {res['data_dir']}   subject: {res['user']}   takes: {res['n_takes']}   generated: {res['generated']}",
         f"plan: {res['plan']}  sha256 {res['plan_sha256'][:12]}…",
         f"rule: per take, board's most frequent class over in-range frames; counted only if share >= "
         f"{res['stable_share']:.0%}, else 'unstable'; correct = matches the performed posture", ""]
    if res["missing"]:
        L += [f"MISSING takes: {', '.join(res['missing'])}", ""]
    L += grid(res["board"], "Board, correct takes (posture × position)") + [""]
    L.append("Placement achieved (mean shift from natural, in zones; worst miss from target):")
    for pos, p in res["placement"].items():
        L.append(f"  {pos:<8} d_row {p['d_row_mean']:+.2f}  d_col {p['d_col_mean']:+.2f}  worst miss {p['abs_miss_max']:.2f}")
    L.append("")
    wrong = [r for r in res["takes"] if not r["board"]["correct"]]
    if wrong:
        L.append("Takes not correct (board):")
        for r in wrong:
            L.append(f"  {r['cls']:<11} take {r['take']:>2} {r['position']:<8} → {r['board']['result']:<10} "
                     f"(correct frames {r['board']['correct_share']:.0%}; d_row {r['d_row']:+.2f} d_col {r['d_col']:+.2f}) "
                     f"{r['board']['counts']}")
        L.append("")
    pc = res["board_vs_pc"]
    L.append(f"board vs PC (E5.keras): {pc['agree']}/{pc['compared']} frames agree, gate mismatches {pc['gate_mismatch']}, "
             f"missing rows {pc['missing_rows']}, max |Δp| {pc['max_abs_diff']:.2e}" if pc
             else "board vs PC: skipped (--no-pc-check)")
    tm = res["timing"]
    if tm["n"]:
        tt = tm["total_us"]
        L.append(f"board per-frame work: n={tm['n']}, mean {tt['mean']:.0f} µs (SD {tt['sd']:.0f}), max {tt['max']:.0f} µs, "
                 f"over 33 ms: {tm['over_budget']}")
    for name, tab in res["extra"].items():
        L += [""] + grid(tab, f"Extra model on PC (report only, never for selection): {name}")
    L.append("limits: n = 1 subject, 2 takes per cell, frames within a take are correlated → descriptive only")
    return "\n".join(L)


if __name__ == "__main__":
    main()
