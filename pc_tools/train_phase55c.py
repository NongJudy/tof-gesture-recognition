"""train_phase55c.py — Phase 5.5c: ย้ายมือให้อยู่กลางภาพ (re-centring) ดีกว่าสูตรเดิมไหม — วัดแบบ LOSO 6 คน
(PROJECT_FACTS §5.1w — ลงทะเบียนล่วงหน้าก่อนรัน)

  แขน B : สูตรของ E5 เดิมทุกอย่าง (train_phase55 แขน B: โครงสร้าง ST + ถ่วงน้ำหนักให้ทุกคนเท่ากัน)
          รันใหม่ในรอบเดียวกัน เพื่อเทียบแบบจับคู่ fold/seed เดียวกันเป๊ะ (ไม่ใช้ตัวเลขเก่า 20 ต.ค.)
  แขน R : แขน B + recentre() ทั้งตอนฝึก (หลัง flip) และตอนทดสอบ — เปลี่ยนอย่างเดียว
  ทุกแขน: seeds 42–46, 6 fold (ST 4 คน + User5 + User6), fold ของคนเรา ทดสอบด้วยชุดหลัก data\\phase4 เท่านั้น
          (เหมือน §5.1i ทุกอย่าง — ใช้ load_all / fold_indices / subject_weights จาก train_phase55.py)
กติกาใช้ R (ล็อกไว้ก่อน §5.1w): ค่าเฉลี่ย 6 คนของ R − B ≥ +1.0 pp  และ  ค่าเฉลี่ย 4 fold ของ ST ของ R ≥ B − 1.0 pp

รันต่อจากที่ค้างได้ (runs.jsonl) — ปิดหน้าต่างแล้วรันคำสั่งเดิม จะข้ามโมเดลที่เสร็จแล้ว

วิธีรัน (ที่ root ของ repo; 60 โมเดล ประมาณ 2–3 ชั่วโมงบน CPU [ประมาณจากรอบ 90 โมเดล ยังไม่ได้วัด]):
    python pc_tools\\train_phase55c.py --st-data C:\\ds\\ST_VL53L8CX_handposture_dataset --ours data\\phase4 data\\phase4_extra
ทดสอบว่าระบบรันได้ (ผลไม่มีความหมาย): ... --smoke-test
ผลลัพธ์: results_phase55c\\ runs.jsonl, report.txt, summary.json
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
import train_phase55 as p55   # noqa: E402
from recentre import recentre, shifts  # noqa: E402

ARMS = ["B", "R"]
OUR_USERS = p55.OUR_USERS
ADOPT_MEAN6_PP = 1.0          # R − B บนค่าเฉลี่ย 6 คน ต้อง ≥ เท่านี้
ST_GUARD_PP = 1.0             # ค่าเฉลี่ย ST ของ R ห้ามต่ำกว่า B เกินเท่านี้
OLD_B_MEAN6 = 85.30           # §5.1i-R (ใช้ตรวจ P5 เท่านั้น)


def train_one(tr, arm: str, X_tr, y_tr, u_tr, X_te, y_te, seed: int, max_epochs: int) -> dict:
    """เทรน 1 โมเดล — เหมือน train_phase55.train_one แขน B ทุกบรรทัด ต่างแค่ R เรียก recentre()"""
    tf = tr.tf
    model = tr.build_model(seed)
    X_aug = tr.augment_flip(X_tr, seed)
    if arm == "R":
        X_aug = recentre(X_aug)          # หลัง flip → โมเดลเห็นแบบเดียวกับตอนใช้งานจริง
        X_te = recentre(X_te)
    perm = np.random.default_rng(seed).permutation(len(X_aug))
    X_fit, y_fit = X_aug[perm], y_tr[perm]
    w_fit = p55.subject_weights(u_tr)[perm]
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    hist = model.fit(X_fit, y_fit, sample_weight=w_fit, batch_size=32, epochs=max_epochs,
                     validation_split=0.1, callbacks=[stop], verbose=0)
    pred = np.argmax(model.predict(X_te, batch_size=256, verbose=0), axis=1)
    k = len(tr.CLASSES)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_te, pred):
        cm[int(t), int(p)] += 1
    recall = {c: (float(cm[i, i] / cm[i].sum() * 100) if cm[i].sum() else float("nan"))
              for i, c in enumerate(tr.CLASSES)}
    return {"acc": float((pred == y_te).mean() * 100), "recall": recall, "confusion": cm.tolist(),
            "epochs": len(hist.history["loss"]), "n_train": int(len(X_tr)), "n_test": int(len(X_te))}


def summarise(runs: list[dict], classes: list[str], st_users: list[int]) -> dict:
    """ค่าเฉลี่ย ± SD ข้าม seed ต่อ (แขน, คน) + กติกาใช้ R + Wilcoxon จับคู่ตามคน"""
    out: dict = {"cells": {}, "per_class": {}, "st_mean": {}, "mean6": {}}
    users = st_users + OUR_USERS
    for arm in ARMS:
        for u in users:
            rs = [r for r in runs if r["arm"] == arm and r["held_out"] == u]
            if rs:
                a = [r["acc"] for r in rs]
                out["cells"][f"{arm}|{u}"] = {"mean": float(np.mean(a)), "n": len(a),
                                              "sd": float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")}
                out["per_class"][f"{arm}|{u}"] = {c: float(np.nanmean([r["recall"][c] for r in rs])) for c in classes}
        st = [out["cells"].get(f"{arm}|{u}", {}).get("mean") for u in st_users]
        allc = [out["cells"].get(f"{arm}|{u}", {}).get("mean") for u in users]
        if all(v is not None for v in st):
            out["st_mean"][arm] = float(np.mean(st))
        if all(v is not None for v in allc):
            out["mean6"][arm] = float(np.mean(allc))
    diffs = [out["cells"][f"R|{u}"]["mean"] - out["cells"][f"B|{u}"]["mean"]
             for u in users if f"R|{u}" in out["cells"] and f"B|{u}" in out["cells"]]
    pair: dict = {"diffs_pp": diffs, "mean_diff_pp": float(np.mean(diffs)) if diffs else float("nan")}
    if len(diffs) >= 2:
        try:
            from scipy.stats import wilcoxon
            res = wilcoxon(diffs, method="exact")
            pair.update({"wilcoxon_W": float(res.statistic), "p": float(res.pvalue)})
        except (ImportError, ValueError, TypeError) as exc:
            pair["wilcoxon_error"] = str(exc)
    out["paired_R_minus_B"] = pair
    adopt = None
    if all(a in out["mean6"] for a in ARMS) and all(a in out["st_mean"] for a in ARMS):
        adopt = bool(out["mean6"]["R"] - out["mean6"]["B"] >= ADOPT_MEAN6_PP
                     and out["st_mean"]["R"] >= out["st_mean"]["B"] - ST_GUARD_PP)
    out["adopt_R"] = adopt
    return out


def render(s: dict, classes: list[str], st_users: list[int], smoke: bool) -> str:
    """ข้อความสรุปบนจอ + ตรวจคำทำนาย P1–P5 (§5.1w)"""
    def cell(arm: str, u: int) -> str:
        c = s["cells"].get(f"{arm}|{u}")
        return f"{c['mean']:5.1f}±{c['sd']:4.1f}" if c and not np.isnan(c["sd"]) else (f"{c['mean']:5.1f}" if c else "—")
    users = st_users + OUR_USERS
    L = [ev.LINE, "  Phase 5.5c  ·  re-centring (R) vs E5 recipe (B), leave-one-subject-out, 6 subjects  ·  §5.1w", ev.LINE]
    if smoke:
        L.append("  *** SMOKE TEST — numbers are meaningless, do not use ***")
    L.append(f"  {'held-out':<10}{'arm B':>14}{'arm R':>14}{'R − B':>10}")
    for u in users:
        b, r = s["cells"].get(f"B|{u}"), s["cells"].get(f"R|{u}")
        d = f"{r['mean'] - b['mean']:+.2f}" if b and r else "—"
        L.append(f"  User{u:<6}{cell('B', u):>14}{cell('R', u):>14}{d:>10}")
    for key, name in (("st_mean", "ST mean"), ("mean6", "mean of 6")):
        L.append(f"  {name:<10}" + "".join(f"{s[key].get(a, float('nan')):>14.2f}" for a in ARMS))
    p = s["paired_R_minus_B"]
    extra = f", Wilcoxon W={p['wilcoxon_W']:.1f}, p={p['p']:.3f}" if "p" in p else ""
    L.append(f"  R − B over {len(p['diffs_pp'])} subjects: mean {p['mean_diff_pp']:+.2f} pp{extra}")
    L.append(f"  Adopt R (mean6 R−B ≥ {ADOPT_MEAN6_PP:g} pp and ST mean R ≥ B − {ST_GUARD_PP:g} pp): "
             + {True: "YES", False: "NO", None: "incomplete"}[s["adopt_R"]])
    for arm in ARMS:
        L.append("")
        L.append(f"  Recall per class, arm {arm} (mean over seeds)")
        L.append(f"  {'':<10}" + "".join(f"{c[:9]:>10}" for c in classes))
        for u in users:
            pc = s["per_class"].get(f"{arm}|{u}")
            if pc:
                L.append(f"  User{u:<6}" + "".join(f"{pc[c]:>10.1f}" for c in classes))
    if s["adopt_R"] is not None:
        m6, stm = s["mean6"], s["st_mean"]
        u5b, u5r = s["cells"]["B|5"]["mean"], s["cells"]["R|5"]["mean"]
        like_r = s["per_class"]["R|5"]["Like"]
        chk = [("P1 mean of 6: R ≥ B + 1 pp", m6["R"] - m6["B"] >= 1, f"{m6['R'] - m6['B']:+.2f}"),
               ("P2 User5: R ≥ B + 3 pp", u5r - u5b >= 3, f"{u5r - u5b:+.2f}"),
               ("P3 User5 Like recall, arm R > 50 %", like_r > 50, f"{like_r:.1f}"),
               ("P4 ST mean |R − B| ≤ 2 pp", abs(stm["R"] - stm["B"]) <= 2, f"{stm['R'] - stm['B']:+.2f}"),
               ("P5 arm B re-run mean of 6 within ±1 pp of 85.30", abs(m6["B"] - OLD_B_MEAN6) <= 1, f"{m6['B']:.2f}")]
        L += ["", "  Pre-registered predictions (§5.1w)"]
        L += [f"    {'HOLDS' if ok else 'FAILS'}  {n}   [{v}]" for n, ok, v in chk]
    L += ["  Folds of our subjects test on their MAIN set only. n = 6 subjects: smallest exact two-sided p = 0.031.", ev.LINE]
    return "\n".join(L)


def main(argv: list[str] | None = None, trainer: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st-data", type=Path, required=True)
    ap.add_argument("--ours", type=Path, nargs="+", required=True)
    ap.add_argument("--train-script", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results_phase55c"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    tr_path = args.train_script or next((p for p in (here / "train_st_cnn2d.py", here.parent / "train_st_cnn2d.py")
                                         if p.is_file()), None)
    if tr_path is None:
        raise SystemExit("train_st_cnn2d.py not found — use --train-script <path>")
    tr = ev.import_module_from(Path(tr_path), "train_st_cnn2d")
    seeds, max_epochs, out_dir = args.seeds, 1000, args.out_dir
    if args.smoke_test:
        seeds, max_epochs = seeds[:1], 2
        out_dir = out_dir.with_name(out_dir.name + "_SMOKE_TEST")
    print("Loading ST + our data with train_st_cnn2d.load_dataset() ...")
    data = p55.load_all(tr, args.st_data, args.ours)
    users = sorted(set(data["user"].tolist()))
    st_users = [u for u in users if u not in OUR_USERS]
    if [u for u in OUR_USERS if u not in users] and not args.smoke_test:
        raise SystemExit("User5 and User6 are both required (§5.1w)")
    dr, dc, cnt = shifts(data["X"])
    print(f"  frames {len(data['y']):,}; re-centring shift |dr|,|dc| mean {np.abs(dr).mean():.2f}, {np.abs(dc).mean():.2f}; "
          f"frames with no hand zone {int((cnt == 0).sum())}")
    for u in users:
        m = data["user"] == u
        print(f"  User{u}: {int(m.sum()):,} frames, mean shift dr {dr[m].mean():+.2f}, dc {dc[m].mean():+.2f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    done = [json.loads(x) for x in runs_path.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if runs_path.exists() else []
    keys = {(r["arm"], r["held_out"], r["seed"]) for r in done}
    plan = [(a, u, s) for u in users for s in seeds for a in ARMS if (a, u, s) not in keys]   # B,R สลับกัน
    print(f"Plan: {len(plan)} models ({len(keys)} already done in {runs_path})")
    fn = trainer or (lambda *a: train_one(tr, *a))
    t0 = time.time()
    for i, (arm, u, seed) in enumerate(plan, 1):
        tr_idx, te_idx = p55.fold_indices(data, u)
        r = fn(arm, data["X"][tr_idx], data["y"][tr_idx], data["user"][tr_idx],
               data["X"][te_idx], data["y"][te_idx], seed, max_epochs)
        r.update({"arm": arm, "held_out": u, "seed": seed, "at": datetime.now().isoformat(timespec="seconds")})
        with runs_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
        done.append(r)
        el = time.time() - t0
        print(f"  [{i}/{len(plan)}] arm {arm}  held-out User{u}  seed {seed}: acc {r['acc']:.1f} %  ({r['epochs']} ep)"
              f"  ·  about {el / i * (len(plan) - i) / 60:.0f} min left")

    s = summarise(done, list(tr.CLASSES), st_users)
    rep = render(s, list(tr.CLASSES), st_users, args.smoke_test)
    print("\n" + rep)
    (out_dir / "report.txt").write_text(rep + "\n", encoding="utf-8")
    meta = {"run_at": datetime.now().isoformat(timespec="seconds"), "smoke_test": args.smoke_test, "seeds": seeds,
            "train_script_sha256": ev.sha256_file(Path(tr_path)), "git_commit": ev.git_commit(), "users": users}
    (out_dir / "summary.json").write_text(json.dumps({"meta": meta, "summary": s}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return {"summary": s, "out": out_dir}


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด
#  1. ถ้าแก้โค้ด/ข้อมูลระหว่างรัน ให้ลบ runs.jsonl แล้วเริ่มใหม่ ไม่งั้นผลปนกัน
#  2. recentre() ต้องเป็นตัวเดียวกับที่ใช้ทดสอบและในเฟิร์มแวร์ (เลขจำนวนเต็ม) — อย่าแก้สูตรหลังเห็นผล
#  3. smoke test เขียนลงโฟลเดอร์ _SMOKE_TEST แยก ห้ามเอาผลไปรายงาน
# ─────────────────────────────────────────────────────────────────────────
