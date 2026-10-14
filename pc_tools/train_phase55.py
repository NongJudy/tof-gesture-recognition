"""train_phase55.py — Phase 5.5: เทรนด้วยข้อมูล ST + ผู้ทดลองของเรา (User 5, User 6) แล้ววัดแบบ
leave-one-subject-out 6 คน เทียบ 3 แขน (A / B / C) ตาม pre-registration PROJECT_FACTS §5.1i

  แขน A : โครงสร้าง ST (train_st_cnn2d.build_model) ไม่ถ่วงน้ำหนัก
  แขน B : โครงสร้าง ST + ถ่วงน้ำหนักให้ "ทุกคนมีน้ำหนักรวมเท่ากัน" (ข้อมูลเรา ~12,600 เฟรม/คน
          มากกว่า ST ~2,100-4,600 ต่อคน ถ้าไม่ถ่วง ข้อมูลเราจะกิน ~70 % ของการเทรน)
  แขน C : โครงสร้าง ST แต่เปลี่ยน Flatten → GlobalAveragePooling2D (ไม่ยึดติดตำแหน่งมือ)
  ทุกแขน: กลับภาพซ้าย-ขวา, สลับลำดับก่อน validation_split=0.1 (แก้บั๊ก #51), Adam 0.01, batch 32,
          early stopping patience 30, สูงสุด 1000 epochs, seeds 42-46

fold ของคนที่ถูกกันไว้:
  - คนของ ST : เทรนด้วย ST อีก 3 คน + ข้อมูลเราทั้ง 2 คน (ชุดหลัก + รอบเพิ่ม) ทดสอบกับคนนั้นทั้งหมด
  - คนของเรา : เทรนด้วย ST 4 คน + อีกคนของเรา (ชุดหลัก + รอบเพิ่ม)
               ทดสอบกับ "ชุดหลัก" (data\\phase4) ของคนนั้นเท่านั้น — รอบเพิ่มของคนนั้นไม่ถูกใช้ใน fold นี้เลย
กติกาเลือกแขน (ล็อกไว้ก่อน): ค่าเฉลี่ยบน 4 fold ของ ST สูงสุดชนะ ถ้าห่างจาก A ไม่เกิน 1 จุด เลือก A
  → ผลของคนเรา 2 คน "ไม่ได้ใช้เลือก" แค่รายงาน กันตัวเลขสูงเกินจริง

รันต่อจากที่ค้างได้: ผลแต่ละโมเดลถูกบันทึกทันทีใน runs.jsonl ปิดหน้าต่างแล้วรันคำสั่งเดิม จะข้ามโมเดลที่เสร็จแล้ว

วิธีรัน (ที่ root ของ repo; ใช้เวลาหลายชั่วโมงบน CPU — 90 โมเดล [ประมาณ ยังไม่ได้วัด]):
    python pc_tools\\train_phase55.py --st-data C:\\ds\\ST_VL53L8CX_handposture_dataset
        --ours data\\phase4 data\\phase4_extra
ทดสอบว่าระบบรันได้ (ผลไม่มีความหมาย ห้ามเอาไปใช้):
    ... --smoke-test     (1 seed, 2 epochs, เขียนลงโฟลเดอร์แยก)

ผลลัพธ์: results_phase55\\runs.jsonl (ทุกโมเดล), report.txt, summary.json
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

ARMS = ["A", "B", "C"]
OUR_USERS = [5, 6]
SELECT_MARGIN_PP = 1.0               # กติกาเลือกแขน §5.1i
PHASE2_FIXED_LOSO = 85.28            # Q1 baseline (§5.1h-R)
USER5_ST_ONLY = 68.8                 # Q2 baseline (§5.1h-R)


# ---------------------------------------------------------------- ข้อมูล
def load_all(tr, st_dir: Path, our_dirs: list[Path]) -> dict:
    """โหลด ST + ข้อมูลเรา ด้วย train_st_cnn2d.load_dataset (เตรียมข้อมูลแบบเดียวกับตอนเทรนจริง)
    คืน dict: X, y, user, is_main (True = ชุดหลักของเรา หรือข้อมูล ST)"""
    st = tr.load_dataset(st_dir)
    parts_X, parts_y, parts_u, parts_main = [st.X], [st.y], [st.groups.astype(int)], [np.ones(len(st.y), bool)]
    if set(np.unique(st.groups).tolist()) & set(OUR_USERS):
        raise SystemExit("ST data contains User5/User6 — wrong --st-data folder?")
    for d in our_dirs:
        ds = tr.load_dataset(d)
        users = set(np.unique(ds.groups).tolist())
        if not users <= set(OUR_USERS):
            raise SystemExit(f"{d} contains users {sorted(users)} — expected only {OUR_USERS}")
        main = "extra" not in d.name.lower()
        parts_X.append(ds.X)
        parts_y.append(ds.y)
        parts_u.append(ds.groups.astype(int))
        parts_main.append(np.full(len(ds.y), main))
    return {"X": np.concatenate(parts_X).astype(np.float32), "y": np.concatenate(parts_y),
            "user": np.concatenate(parts_u), "is_main": np.concatenate(parts_main)}


def fold_indices(data: dict, held_out: int) -> tuple[np.ndarray, np.ndarray]:
    """train / test index ของ fold ที่กัน held_out ไว้ (ตามนิยามใน §5.1i)"""
    u, main = data["user"], data["is_main"]
    train = np.where(u != held_out)[0]
    test = np.where((u == held_out) & (main if held_out in OUR_USERS else True))[0]
    return train, test


def subject_weights(users: np.ndarray) -> np.ndarray:
    """แขน B: น้ำหนักต่อเฟรม = 1 / จำนวนเฟรมของคนนั้น แล้ว normalize ให้ค่าเฉลี่ยเป็น 1
    → ทุกคนมีน้ำหนักรวมเท่ากัน ไม่ว่าจะมีกี่เฟรม"""
    uniq, counts = np.unique(users, return_counts=True)
    per = {int(a): 1.0 / c for a, c in zip(uniq, counts)}
    w = np.array([per[int(x)] for x in users], dtype=np.float64)
    return (w / w.mean()).astype(np.float32)


# ---------------------------------------------------------------- โมเดล
def build_gap_model(tr, seed: int):
    """แขน C: เหมือน build_model ของ train_st_cnn2d ทุกชั้น ยกเว้น Flatten → GlobalAveragePooling2D"""
    tf, layers, models = tr.tf, tr.layers, tr.models
    tf.keras.utils.set_random_seed(seed)
    model = models.Sequential([
        layers.Input(shape=(8, 8, 2)),
        layers.Conv2D(8, 3, padding="same"),
        layers.Activation("relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dense(tr.N_CLASSES, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_one(tr, arm: str, X_tr, y_tr, u_tr, X_te, y_te, seed: int, max_epochs: int) -> dict:
    """เทรน 1 โมเดล (1 แขน × 1 fold × 1 seed) แล้ววัดผลบน test"""
    tf = tr.tf
    model = build_gap_model(tr, seed) if arm == "C" else tr.build_model(seed)
    X_aug = tr.augment_flip(X_tr, seed)
    perm = np.random.default_rng(seed).permutation(len(X_aug))      # แก้บั๊ก #51: สลับก่อนแบ่ง
    X_fit, y_fit = X_aug[perm], y_tr[perm]
    w_fit = subject_weights(u_tr)[perm] if arm == "B" else None
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


# ---------------------------------------------------------------- สรุปผล
def summarise(runs: list[dict], classes: list[str], st_users: list[int]) -> dict:
    """ค่าเฉลี่ย ± SD ข้าม seed ต่อ (แขน, คน) + กติกาเลือกแขน + เทียบแขนแบบจับคู่ตามคน"""
    out: dict = {"cells": {}, "st_mean": {}, "per_class": {}}
    for arm in ARMS:
        for u in st_users + OUR_USERS:
            accs = [r["acc"] for r in runs if r["arm"] == arm and r["held_out"] == u]
            if accs:
                out["cells"][f"{arm}|{u}"] = {"mean": float(np.mean(accs)), "n": len(accs),
                                              "sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else float("nan")}
                rec = [r["recall"] for r in runs if r["arm"] == arm and r["held_out"] == u]
                out["per_class"][f"{arm}|{u}"] = {c: float(np.nanmean([x[c] for x in rec])) for c in classes}
        st_cells = [out["cells"].get(f"{arm}|{u}", {}).get("mean") for u in st_users]
        if all(v is not None for v in st_cells):
            out["st_mean"][arm] = float(np.mean(st_cells))
    sel = None
    if all(a in out["st_mean"] for a in ARMS):
        best = max(ARMS, key=lambda a: out["st_mean"][a])
        sel = "A" if out["st_mean"][best] - out["st_mean"]["A"] <= SELECT_MARGIN_PP else best
    out["selected"] = sel
    # เทียบแขนแบบจับคู่ตามคน (6 fold) — Wilcoxon signed-rank (exact) ถ้ามี scipy
    out["paired"] = {}
    for other in ("B", "C"):
        diffs = []
        for u in st_users + OUR_USERS:
            a, b = out["cells"].get(f"A|{u}"), out["cells"].get(f"{other}|{u}")
            if a and b:
                diffs.append(b["mean"] - a["mean"])
        entry: dict = {"diffs_pp": diffs, "mean_diff_pp": float(np.mean(diffs)) if diffs else float("nan")}
        if len(diffs) >= 2:
            try:
                from scipy.stats import wilcoxon
                res = wilcoxon(diffs, method="exact")
                entry.update({"wilcoxon_W": float(res.statistic), "p": float(res.pvalue)})
            except (ImportError, ValueError, TypeError) as exc:
                entry["wilcoxon_error"] = str(exc)
        out["paired"][f"{other}-A"] = entry
    return out


def render(summary: dict, classes: list[str], st_users: list[int], smoke: bool) -> str:
    """ข้อความสรุปบนจอ"""
    def cell(arm: str, u: int) -> str:
        c = summary["cells"].get(f"{arm}|{u}")
        return f"{c['mean']:5.1f}±{c['sd']:4.1f}" if c and not np.isnan(c["sd"]) else (
            f"{c['mean']:5.1f}     " if c else "    —     ")
    L = [ev.LINE, "  Phase 5.5  ·  ST + our subjects, leave-one-subject-out (6 subjects)  ·  §5.1i", ev.LINE]
    if smoke:
        L.append("  *** SMOKE TEST — numbers are meaningless, do not use ***")
    L.append(f"  {'held-out':<10}" + "".join(f"{'arm ' + a:>15}" for a in ARMS))
    for u in st_users + OUR_USERS:
        L.append(f"  User{u:<6}" + "".join(f"{cell(a, u):>15}" for a in ARMS))
    L.append(f"  {'ST mean':<10}" + "".join(
        f"{summary['st_mean'].get(a, float('nan')):>15.2f}" for a in ARMS))
    sel = summary["selected"]
    L.append(f"  Selected arm (highest ST-fold mean, ties ≤ {SELECT_MARGIN_PP:g} pp → A): {sel or 'incomplete'}")
    for k, v in summary["paired"].items():
        extra = f"  Wilcoxon W={v['wilcoxon_W']:.1f}, p={v['p']:.3f}" if "p" in v else ""
        L.append(f"  {k}: mean {v['mean_diff_pp']:+.2f} pp over {len(v['diffs_pp'])} subjects{extra}")
    if sel:
        L.append("")
        L.append(f"  Recall per class, arm {sel} (mean over seeds)")
        L.append(f"  {'':<10}" + "".join(f"{c[:9]:>10}" for c in classes))
        for u in st_users + OUR_USERS:
            pc = summary["per_class"].get(f"{sel}|{u}")
            if pc:
                L.append(f"  User{u:<6}" + "".join(f"{pc[c]:>10.1f}" for c in classes))
        st_a = summary["st_mean"].get("A", float("nan"))
        u5 = summary["cells"].get("A|5", {}).get("mean", float("nan"))
        like5 = summary["per_class"].get("A|5", {}).get("Like", float("nan"))
        L += ["", "  Pre-registered checks (§5.1i)",
              f"    Q1  arm A ST-fold mean {st_a:.2f} ≥ {PHASE2_FIXED_LOSO - 3:.2f}  →  "
              f"{'HOLDS' if st_a >= PHASE2_FIXED_LOSO - 3 else 'DOES NOT HOLD'}",
              f"    Q2  arm A User5 {u5:.1f} > {USER5_ST_ONLY}  →  {'HOLDS' if u5 > USER5_ST_ONLY else 'DOES NOT HOLD'}",
              f"    Q3  arm A User5 Like recall {like5:.1f} > 30  →  {'HOLDS' if like5 > 30 else 'DOES NOT HOLD'}",
              f"    Q4  arm B ST mean {summary['st_mean'].get('B', float('nan')):.2f} ≥ A {st_a:.2f}  →  "
              f"{'HOLDS' if summary['st_mean'].get('B', -1) >= st_a else 'DOES NOT HOLD'}",
              f"    Q5  arm C ST mean ≤ A − 3  →  "
              f"{'HOLDS' if summary['st_mean'].get('C', 999) <= st_a - 3 else 'DOES NOT HOLD'}"]
    L += ["  Folds of our subjects test on their MAIN set only; their extras never used in their own fold.", ev.LINE]
    return "\n".join(L)


def main(argv: list[str] | None = None, trainer: Callable | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st-data", type=Path, required=True)
    ap.add_argument("--ours", type=Path, nargs="+", required=True, help=r"data\phase4 data\phase4_extra")
    ap.add_argument("--train-script", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results_phase55"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--smoke-test", action="store_true", help="1 seed, 2 epochs — ทดสอบว่ารันได้ ผลห้ามใช้")
    args = ap.parse_args(argv)

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
    data = load_all(tr, args.st_data, args.ours)
    users_present = sorted(set(data["user"].tolist()))
    st_users = [u for u in users_present if u not in OUR_USERS]
    missing = [u for u in OUR_USERS if u not in users_present]
    if missing and not args.smoke_test:
        raise SystemExit(f"our subjects missing: {missing} — Phase 5.5 needs both User5 and User6 (§5.1i). "
                         "Use --smoke-test only to check the pipeline.")
    for u in users_present:
        n_all = int((data["user"] == u).sum())
        n_main = int(((data["user"] == u) & data["is_main"]).sum())
        print(f"  User{u}: {n_all:,} frames" + (f" (main {n_main:,}, extra {n_all - n_main:,})" if u in OUR_USERS else ""))

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    done = []
    if runs_path.exists():
        done = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    done_keys = {(r["arm"], r["held_out"], r["seed"]) for r in done}
    plan = [(a, u, s) for a in args.arms for u in users_present for s in seeds if (a, u, s) not in done_keys]
    print(f"Plan: {len(plan)} models to train ({len(done_keys)} already done, resumed from {runs_path})")
    fn = trainer or (lambda *a: train_one(tr, *a))
    t0 = time.time()
    for i, (arm, u, seed) in enumerate(plan, 1):
        tr_idx, te_idx = fold_indices(data, u)
        r = fn(arm, data["X"][tr_idx], data["y"][tr_idx], data["user"][tr_idx],
               data["X"][te_idx], data["y"][te_idx], seed, max_epochs)
        r.update({"arm": arm, "held_out": u, "seed": seed, "at": datetime.now().isoformat(timespec="seconds")})
        with runs_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
        done.append(r)
        el = time.time() - t0
        print(f"  [{i}/{len(plan)}] arm {arm}  held-out User{u}  seed {seed}: acc {r['acc']:.1f} %  "
              f"({r['epochs']} ep)  ·  about {el / i * (len(plan) - i) / 60:.0f} min left")

    summary = summarise(done, list(tr.CLASSES), st_users)
    report = render(summary, list(tr.CLASSES), st_users, args.smoke_test)
    print("\n" + report)
    (out_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    meta = {"run_at": datetime.now().isoformat(timespec="seconds"), "smoke_test": args.smoke_test,
            "seeds": seeds, "arms": args.arms, "train_script": str(tr_path),
            "train_script_sha256": ev.sha256_file(Path(tr_path)), "git_commit": ev.git_commit(),
            "users": users_present, "n_frames": int(len(data["y"]))}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, ensure_ascii=False, indent=2)
    return {"summary": summary, "out": out_dir, "runs": done}


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────
# จุดที่อาจพลาด (failure modes)
#  1. ใช้เวลานานมาก (90 โมเดล) — รันต่อจากที่ค้างได้ แต่ถ้าแก้โค้ดหรือข้อมูลระหว่างทาง ให้ลบ runs.jsonl
#     แล้วเริ่มใหม่ ไม่งั้นผลเก่าและใหม่จะปนกัน
#  2. sample_weight + validation_split: Keras ตัดน้ำหนักไปพร้อมข้อมูล [จำมา] — validation ของแขน B
#     จึงถ่วงน้ำหนักด้วย มีผลแค่การหยุดเทรน ไม่กระทบตัวเลขทดสอบ
#  3. เฟรมในรอบเดียวกันติดกันมาก validation (สุ่ม 10 % ของเฟรม) จึงมองโลกในแง่ดี — ใช้หยุดเทรนเท่านั้น
#  4. ห้ามเพิ่มแขนหรือเปลี่ยนกติกาเลือกหลังเห็นผล (§5.1i) — ถ้าจำเป็นต้องลงทะเบียนใหม่
#  5. smoke test เขียนลงโฟลเดอร์ _SMOKE_TEST แยก — ห้ามเอาผลไปรายงาน
# ─────────────────────────────────────────────────────────────────────────
