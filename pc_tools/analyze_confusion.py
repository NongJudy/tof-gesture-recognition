"""อ่าน confusion matrix จาก phase2_results.json ที่เทรนไปแล้ว แสดงเป็นตาราง
มีชื่อคลาส อ่านง่าย ไม่ต้องเทรนใหม่

วิธีรัน:
    python analyze_confusion.py --results C:\\...\\results_phase2\\phase2_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = ["Fist", "FlatHand", "Dislike", "Like", "Love", "CrossHands", "BreakTime"]


def print_confusion(cm: list, title: str) -> None:
    cm = np.array(cm)
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    header = "actual\\pred".ljust(12) + "".join(c[:8].rjust(9) for c in CLASSES)
    print(header)
    for i, row in enumerate(cm):
        line = CLASSES[i][:11].ljust(12) + "".join(str(int(v)).rjust(9) for v in row)
        print(line)

    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    print("\nสัดส่วนถูกต้องต่อคลาส (แถวในตารางนี้คือ 'ของจริงเป็นคลาสนี้ ทายถูกกี่ %'):")
    for c, a in zip(CLASSES, per_class_acc):
        print(f"  {c:12s}: {a*100:5.1f}%")

    # หาคู่ที่สับสนกันมากที่สุด (นอกแนวทแยง)
    off_diag = cm.copy().astype(float)
    np.fill_diagonal(off_diag, 0)
    flat_idx = np.argsort(off_diag, axis=None)[::-1]
    print("\nคู่ที่สับสนกันมากที่สุด (จริง -> ทายผิดเป็น):")
    shown = 0
    for idx in flat_idx:
        i, j = np.unravel_index(idx, off_diag.shape)
        if off_diag[i, j] <= 0 or shown >= 5:
            break
        print(f"  {CLASSES[i]:12s} -> ทายเป็น {CLASSES[j]:12s} : {int(off_diag[i, j])} เฟรม")
        shown += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        d = json.load(f)

    if "random_split" in d:
        rs = d["random_split"]
        cm = rs.get("sample_confusion_matrix", rs.get("confusion_matrix"))
        print_confusion(
            cm, f"RANDOM SPLIT (ตัวอย่างจาก seed แรก จาก {rs.get('n_seeds', 1)} "
                f"seeds, mean acc={rs.get('mean_accuracy', rs.get('test_accuracy')):.4f})"
        )

    if "subject_independent" in d:
        for r in d["subject_independent"]["fold_results"]:
            cm = r.get("sample_confusion_matrix", r.get("confusion_matrix"))
            mean_acc = r.get("mean_accuracy", r.get("test_accuracy"))
            sd_acc = r.get("sd_accuracy", 0.0)
            title = (f"SUBJECT-INDEPENDENT — held out user{r['held_out_user']} "
                     f"(mean acc={mean_acc:.4f} ± {sd_acc:.4f}, "
                     f"ตาราง confusion เป็นตัวอย่างจาก seed แรกเท่านั้น)")
            print_confusion(cm, title)


if __name__ == "__main__":
    main()
