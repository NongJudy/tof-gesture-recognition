"""quarantine_takes.py — ย้ายรอบที่เก็บผิด (เช่น ทำท่าผิด) ออกจากชุดข้อมูลแบบมีบันทึก

ทำไมต้อง "ย้าย" ไม่ "ลบ":
    ต้องมีหลักฐานว่าตัดอะไรออก เมื่อไหร่ เพราะอะไร (ตรวจย้อนได้ตอนสอบ)
    โค้ดเทรนอ่านเฉพาะ <out-dir>/<Class>/*/npz/*.npz → ย้ายออกไปโฟลเดอร์อื่นแล้วจะไม่ถูกใช้

ค่าเริ่มต้นคือ "ดูเฉยๆ" (dry run) — จะย้ายจริงต้องใส่ --apply

วิธีใช้ (cmd บน Windows):
    1) ดูก่อนว่าจะย้ายอะไร
       python pc_tools\\quarantine_takes.py --src data\\phase4_extra --user 6 --since 20261015_000000
              --reason "wrong posture in one take, take unknown"
    2) ถ้ารายการถูกต้อง ใส่ --apply ต่อท้ายคำสั่งเดิม

ผลลัพธ์:
    <dst>/<Class>/log__...__User<N>__take<k>/   (โฟลเดอร์รอบที่ถูกย้าย)
    <dst>/quarantine_log.csv                     (บันทึกทุกรอบที่ย้าย)

จุดที่อาจพัง:
    - --since/--until อ่านเวลาจากชื่อโฟลเดอร์ (log__YYYYmmdd_HHMMSS__...) ถ้าชื่อผิดรูปแบบจะถูกข้ามและแจ้งเตือน
    - ปลายทางมีโฟลเดอร์ชื่อซ้ำ → หยุด ไม่เขียนทับ
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ชื่อโฟลเดอร์รอบ: log__20261015_000512__User6__take1
LOG_RE = re.compile(r"^log__(\d{8}_\d{6})__User(\d+)__take(\d+)$")
LOG_COLS = ["moved_at", "class", "user", "take", "recorded_at", "src", "dst", "reason"]


def find_takes(src: Path, user: int, since: str | None, until: str | None) -> list[tuple[str, Path, dict]]:
    """หาโฟลเดอร์รอบของผู้ทดลองที่ระบุ ภายในช่วงเวลา (ถ้าระบุ)"""
    found: list[tuple[str, Path, dict]] = []
    for cls_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        for log_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            m = LOG_RE.match(log_dir.name)
            if not m:
                if "__User" in log_dir.name:
                    print(f"  [WARN] skipped (name not recognised): {log_dir}")
                continue
            stamp, u, take = m.group(1), int(m.group(2)), int(m.group(3))
            if u != user:
                continue
            # เทียบเป็นสตริงได้เพราะรูปแบบ YYYYmmdd_HHMMSS เรียงตามเวลาอยู่แล้ว
            if since and stamp < since:
                continue
            if until and stamp > until:
                continue
            found.append((cls_dir.name, log_dir, {"stamp": stamp, "user": u, "take": take}))
    return found


def check_stamp(value: str | None, name: str) -> None:
    """ตรวจรูปแบบเวลา YYYYmmdd_HHMMSS"""
    if value is None:
        return
    try:
        datetime.strptime(value, "%Y%m%d_%H%M%S")
    except ValueError:
        raise SystemExit(f"--{name} must look like 20261015_000000 (got {value!r})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help=r"เช่น data\phase4_extra")
    ap.add_argument("--dst", type=Path, default=None, help=r"ค่าเริ่มต้น: data\phase4_rejected")
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--since", default=None, help="เวลาเริ่ม (รวม) เช่น 20261015_000000")
    ap.add_argument("--until", default=None, help="เวลาสิ้นสุด (รวม) เช่น 20261015_235959")
    ap.add_argument("--reason", required=True, help="เหตุผล (บันทึกลง quarantine_log.csv)")
    ap.add_argument("--apply", action="store_true", help="ย้ายจริง (ไม่ใส่ = ดูเฉยๆ)")
    args = ap.parse_args()

    check_stamp(args.since, "since")
    check_stamp(args.until, "until")
    if not args.src.is_dir():
        raise SystemExit(f"Not a folder: {args.src}")
    dst_root = args.dst or (args.src.parent / "phase4_rejected")
    if dst_root.resolve() == args.src.resolve():
        raise SystemExit("--dst must differ from --src")

    takes = find_takes(args.src, args.user, args.since, args.until)
    print(f"\n  Source      : {args.src}")
    print(f"  Destination : {dst_root}")
    print(f"  User        : {args.user}   since={args.since}   until={args.until}")
    print(f"  Reason      : {args.reason}")
    print(f"  Mode        : {'APPLY (moving)' if args.apply else 'DRY RUN (nothing moved)'}\n")
    for cls, path, meta in takes:
        print(f"  {cls:<11} take{meta['take']}  {meta['stamp']}  {path.name}")
    print(f"\n  Total: {len(takes)} takes")

    if not takes:
        return
    # กันเขียนทับ: เช็คทั้งหมดก่อนย้ายสักอัน
    clashes = [dst_root / cls / p.name for cls, p, _ in takes if (dst_root / cls / p.name).exists()]
    if clashes:
        raise SystemExit(f"Destination already has: {clashes[0]} — stopped, nothing moved")
    if not args.apply:
        print("  Dry run only. Add --apply to move these takes.")
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    log_path = dst_root / "quarantine_log.csv"
    new_log = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new_log:
            w.writeheader()
        for cls, path, meta in takes:
            target = dst_root / cls / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            w.writerow({"moved_at": datetime.now().isoformat(timespec="seconds"), "class": cls,
                        "user": meta["user"], "take": meta["take"], "recorded_at": meta["stamp"],
                        "src": str(path), "dst": str(target), "reason": args.reason})
    print(f"  Moved {len(takes)} takes. Log: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
