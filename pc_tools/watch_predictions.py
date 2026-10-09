"""อ่าน serial แล้วพิมพ์เฉพาะบรรทัด P, (ผลทำนายล่าสุด)
บอร์ดพิมพ์ P, แค่ตอนผลเปลี่ยนเท่านั้น ดังนั้นถ้าไม่มีอะไรเปลี่ยน
จอนี้จะไม่พิมพ์อะไรใหม่ — บรรทัดล่าสุดที่เห็นค้างอยู่ก็คือสถานะปัจจุบัน"""
import argparse
import serial

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=460800)
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"เปิด {args.port} @ {args.baud} — Ctrl+C เพื่อหยุด")
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line.startswith("P,"):
                parts = line.split(",")
                gesture = parts[2] if len(parts) >= 3 else "?"
                print(gesture)
    except KeyboardInterrupt:
        print("\nหยุดแล้ว")
    finally:
        ser.close()

if __name__ == "__main__":
    main()