"""อ่าน serial แล้วพิมพ์เฉพาะบรรทัด VOTE, และ P, (กรองข้อมูลดิบ F,/S,/G, ทิ้ง)"""
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
            if line.startswith("VOTE") or line.startswith("P,") or line.startswith("LAT"):
                print(line)
    except KeyboardInterrupt:
        print("\nหยุดแล้ว")
    finally:
        ser.close()

if __name__ == "__main__":
    main()