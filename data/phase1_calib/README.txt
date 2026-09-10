ข้อมูลสอบเทียบนาฬิกา logic analyzer
เก็บวันที่ 9 ก.ย. 2026

เงื่อนไข:
  บอร์ด STM32F411RE ปล่อยคลื่น 1000.000000 Hz (TIM3 CH3 ขา PC8)
  MY_CAL_ENABLE=1  MY_CAL_FREQ_HZ=1000U  PSC=0  ARR=41999
  ไม่ได้เสียบบอร์ดขยาย ToF
  สาย: CH0 -> CN10 ขา 2 (PC8), GND -> CN10 ขา 9

คำสั่งที่ใช้เก็บ:
  sigrok-cli -d fx2lafw --config samplerate=1m --samples 10000000
             -O binary -o cal1k_runN.bin

คำสั่งวิเคราะห์:
  python analyze_calib.py cal1k_run1.bin cal1k_run2.bin cal1k_run3.bin

ผล (n=3):
  รอบ 1  999.905071 Hz  -94.929 ppm
  รอบ 2  999.904371 Hz  -95.629 ppm
  รอบ 3  999.904271 Hz  -95.729 ppm
  เฉลี่ย -95.429 ppm  SD 0.356 ppm
  duty 50.000% ทุกรอบ  ไม่มีคาบผิดปกติ

สรุป: นาฬิกาบอร์ดกับ logic analyzer ตรงกันภายใน 95 ppm
      ห่างจากค่าที่ต้องแยก (7000 ppm) 74 เท่า
      ใช้เป็นนาฬิกาอ้างอิงอิสระได้