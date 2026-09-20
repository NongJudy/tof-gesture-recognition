"""เตรียมไฟล์ validate_input.npy / validate_output.npy
สำหรับใช้ใน X-CUBE-AI "Validate on desktop"
ใช้ st_dataset.py ตัวจริงจาก pc_tools/ (ไม่เดา signature แล้ว)"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# เพิ่ม pc_tools/ เข้า path เพื่อ import st_dataset ได้
REPO_ROOT = Path(r"C:\Users\gaming\Documents\GitHub\tof-gesture-recognition")
sys.path.insert(0, str(REPO_ROOT / "pc_tools"))

from st_dataset import HAND_POSTURE_DICT, load_dataset  # noqa: E402

DATA_DIR = r"C:\ds\ST_VL53L8CX_handposture_dataset"
MODEL_PATH = REPO_ROOT / "results_phase2_final" / "production_model.keras"
OUT_DIR = REPO_ROOT / "results_phase2_final"
N_SAMPLES = 50
SEED = 42

# โหลดข้อมูลจริง — ตัด None ออก เพราะโมเดล production เทรนด้วย 7 คลาส (--drop-none)
class_names = [c for c in HAND_POSTURE_DICT if c != "None"]
data = load_dataset(DATA_DIR, class_names=class_names, verbose=True)

# สุ่มตัวอย่างแบบทำซ้ำได้ (seed คงที่)
rng = np.random.default_rng(seed=SEED)
idx = rng.choice(len(data), size=N_SAMPLES, replace=False)
X_sample = data.X[idx].astype(np.float32)

# ใช้โมเดล Keras ต้นฉบับสร้าง "คำตอบอ้างอิง" มาเทียบกับ C code ทีหลัง
model = tf.keras.models.load_model(MODEL_PATH)
y_pred = model.predict(X_sample, verbose=0)

np.save(OUT_DIR / "validate_input.npy", X_sample)
np.save(OUT_DIR / "validate_output.npy", y_pred)

print(f"บันทึก {N_SAMPLES} ตัวอย่าง")
print(f"  input  shape: {X_sample.shape}  -> {OUT_DIR / 'validate_input.npy'}")
print(f"  output shape: {y_pred.shape}  -> {OUT_DIR / 'validate_output.npy'}")