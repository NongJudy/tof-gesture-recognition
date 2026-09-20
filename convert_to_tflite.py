"""แปลง production_model.keras -> .tflite
เพื่อเลี่ยงปัญหา Keras 3.x เข้ากันไม่ได้กับ X-CUBE-AI parser"""
import tensorflow as tf

MODEL_PATH = r"C:\Users\gaming\Documents\GitHub\tof-gesture-recognition\results_phase2_final\production_model.keras"
OUTPUT_PATH = r"C:\Users\gaming\Documents\GitHub\tof-gesture-recognition\results_phase2_final\production_model.tflite"

model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"saved: {OUTPUT_PATH} ({len(tflite_model)} bytes)")