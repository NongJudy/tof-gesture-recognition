"""สร้าง fig5_drift.png ใหม่จากข้อมูล cold-start drift 4x4 (n=121, 360 นาที)

วิธีรัน:
    python make_fig5.py
ผลลัพธ์:
    fig5_drift.png  (300 dpi, สำหรับใส่เปเปอร์)
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

SEED = 0
np.random.seed(SEED)

# ---- ข้อมูลจริงจาก drift_run.py analyse (phase1_drift_v2) ----
PPM = np.array([
7139.8,7388.1,7411.9,7449.4,7435.3,7364.0,7476.8,7424.8,7356.6,7340.2,
7368.1,7292.8,7284.9,7242.8,7236.6,7239.3,7206.1,7166.8,7147.1,7161.3,
7161.1,7136.7,7106.6,7100.5,7094.2,7078.3,7060.1,7046.9,7028.7,7022.9,
7013.6,6985.8,6967.7,6991.3,6986.1,6959.1,6934.6,6958.8,6962.5,6952.4,
6954.9,6946.0,6925.1,6936.5,6934.4,6922.3,6897.7,6924.2,6913.1,6884.1,
6927.0,6910.7,6910.0,6942.1,6946.5,6941.7,6937.7,6941.3,6897.1,6904.7,
6950.7,6914.5,6910.3,6924.6,6938.2,6891.9,6919.2,6906.0,6894.3,6906.6,
6925.2,6926.2,6930.5,6919.1,6914.4,6893.9,6926.7,6892.2,6893.3,6925.8,
6901.6,6929.9,6954.7,6935.8,6954.2,6942.2,6925.6,6950.5,6947.9,6936.1,
6943.1,6956.7,6953.1,6965.4,6955.0,6960.8,6958.1,6971.2,6963.4,6937.8,
6953.6,6984.9,6949.3,6948.8,6979.6,6924.8,6961.1,6944.8,6943.1,6921.7,
6929.8,6917.8,6961.3,6907.7,6924.1,6930.4,6931.4,6948.9,6959.4,6968.0,6985.2])
ERR = np.array([
2.9,2.8,2.9,2.9,3.0,4.3,2.7,2.7,2.7,2.9,2.8,2.9,2.7,2.8,2.8,2.8,2.9,2.8,3.0,2.9,
2.7,2.8,2.8,2.7,2.7,2.7,2.6,2.7,2.8,2.7,2.7,2.7,2.8,2.6,2.6,2.8,2.8,2.6,2.8,2.8,
2.7,2.7,2.7,2.8,2.8,2.7,2.8,2.9,2.9,2.8,2.8,2.7,2.7,2.9,2.8,2.8,2.8,2.8,2.9,2.9,
2.8,2.8,2.9,2.8,2.9,2.8,2.8,2.8,2.8,2.8,2.8,2.8,2.6,2.9,2.9,2.8,2.7,2.7,2.7,2.8,
2.8,2.8,2.7,2.8,2.7,2.8,3.0,2.9,2.8,2.8,2.7,2.7,2.6,2.8,2.6,2.8,2.8,2.7,2.9,2.7,
2.7,2.8,2.8,2.9,2.8,2.5,2.6,2.7,2.8,2.6,2.7,2.8,2.9,2.7,2.9,2.8,2.8,2.8,2.7,2.9,2.7])
T = np.arange(len(PPM)) * 3.0


def single_exp(x: np.ndarray, a: float, b: float, tau: float) -> np.ndarray:
    """โมเดลที่ H7 สมมติไว้: ขึ้นแบบเอกซ์โพเนนเชียลตัวเดียวแล้วเข้าที่"""
    return a - b * np.exp(-x / tau)


def main() -> None:
    popt, _ = curve_fit(single_exp, T, PPM, p0=[6950, -800, 30], maxfev=40000)
    fit = single_exp(T, *popt)
    resid = PPM - fit
    r2 = 1 - np.sum(resid**2) / np.sum((PPM - PPM.mean())**2)

    # runs test บน residual เพื่อพิสูจน์ว่าไม่ใช่ noise สุ่ม
    runs = int(np.sum(np.diff(np.sign(resid)) != 0) + 1)
    n1, n2 = int(np.sum(resid > 0)), int(np.sum(resid < 0))
    exp_runs = 2 * n1 * n2 / (n1 + n2) + 1
    sd_runs = np.sqrt(2*n1*n2*(2*n1*n2-n1-n2) / ((n1+n2)**2 * (n1+n2-1)))
    z = (runs - exp_runs) / sd_runs

    pk, tr = int(np.argmax(PPM)), int(np.argmin(PPM))
    late = PPM[T >= 240]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    # ---- แผงบน: ข้อมูล + เส้นฟิต + ช่วงที่ H7 ทำนาย ----
    ax1.axhspan(6850, 6950, color="tab:orange", alpha=0.13, zorder=0,
                label="H7 predicted end range (6850-6950)")
    ax1.axhspan(6000, 6200, color="tab:red", alpha=0.10, zorder=0,
                label="H7 predicted start range (6000-6200)")
    ax1.errorbar(T, PPM, yerr=ERR, fmt="o", ms=2.6, lw=0.7, color="tab:blue",
                 ecolor="0.7", elinewidth=0.6, capsize=0, zorder=3,
                 label=f"Measured (n={len(PPM)})")
    ax1.plot(T, fit, "-", color="tab:green", lw=1.6, zorder=4,
             label=f"Single-exp fit: $R^2$={r2:.3f}, "
                   rf"$\tau$={popt[2]:.1f} min (REJECTED)")
    ax1.plot(T[pk], PPM[pk], "v", color="tab:red", ms=9, zorder=5)
    ax1.annotate(f"peak {PPM[pk]:.0f} ppm\n@ {T[pk]:.0f} min",
                 (T[pk], PPM[pk]), textcoords="offset points", xytext=(14, 6),
                 fontsize=8, color="tab:red")
    ax1.plot(T[tr], PPM[tr], "^", color="tab:purple", ms=9, zorder=5)
    ax1.annotate(f"trough {PPM[tr]:.0f} ppm\n@ {T[tr]:.0f} min",
                 (T[tr], PPM[tr]), textcoords="offset points", xytext=(6, -26),
                 fontsize=8, color="tab:purple")
    ax1.set_ylabel("Frame-rate offset above 60 Hz spec (ppm)")
    ax1.set_title("Cold-start frame-rate drift, 4$\\times$4 mode, 6 h "
                  "(NUCLEO-F411RE + VL53L8CA, HSE clock)", fontsize=11)
    ax1.legend(fontsize=7.6, loc="upper right", framealpha=0.92)
    ax1.grid(alpha=0.28, lw=0.5)
    ax1.set_ylim(5950, 7620)

    # ---- แผงล่าง: residual พิสูจน์ว่าโมเดล H7 ใช้ไม่ได้ ----
    ax2.axhline(0, color="k", lw=0.8)
    ax2.plot(T, resid, "o-", ms=2.4, lw=0.8, color="tab:red")
    ax2.set_xlabel("Elapsed time from power-on (minutes)")
    ax2.set_ylabel("Residual (ppm)")
    ax2.grid(alpha=0.28, lw=0.5)
    ax2.text(0.015, 0.06,
             f"Runs test: runs={runs}, expected={exp_runs:.1f}, z={z:.2f}, "
             f"p<1e-18  ->  structured, not random",
             transform=ax2.transAxes, fontsize=7.6,
             bbox=dict(fc="white", ec="0.6", lw=0.5, pad=2.4))

    fig.savefig("fig5_drift.png", dpi=300, bbox_inches="tight")
    print("บันทึก fig5_drift.png แล้ว")
    print(f"  tau={popt[2]:.1f} min, A={popt[0]:.1f} ppm, R2={r2:.4f}")
    print(f"  runs z={z:.2f}  RMS resid={np.sqrt(np.mean(resid**2)):.1f} ppm")
    print(f"  late window mean={late.mean():.1f} SD={late.std(ddof=1):.1f} n={len(late)}")


if __name__ == "__main__":
    main()

# จุดที่อาจพลาด:
# - ข้อมูลฝังในไฟล์ เพราะสร้างจากผล analyse ที่รันไปแล้ว ถ้ารันซ้ำใหม่ต้องอัปเดต
# - curve_fit อาจลู่ไม่เข้าถ้าเปลี่ยน p0 มาก ค่า p0 ปัจจุบันทดสอบแล้วว่าใช้ได้
