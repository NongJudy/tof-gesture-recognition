# Phase 1 — Cold-start drift, 4x4 mode (H7 test)

## Configuration
- Date: 2026-09-11, capture 17:23:38 -> 23:23:48
- Mode: 4x4 @ 60 Hz nominal, USE_INT=1, TIMING_MODE=1
- Clock: HSE 8 MHz from ST-LINK MCO, RCC_HSE_BYPASS, PLLM=8
- Logic analyzer: 1 MHz sample rate, 10 s per round
- Calibration applied: -95.408 ppm (SD 0.349, n=3)
- n = 121 rounds, 3 min apart, 360 min total
- Board cold-started: USB disconnected prior to run (duration not logged — see limitation)
- Ambient temperature: NOT MEASURED (limitation)
- Data: `data/phase1_drift_v2/` (121 x .bin, manifest.csv, drift_results.csv)

## H7 (pre-registered, written before the run, unmodified)
- Predicted: monotonic exponential rise to plateau
- Start 6000-6200 ppm -> end 6850-6950 ppm
- Total change +700 to +950 ppm
- Time constant 20-40 min

## Result: H7 REJECTED on all five counts

| Prediction | Predicted | Observed | Verdict |
|---|---|---|---|
| Start value | 6000-6200 | 7139.8 | FAIL (+940 above) |
| End value | 6850-6950 | 6985.2 | FAIL (+35.2 above) |
| Total change | +700 to +950 | -154.6 | FAIL (opposite sign) |
| Shape | monotonic rise | 3-phase | FAIL |
| Time constant | 20-40 min | 56.7 +- 4.3 min | FAIL |

## Observed shape — three phases, not monotonic
- Rise:      7139.8 -> 7476.8 ppm  (t = 0 -> 18 min,    +337.0 ppm)
- Decay:     7476.8 -> 6884.1 ppm  (t = 18 -> 147 min,  -592.7 ppm)
- Slow rise: 6884.1 -> 6985.2 ppm  (t = 147 -> 360 min, +101.1 ppm)

Peak: 7476.8 ppm at t = 18 min
Trough: 6884.1 ppm at t = 147 min
Final: 6985.2 ppm at t = 360 min

## Single-exponential fit fails
Model: ppm(t) = A - B*exp(-t/tau)
- A = 6918.4 +- 7.5 ppm
- tau = 56.7 +- 4.3 min
- R^2 = 0.8785, RMS residual = 51.0 ppm
- Residual runs test: runs = 12, expected = 61.4, z = -9.03, p < 1e-18
- RMS residual (51.0 ppm) is 18x the per-point measurement uncertainty (+-2.8 ppm)

CONCLUSION: residuals are structured, not random. A single exponential does not
describe this process. At least two processes act in opposite directions.

## Late-window stability (t = 240-360 min)
- Mean 6946.8 +- 19.3 ppm (n = 41)
- Slope +2.67 ppm/hr, p = 0.607 -> statistically stationary

## Unexplained discrepancy — MUST NOT be explained away
Start value this run: 7139.8 ppm
Start value 2026-09-10 run (same 4x4 mode, also claimed cold start): 6038 ppm
Difference: ~1100 ppm

Untested hypotheses (none verified, do not cite as cause):
1. Shorter power-off duration this run -> board not equally cold
2. Different ambient temperature (never measured in either run)
3. Unknown factor

This discrepancy is the strongest evidence so far that temperature matters and
must be instrumented in any follow-up.

## Reproducibility
- Capture command:
  `python ..\..\pc_tools\drift_run.py capture --outdir phase1_drift_v2 --interval-s 180 --duration-min 360`
- Analyse command:
  `python ..\..\pc_tools\drift_run.py analyse --outdir phase1_drift_v2`
- Firmware: my_tof.h with MY_TOF_USE_4X4 = 1, built and flashed 2026-09-11 before run
- Probe wiring: same 4-wire set as 2026-09-10, untouched throughout the run
