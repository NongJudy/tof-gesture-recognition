# PROJECT_FACTS.md

**Verified facts for: Hand Gesture Recognition using Time-of-Flight Sensor**
Student: Chin · Board: NUCLEO-F411RE + X-NUCLEO-53L8A1 · Repo: github.com/NongJudy/tof-gesture-recognition

---

## HOW TO USE THIS FILE — read this first, every session

This file exists because the assistant repeatedly stated facts from memory that
turned out to be wrong, including telling the student their advisor was wrong when
the advisor was right.

**Rules:**

1. Before stating any number, spec, or "this does not exist" claim about this
   project — check this file first. If it is here, quote it. Do not re-derive.
2. If a fact is NOT in this file, say `[ยังไม่ได้ตรวจ]` and go verify it.
   Do not answer from memory and present it as checked.
3. Never tell the student that they, or their advisor, are wrong about a spec
   unless the correction is in this file with a source, or you open the source
   document in the current turn.
4. When a new fact is verified, tell the student to add it here.
5. Every entry must have a source. No source = not a fact = do not state it.

**Tags used below:**
`[DS]` datasheet DS14161 · `[UM3109]` ULD driver guide · `[UM3120]` shield manual
`[CODE]` source file in repo · `[MEASURED]` measured by us · `[WEB]` external source
`[PAPER]` paper in Project Knowledge

---

# 1. HARDWARE SPECIFICATIONS

## 1.1 Sensor — VL53L8CX / VL53L8CA

| Fact | Value | Source |
|---|---|---|
| Chip actually on our shield | **VL53L8CA** (evaluation part, not the mass-production part) | `[UM3120]` |
| VL53L8CA relationship | ST support states it is a "super product" covering both VL53L8CX and VL53L8CH, and does support histograms | `[WEB]` ST community |
| Field of **View** (FoV) | **45° × 45°** | `[DS]` §2.2, p.4 |
| Field of **Illumination** (FoI) | **43.4° × 43.4°** at 75% of max signal · **57.9° × 57.9°** at 10% | `[DS]` §2.3, p.5 |
| Zones | 8×8 = 64, or 4×4 = 16 (software selectable) | `[DS]` |
| Range | 2 cm to 400 cm per zone | `[DS]` Table 1 |
| VCSEL wavelength | 940 nm, Class 1 eye safe | `[DS]` |
| Package | Optical LGA16, 6.4 × 3.0 × 1.75 mm | `[DS]` Table 1 |
| Max ranging freq 8×8 | **15 Hz** | `[UM3109]` Table 2 |
| Max ranging freq 4×4 | **60 Hz** | `[UM3109]` Table 2 |
| I2C max clock (sensor side) | **1 MHz** | `[DS]` |
| SPI max clock (sensor side) | **3 MHz** | `[DS]` |

### ⚠️ TRAP #1 — FoV vs FoI are DIFFERENT quantities

**45° and 43.4° are both correct. They measure different things.**

- FoV = the angle the **receiver** sees
- FoI = the angle the **VCSEL emitter** illuminates, defined at a signal threshold

The assistant previously told the student "43.4° is wrong, it is 45°" and implied
the advisor had made an error. **That was the assistant's error.** The advisor was
quoting FoI from the datasheet. The ShortRange paper in Project Knowledge also uses
43.4° and explicitly writes "defined at 75% of the maximum signal intensity".

**Never repeat this mistake. Both numbers are in the datasheet.**

### Zone angular size — derived, useful for explaining coverage

```
one zone ≈ 45° / 8 = 5.6°
zone width at distance d ≈ d / 10        (rule of thumb)
at 300 mm → one zone ≈ 30 mm; a 100 mm hand covers ≈ 3 zones
```

## 1.2 MCU — STM32F411RE

| Fact | Value | Source |
|---|---|---|
| Core | Cortex-M4 @ 84 MHz | `[CODE]` |
| **Clock source (from 2 Sep)** | **HSE = 8 MHz MCO from ST-LINK, `RCC_HSE_BYPASS`, PLLM=8** | `[CODE]` `main.c` |
| Clock source (before) | HSI 16 MHz, PLLM=16 — **ran 1.36% slow** | `[MEASURED]` |
| Board revision | **MB1136-F411RE-C04** — C-02 and above wire ST-LINK MCO to OSC_IN | `[WEB]` Mastering STM32 §10.1.3.1.1 |
| Flash / RAM | 512 KB / 128 KB | ST |
| **I2C max clock** | **400 kHz — Fast Mode only** | `[WEB]` |
| Fast Mode Plus (1 MHz) | **NOT available on any STM32F4** (exists on F410/F412/F413/F446, L4, F7) | `[WEB]` |

### ⚠️ TRAP #2 — the I2C ceiling is the MCU's, not the sensor's

The sensor accepts 1 MHz. **Our board cannot produce it.**
If higher bus throughput is ever needed, the options are: reduce payload, or move to SPI.
This is a legitimate, citable limitation to write in the thesis.

### ⚠️ TRAP #11 — timing figures are only as good as the reference clock

Everything measured before 2 Sep used HSI (±1%). A controlled comparison at a
fixed 30 Hz setting gave:

| Clock | Measured | Deviation |
|---|---|---|
| HSI | 29.697 Hz | −1.010% |
| HSE | 30.102 Hz | +0.340% |

Welch t = 43.5. **The old clock ran 1.36% slow, so every pre-2-Sep timing figure
was 1.36% too high.** Superseded values are in §3. Do not quote the old numbers.

## 1.3 Wiring / pins

| Signal | Pin | Note |
|---|---|---|
| I2C SCL / SDA | PB8 / PB9 | 400 kHz |
| Sensor INT | **PA4**, EXTI4, falling edge, NVIC priority 0 | `[CODE]` `app_tof_pin_conf.c` |
| UART | USART2 / PA2, **460800 baud**, COM3 | `[CODE]` |
| I2C address | **0x52, sent directly, NOT shifted** | HAL expects 8-bit format |

## 1.4 Logic analyser — verified on this unit, 4 Sep 2026

Cheap "24MHz 8CH" clone. Everything below was read off **this** unit with
`sigrok-cli --driver fx2lafw --show`, not taken from a web page.

| Property | Value | Source |
|---|---|---|
| Chip | Cypress CY7C68012A / FX2LP | `[WEB]` sigrok wiki |
| USB VID:PID | **0925:3881** — "Lakeview Research Saleae Logic" | `[MEASURED]` Zadig + sigrok-cli |
| Reported as | `Saleae Logic [S/N: Saleae Logic] with 8 channels` | `[MEASURED]` |
| Channels | **D0–D7**, matching the CH0–CH7 silkscreen | `[MEASURED]` |
| Samplerates | 20k 25k 50k 100k 200k 250k 500k · 1M 2M 3M 4M 6M 8M 12M 16M **24M 48M** | `[MEASURED]` |
| Triggers | **`0 1 r f e`** — level low/high, rising, falling, either | `[MEASURED]` |
| Timebase | **24 MHz crystal** (not a ceramic resonator) | `[WEB]` sigrok: *"All devices use a 24MHz crystal"* |
| Measured crystal error | **≈79 ppm = 0.0079%** on an FX2 device | `[WEB]` fx2adc project |
| Max input voltage | 5.25 V (FX2 hardware manual). Our signals are 3.3 V | `[WEB]` |
| Series resistance | 100 Ω per channel; some clones have a 74HC245 buffer, some do not | `[WEB]` |

**Why the crystal matters:** TRAP #12 needs a 0.7% discrepancy resolved. A crystal at
79 ppm is ~89× finer than that, so this instrument *can* settle it. A ceramic resonator
(±0.5%) could not have. This was the single most important thing to check before
trusting any number from it.

**Why 12 MHz and 1 MHz were chosen:** every listed rate is an exact integer division of
48 MHz (12 = 48/4, 1 = 48/48), so no fractional-divider error is added on top of the
crystal. 10 MHz and 20 MHz are **not** offered — do not assume an arbitrary rate exists.

**48 MHz is listed but must not be used.** It implies 48 MB/s over USB 2.0, well past
what the link sustains; it will drop samples. Note that the sigrok pages consulted
beforehand listed only up to 24 MHz — the extra rate appeared only on the real unit,
which is the argument for checking hardware rather than trusting a wiki table.

### ⚠️ TRAP #19 — one program at a time

`LIBUSB_ERROR_ACCESS` while PulseView is open. USB devices are claimed exclusively:
whichever program grabs it first holds it.

> Before blaming drivers, cables or the device: **close the other program.**
> This cost time once already; it will look like a driver fault every time.

### Windows install sequence that actually worked

1. PulseView **and** sigrok-cli from sigrok.org, **Nightly** build.
   Nightly, not Release — the tri-state fix (fx2lafw 0.1.2, 2014) matters: older
   firmware left the data pins **driven** rather than tri-stated after a capture,
   which can contend with an MCU pin driving the same net.
2. Run **Zadig (PulseView)** as administrator, `Options → List All Devices`,
   select `Unknown Device #1`, **confirm USB ID reads 0925:3881**, install **WinUSB**.
   The list also contains the mouse, keyboard, Bluetooth and camera — selecting the
   wrong row breaks that device immediately.
3. Run `sigrok-cli --scan`. This uploads the firmware, after which the device
   **renames itself** from `Unknown Device #1` to `Saleae Logic`. PulseView only sees
   it after this step — scanning in PulseView before it will fail, which looks like
   a failed driver install but is not.
4. `sigrok-cli --scan -l 5` is the diagnostic to reach for: PulseView fails silently,
   sigrok-cli prints the actual libusb error.

---

# 2. SOFTWARE / DRIVER FACTS

## 2.1 ⚠️ TRAP #3 — the BSP REMAPS target_status

```c
/* vl53l8cx.c line 781 — vl53l8cx_map_target_status() */
if (status == 5 || status == 9)  return 0;    /* ranging OK   */
else if (status == 0)            return 255;  /* no new data  */
else                             return status;
```

| Meaning | Raw value (UM3109 Table 4, and ST's dataset) | What OUR firmware receives |
|---|---|---|
| Valid, 100% confidence | **5** | **0** |
| Valid, 50% confidence | 9 | 0 |
| No update | 0 | 255 |

**Consequences — do not forget these:**

1. In our logs, `status == 0` means **GOOD**, not bad. A log showing "100% status 0"
   means 100% of readings are valid.
2. ST's `ST_VL53L8CX_handposture_dataset` stores **raw** status (5 = valid).
   When we record our own dataset for compatibility with ST's scripts we must
   convert back: `raw = (bsp == 0) ? 5 : bsp`.
3. Writing `if (status != 5) discard;` on BSP output would discard **100% of good data**.

## 2.2 `EnableSignal` vs the DISABLE macros — they do different things

| Mechanism | Effect | Where |
|---|---|---|
| `Profile.EnableSignal = 1` | BSP copies `signal_per_spad` into `Result.ZoneResult[i].Signal[j]`. **Does not change I2C payload.** | `vl53l8cx.c:340, :760` |
| `#define VL53L8CX_DISABLE_*` | ULD does not request that output block from the sensor → **fewer bytes on the wire** | `vl53l8cx_api.c:555-615` |

**Both are needed.** Setting only one does not achieve the goal.

Macro placement: define them in `Drivers/BSP/Components/vl53l8cx/porting/platform.h`.
`vl53l8cx_api.h` includes `platform.h` at line 22 and tests the `#ifndef` guards at
line 189+, so definitions there are seen in time. `[CODE]` verified.

Available macros:
```
VL53L8CX_DISABLE_AMBIENT_PER_SPAD      VL53L8CX_DISABLE_RANGE_SIGMA_MM
VL53L8CX_DISABLE_NB_SPADS_ENABLED      VL53L8CX_DISABLE_DISTANCE_MM
VL53L8CX_DISABLE_NB_TARGET_DETECTED    VL53L8CX_DISABLE_REFLECTANCE_PERCENT
VL53L8CX_DISABLE_SIGNAL_PER_SPAD       VL53L8CX_DISABLE_TARGET_STATUS
VL53L8CX_DISABLE_MOTION_INDICATOR
```

**Must stay enabled for this project:** `DISTANCE_MM`, `SIGNAL_PER_SPAD`,
`TARGET_STATUS`, `NB_TARGET_DETECTED`.
Reason: ST's model input is `(8, 8, 2)` = distance + signal_per_spad; status is needed
for filtering; nb_target_detected is required alongside per ST guidance.

## 2.3 Other verified driver details

| Fact | Detail | Source |
|---|---|---|
| `VL53L8CX_SwapBuffer()` returns | `void`, not a status code | `platform.h:143` |
| I2C register index width | `I2C_MEMADD_SIZE_16BIT` | `[DS]` §4 |
| Transfer chunking | 512 bytes max per HAL call — ULD passes `uint32_t`, HAL takes `uint16_t`; without chunking this overflows silently | `[CODE]` `my_platform.c` |
| HAL timeout chosen | 100 ms = 8.6× the worst-case 11.6 ms bus time at 400 kHz | `[MEASURED]` |
| ASYNC vs BLOCKING | `RS_MODE_ASYNC_CONTINUOUS` sets poll timeout 0, so `GetDistance` returns immediately | `vl53l8cx.c:438` |
| `ToF_EventDetected` flag | declared `app_tof.c:47`, set in `HAL_GPIO_EXTI_Callback` in `app_tof_pin_conf.c:31` | `[CODE]` |

---

# 3. OUR OWN MEASUREMENTS

All measured with DWT cycle counter (11.9 ns resolution) on the actual board.
Method note: Boner et al. 2022 (ETH Zürich) use the same DWT technique — citable.

## 3.1 The measurement matrix — 8 conditions, one variable at a time

**All eight rows use the HSE crystal clock. dup = 0 in every row**, verified
against the sensor's own `streamcount`.
Raw data: `pc_tools/timing_8x8_matrix.csv`. Analysis: `pc_tools/plot_matrix.py`.

| | Config | Fields | Acq. | Path | I2C | Bytes | MCU work | Period | **Rate** | Load | Idle | delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 8×8 | all | polling | bsp | 115.8 | 1444 | 55.49 ms | 66.17 | 15.11 Hz | **83.9%** | 10.68 | 4.0 |
| B | 8×8 | all | **int** | bsp | 4.0 | 1444 | 33.92 | 66.17 | 15.11 | 51.3% | 32.25 | 4.0 |
| C | 8×8 | **slim** | int | bsp | 4.0 | 580 | 14.37 | 66.17 | 15.11 | 21.7% | 51.79 | 4.0 |
| **D** | 8×8 | slim | int | **direct** | **1.0** | 580 | **13.44** | 66.17 | 15.11 | **20.3%** | 52.73 | 4.0 |
| E | 4×4 | all | polling | bsp | 13.5 | 532 | 13.33 | 24.65 | 40.57 | 54.1% | 11.32 | **1.5** |
| F | 4×4 | all | int | bsp | 4.0 | 532 | 13.32 | 24.70 | 40.49 | 53.9% | 11.38 | **1.5** |
| G | 4×4 | **slim** | int | bsp | 4.0 | 196 | 5.63 | **16.55** | **60.42** | 34.0% | 10.92 | 1.0 |
| **H** | 4×4 | slim | int | **direct** | **1.0** | 196 | **4.69** | 16.55 | **60.42** | **28.3%** | 11.86 | 1.0 |

### Per-step attribution — and it differs by mode

| Step | Change | 8×8 | 4×4 |
|---|---|---|---|
| 1st | polling → INT | **−21.57 ms**, rate unchanged | −0.01 ms, rate unchanged |
| 2nd | disable unused output fields | **−19.55 ms**, rate unchanged | **−7.69 ms, rate 40.5 → 60.4 Hz** |
| 3rd | bypass redundant BSP I2C | −0.94 ms | −0.94 ms |
| **total** | | **−75.8% work, idle ×4.9** | **−64.8% work, rate +49%** |

### ⚠️ TRAP #15 — the same change does different things in different modes

The assistant predicted that INT would lift 4×4 to 60 Hz. **It did not** — E→F
moved the rate by 0.08 Hz. What lifted the rate was **reducing the payload**.

The mechanism is a deadline, not a proportion:

```
sensor tick = 16.55 ms — the MCU must finish within one tick to catch the next

F: work 13.32 ms, margin 3.23 ms  -> too tight, misses ticks, delta 1.5, 40.5 Hz
G: work  5.63 ms, margin 10.92 ms -> comfortable, catches all, delta 1.0, 60.4 Hz
```

At 8×8 the budget is 66.17 ms (4 ticks), so nothing the MCU does changes the
rate — the improvement shows up purely as headroom. At 4×4 the budget is one
tick, so the same improvement converts into throughput.

**A margin of 3.23 ms was not enough.** Being under the deadline on average is
not sufficient; per-frame jitter means some frames still overrun. Report margin,
not just the mean.

### Headroom for inference (1.46 ms, st_cnn2d on F401 @ 84 MHz)

| | Load with AI | Inferences fitting in idle |
|---|---|---|
| A | 86.1% | 7 |
| **D** | **22.5%** | **36** |

## 3.1b Frame rate vs. datasheet — both modes at the ceiling

| Mode | UM3109 spec | Measured (HSE) | Ratio | delta | Ticks missed |
|---|---|---|---|---|---|
| 8×8 | 15 Hz | **15.11 Hz** | 100.7% | 4.0 exact | **none** |
| 4×4 | 60 Hz | **60.42 Hz** | 100.7% | 1.0 exact | **none** |

`delta` is exact and `dup = 0`, so the system reads **every** frame the sensor
produces in both modes. There is no remaining rate to recover.

What is left to optimise is negligible: of D/H's remaining MCU work, the I2C
payload read is 96% (4.51 of 4.69 ms at 4×4) and is bounded by the 400 kHz bus.
Eliminating everything else would save 0.18 ms and change no rate.

### ⚠️ TRAP #12 — do not write "we exceeded the datasheet"

The datasheet says *"up to"* — a configuration ceiling, not a guaranteed exact
rate. Both modes overshoot by the **same** +0.7%, so it is systematic, not noise.

Two candidates, **not yet distinguished**:
- the sensor's internal RC oscillator runs ~0.7% fast (unremarkable for an RC), or
- the board's HSE reference is off (would need 8.056 MHz; a crystal is ±0.005%,
  ~140× too small to explain it)

The second is implausible, so the sensor is the likely source — **but proving it
needs a third, independent clock.** The logic analyser has one. Until then,
report the deviation; do not attribute it.

### The sensor has one internal tick, ~60.4 Hz

```
8×8:  15.11 Hz × delta 4  =  60.46 ticks/s
4×4:  60.42 Hz × delta 1  =  60.42 ticks/s
```

Both modes share one timebase, measured independently and agreeing to 0.07%.
**8×8 is not "a slower mode" — it consumes 4 ticks per frame** (4× the zones).
Its 15 Hz is 60.4/4, not an independently specified limit. This also explains the
matched +0.7% overshoot in both modes: it is one clock, seen two ways.

## 3.2 Data equivalence check (advisor task: "make interrupt actually work")

Two captures, same scene, 4 minutes apart, sensor untouched:

| | polling | INT |
|---|---|---|
| Frames | 532 | 530 |
| Dropped frames | **0** | **0** |
| Overall mean | 1246.72 mm | 1248.08 mm |

- Difference **+1.36 mm = 0.1%** — a uniform positive offset across 63/64 zones,
  consistent with thermal drift during the 4 minutes, not noise.
- Zone 60 differed by 15.1 mm. Investigated: zone 60 sits on an object boundary
  (neighbour zone 61 reads 278 mm, zone 59 reads 1972 mm) and is **bimodal**.
  Near-peak occurred 11/532 vs 5/530 → **Fisher exact p = 0.207, not significant**.
  This fully accounts for the 15.1 mm as sampling variation.
- Conclusion: **interrupt-driven acquisition is functionally equivalent to polling.**

## 3.2b Verifying that every counted frame is genuinely new

Bypassing the BSP removed `vl53l8cx_check_data_ready`, which was the only thing
confirming that a read returned *new* data. Replacement check: read the sensor's
own `streamcount` (`vl53l8cx_api.h:277`, "auto-incremented at each range") and
compare consecutive values. Subtraction is done in `uint8_t` so wraparound is
handled automatically.

| Mode | Expected delta | dup observed |
|---|---|---|
| 4×4 | 1 | **0** in 318 frames |
| 8×8 | **4** | **0** in 600+ frames |

### ⚠️ TRAP #13 — the expected delta depends on the mode

The first version of the checker assumed delta should always be 1 and counted
everything above that as a dropped frame, so `skip` climbed by 3 every frame in
8×8 for no reason. **`dup` is the meaningful signal; `skip` must be compared
against the mode's expected delta.**

### ⚠️ TRAP #14 — `MY_TOF_FAST_READ` is only safe with INT

Run with `FAST_READ=1` **and** `USE_INT=0` and 43% of frames come back as exact
duplicates (`dup` climbed to 264 in 600 frames; identical F lines in the log).
Nothing gates the read in polling mode. The optimisation works because the INT
pin took over the gating role, not because the check was unnecessary.

Removing the vendor's safety check is only valid when something else provides the
same guarantee. **Verify, do not assume.**

## 3.3 Status field survey

537 frames × 64 zones = **34,368 readings → 100% reported valid** (BSP value 0).

**Therefore:** the low outliers we saw (291 mm in an early capture; zone 60 at ~235 mm)
are **real objects in the field of view**, not sensor errors.
The student proposed this explanation and was correct; the assistant initially assumed
they were noise and was wrong.

## 3.4 4×4 @ 60 Hz — the frame-rate ceiling

| Condition | Achieved rate | n (R lines) |
|---|---|---|
| Target setting | 60 Hz (period 16.67 ms) | — |
| Polling | 40.47 Hz (24.71 ms) | 3 |
| INT | 40.80 Hz (24.45 ms) | 14 |

Per-frame time breakdown at 4×4 (payload 532 B):

```
large I2C read      12.18 ms   50%
waiting for sensor  11.18 ms   46%   (CPU idle)
small I2C traffic    0.96 ms    4%
UART                 0.13 ms  0.5%
                    ────────
total               24.45 ms
```

**Causes ruled out by experiment:**
- Not the MCU code — polling and INT give the same rate (Δ = 0.33 Hz)
- Not integration time — 10 ms vs 5 ms gives the same rate (t = 0.42, n.s.)

### ⚠️ TRAP #4 — polling call count depends on idle time, not on "polling being bad"

At 8×8/15 Hz polling made ~116 I2C calls per frame. At 4×4/60 Hz it made ~4.
The 116 was a symptom of **having spare time to poll in**, not an intrinsic property
of polling. Do not generalise "polling = 116 calls".

## 3.5 Dose-response experiment — sensor waits for the MCU

Method: inject a busy-wait delay after each frame is handled, measure the resulting
frame period. 6 delay levels, ~6 R-lines each.

| Delay (ms) | Period (ms) | Rate (Hz) | Local slope |
|---|---|---|---|
| 0 | 24.45 | 40.90 | — |
| 2 | 25.29 | 39.53 | 0.42 |
| 4 | 27.09 | 36.92 | 0.90 |
| 6 | 29.07 | 34.40 | 0.99 |
| 8 | 31.06 | 32.19 | 1.00 |
| 12 | 35.05 | 28.53 | 1.00 |

```
Fit over delay ≥ 2 ms:   period = 23.24 + 0.981 × delay      R² = 0.9997
```

**Interpretation:** the sensor and the MCU read are **serialised ~98%**. Every 1 ms of
MCU time added extends the frame period by 0.98 ms. Equivalently, every 1 ms saved
shortens it by 0.98 ms.

Predictions were written *before* each run and matched:
6 ms → predicted 29.09, got 29.07 · 8 ms → predicted 31.04, got 31.06 ·
12 ms → predicted 34.89, got 35.05 (this one was extrapolation outside the fit range).

The low slope at 2 ms (0.42) is explained by ~1 ms of idle slack absorbing the first
part of the delay; the slope saturates at 1.00 once slack is exhausted.

### Derived predictions from this model — TO BE TESTED

| Config | Payload | Predicted transfer | Predicted period | Predicted rate |
|---|---|---|---|---|
| 4×4, current | 532 B | 12.06 ms | 24.45 ms | 40.9 Hz |
| 4×4, dist+signal+status+nb | ~148 B | 3.42 ms | ~15.98 ms | ~62.6 Hz |
| 8×8, current | 1,444 B | 32.58 ms | 33.65 ms | 29.7 Hz |
| 8×8, dist+signal+status+nb | ~536 B | 12.15 ms | ~13.61 ms | ~73.5 Hz |

Payload figures are estimates; the firmware reports actual bytes in the `T,` line
field `max_bytes`. **Use the measured value, not the estimate, once available.**

## 3.6 Gesture separability (early experiment)

Static, at 300 mm, threshold-based zone counting:
open hand covered **24–32 zones**, fist covered **16–21 zones** — non-overlapping
across 5 runs / 625 frames, 0 dropped frames.

---

# 4. FIRMWARE STRUCTURE (our own code)

| File | Role |
|---|---|
| `my_platform.c` | I2C platform layer calling HAL directly (bypasses ST's BSP function-pointer struct) + DWT cycle counter + per-call statistics |
| `my_uart.c` | Interrupt-driven UART TX, 2048-byte ring buffer, overrides `_write()` so `printf` is non-blocking |
| `my_tof.c` | Frame acquisition, output formatting, frame-rate measurement |
| `my_tof.h` | Resolution/frequency switch |

**Compile-time switches (all A/B testable by changing one number):**

| Switch | File | Values |
|---|---|---|
| `MY_TOF_USE_4X4` | `my_tof.h` | 1 = 4×4 @ 60 Hz · 0 = 8×8 @ 15 Hz |
| `MY_TOF_USE_INT` | `my_tof.c` | 1 = INT/ASYNC · 0 = polling/BLOCKING |
| `MY_TOF_TIMING_MODE` | `my_tof.c` | 1 = `T,` timing lines · 0 = `F,` + `S,` data lines |
| `MY_TOF_DELAY_US` | `my_tof.c` | injected delay for the dose-response experiment |
| `MY_UART_USE_IRQ` | `my_uart.c` | 1 = interrupt TX |

**Timing budget must be < the frame period.** 4×4 @ 60 Hz → 10 ms. 8×8 @ 15 Hz → 30 ms.

**Serial output line formats:**
```
F,<frame>,<64 distances>                                     data mode
S,<frame>,<64 statuses>                                      data mode
T,<frame>,<calls>,<bytes>,<us>,<max_bytes>,<max_us>,<uart_us>  timing mode
R,<frame>,<count>,<elapsed_ms>                               both modes, every 60 frames
CLK,<SystemCoreClock>                                        once at boot
MY_TOF: init OK (...)                                        once at boot, states all settings
```

`read_tof.py` ignores any line not starting with `F,` — so `S,` and `R,` lines are
backward compatible with the older tools.

---

# 5. DATASETS — verified status

## 5.1 `ST_VL53L8CX_handposture_dataset` — the one usable dataset

Location: `stm32ai-modelzoo-services/hand_posture/datasets/ST_VL53L8CX_handposture_dataset.zip`
(2.65 MB in the repo, free, no registration). **Downloaded and inspected directly.**

| Property | Value |
|---|---|
| Sensor | **VL53L8CX** — exact match to ours |
| Resolution | **8×8** |
| Total frames | **11,448** (counted, not quoted) |
| Log sessions / npz files | 42 / 162 |
| Users | **4** |
| Distance range used | 100–400 mm |

**Per-class frame counts (counted):**

| Class | Meaning | Frames | Share |
|---|---|---|---|
| Fist | closed fist | 2,274 | 19.9% |
| FlatHand | open palm | 1,738 | 15.2% |
| Dislike | thumb down | 1,686 | 14.7% |
| Like | thumb up | 1,580 | 13.8% |
| Love | love sign | 1,274 | 11.1% |
| CrossHands | crossed hands | 1,266 | 11.1% |
| BreakTime | T shape | 1,243 | 10.9% |
| **None** | **not one of the 7 postures** — see TRAP #18 | **387** | **3.4%** ← under-represented AND single-user |

(A stray duplicate `FlatHand` row in the previous version of this table has been
removed. Counts re-verified 3 Sep with `st_dataset.py --data-dir ...`; after the
100–400 mm filter the per-class totals are 2,274 / 1,738 / 1,686 / 1,580 / 1,274 /
1,266 / 1,243 / **382**, total **11,443**.)

**Per-user distribution (counted):**

| User | Frames | Share | Classes performed |
|---|---|---|---|
| User1 | 4,971 | 43.4% | **8** |
| User2 | 2,134 | 18.6% | 7 |
| User3 | 2,172 | 19.0% | 7 |
| User4 | 2,171 | 19.0% | 7 |

→ Subject-independent splits are possible but the base is narrow, and User1 is the
only user with all 8 classes.

### ⚠️ TRAP #18 — the `None` class blocks subject-independent evaluation

`None` exists **only for User1** (3 sessions, 382 frames after filtering).
Whichever user is held out, the split is defective:

| Held-out user | Outcome |
|---|---|
| User1 | **fatal** — every `None` frame lands in test; the model never sees the class in training |
| User2/3/4 | test set contains **zero** `None` frames; the class is never evaluated |

**`None` is not an empty scene.** Inspected 3 Sep — it contains *more* occupied zones
than `Fist`:

| Class | Median nearest distance | Occupied zones (sample frame) |
|---|---|---|
| FlatHand | 155 mm | 53 / 64 |
| Fist | 262 mm | 16 / 64 |
| **None** | **307 mm** | **45 / 64** |

`None` means *"something is there, but it is not one of the 7 postures"* — an open-set
reject class. That is why it needs **more** subject diversity than any other class,
not less: it must cover each person's scene, idle posture and transitions.

**Decision for Phase 2:** drop `None`, train on **7 classes**. Record the reason in the
thesis; it is a defensible methodological choice, not a shortcut. Collect `None` from
all volunteers in Phase 5 and restore the 8th class there.

Bartoli et al. 2024 `[PAPER]` used the same construct under the name **"Null"**
(4 classes: Paper, Scissors, Rock, Null). Ahmed et al. 2021 `[PAPER]` calls it
**"Empty gesture"**. The construct is standard; ST's execution of it is the weak part.

**File format (inspected):**
```
*.npz per log, keys: start_tstmp, end_tstmp, zone_data, glob_data, zone_head, glob_head
zone_data shape = (4, 64, N_frames)
zone_head = ['target_status', 'valid', 'signal_per_spad', 'distance_mm']
glob_head = ['.GestureGT']            ← ground-truth label
target_status here is RAW (5 = valid)  ← see TRAP #3
```

**Model shipped with it:** `st_cnn2d_handposture_8classes`
Source for every figure below: `stm32ai-modelzoo/hand_posture/st_cnn2d_handposture/README.md`
(a **different repo** from `-services`; opened 3 Sep 2026)

- input `(8, 8, 2)` = distance + signal_per_spad
- 2,752 parameters
- accuracy **98.47%** — see TRAP #16, this is NOT 99.43%
- **inference 1.46 ms on NUCLEO-F401RE @ 84 MHz** ← same MCU family and clock as ours
- RAM 1.91 KB, Flash 16.19 KB

**Architecture, read directly from the `.keras` file (3 Sep, no TensorFlow needed):**
```
Input (8, 8, 2)
Conv2D    8 filters, 3×3
Activation
MaxPooling2D 2×2
Dropout   0.2
Flatten
Dense     32, relu
Dense     8, softmax
```
Use this exact architecture for our own training, so that only the *split* differs
from ST and the comparison stays fair.

### ⚠️ TRAP #5 — inference cost was overstated by 3.7×

The assistant used **5.34 ms** (from Bartoli 2024's own CNN) for weeks.
The correct figure for the model we would actually deploy is **1.46 ms**.
Always state which model a latency number belongs to.

### ST's preprocessing (read from `tf/src/preprocessing/data_loader.py`)
```
Max_distance 400 mm · Min_distance 100 mm · Background_distance 120 mm
valid_status = [5, 9]
invalid zone → distance = 4000, signal = 0
normalisation: (distance - 295)/196 ,  (signal - 281)/452
augmentation: random horizontal flip
training: batch 32, epochs 1000, Adam lr 0.01, dropout 0.2, seed 42
```
### ⚠️ TRAP #16 — 3,031 / 1,146 belongs to a DIFFERENT dataset

An earlier version of this file said: *"the model README reports 3,031 train / 1,146
test frames — i.e. most of the 11,448 raw frames are filtered out."*
**Both halves of that sentence were wrong.** Verified 3 Sep 2026:

| Claim | Reality | Source |
|---|---|---|
| 3,031 / 1,146 come from our dataset | They come from **ST's internal dataset**, path `/local/datasets/hpr_st_vl53l8cx_handposture_dataset` — a different name, never published | `st_cnn2d_handposture_8classes_config.yaml:23` |
| most of 11,448 is filtered out | **11,443 of 11,448 survive** (99.96%). Only 5 frames exceed 400 mm | `[MEASURED]` `verify_against_st.py` |
| do not quote 11,448 | **11,448 is correct** and is ST's own published figure | `modelzoo-services/hand_posture/docs/README_DATASETS.md:6` |

So the public dataset (11,448) and the dataset ST actually trained on (4,177) are
**two different datasets**. ST never released the one the shipped model was trained on.

**Consequence for the thesis:** we cannot reproduce ST's 98.47% exactly, because we do
not have their training data. What we *can* do — and what contribution #4 is — is train
the same architecture on the public dataset under both split regimes and report both.

### ⚠️ TRAP #17 — accuracy figures carry conditions

| Figure | Belongs to | Measured on |
|---|---|---|
| **98.47%** | `st_cnn2d_handposture`, VL53L8CX | **validation set** of ST's internal dataset (filename: `float_model_confusion_matrix_validation_set.png`) |
| 99.21% | same model, **VL53L5CX** dataset | different sensor — not our number |
| ~~99.43%~~ | **found nowhere** in either ST repo (grepped both, 3 Sep) | unknown provenance — do not use |

ST's split, from `data_loader.py:184-191`: `shuffle()` then `take`/`skip`.
No grouping by user anywhere in the file. **This is the gap contribution #4 fills.**

### Frames vs zones — two different filters, do not confuse them

| Unit | Removed | Kept |
|---|---|---|
| **Frames** (whole 8×8 captures) | 5 of 11,448 | **11,443 — 99.96%** |
| **Zones** (individual cells) | 460,344 of 732,672 | **272,328 — 37.2%** |

Zone removal splits into 23.9% bad `target_status` and 38.9% background
(deeper than the nearest point + 120 mm). Removed zones are not deleted — they are
set to distance 4000 / signal 0, so every surviving frame still has all 64 cells.
`[MEASURED]` 3 Sep, `verify_against_st.py --data-dir ...`

## 5.1b Our own loader — written and verified 3 Sep 2026

Two files in `pc_tools/`:

| File | Role |
|---|---|
| `st_dataset.py` | loads the `.npz` files, applies ST's preprocessing, **and keeps the user id** parsed from the folder name (`__User2__`) so a group-wise split is possible |
| `verify_against_st.py` | contains ST's own for-loop logic copied verbatim, runs both implementations on every file and compares |

**Verification result:** `162 / 162 files identical`, compared with `atol=0, rtol=0`
(bit-for-bit). Our vectorised implementation is therefore a drop-in equivalent of
ST's, and any later difference in results comes from the split, not the preprocessing.
This is re-runnable — do not take it on trust, run it again after any edit.

Why we did not simply edit ST's `data_loader.py`: it never records which user a frame
came from, and the advisor's standing instruction is to write our own layer rather than
adapt vendor demo code (§7 item 2).

## 5.2 Dynamic-gesture datasets — what exists

| Source | Gestures | Sensor | Data available? |
|---|---|---|---|
| **IEEE DataPort — ToF Gesture Recognition** | **4 dynamic**: Approaching, Swipe, Hold, Tap · 8,400 samples (2,100 each) · 32 Hz · CSV · distance only | **VL53L5CX** on B-U585I-IOT02A | 🟡 yes, IEEE account required |
| STSW-IMG035 | 7 dynamic: tap, double tap, swipe ×4, circular, level control | VL53L7CX / VL53L8CX | ❌ compiled library only. ST states it is **algorithmic (hand-centroid tracking), NOT AI** — re-confirmed 3 Sep from the DB4700 data brief, which separates *"gesture recognition algorithms"* from *"hand posture recognition … enabled via AI algorithms"* |
| Polito MSc thesis (2021) | 7 dynamic: 4 swipes, CW, CCW, unknown · 17,500 samples · 5 users | VL6180 ×3 (single-zone) | ❌ not released |

### ⚠️ TRAP #6 — do not say "there is no dynamic ToF dataset"

There is one downloadable dynamic multizone-ToF dataset (VL53L5CX).
The accurate statement is:

```
dynamic + multizone ToF + downloadable        → 1 dataset (VL53L5CX)
dynamic + VL53L8CX                            → none
dynamic + VL53L8CX + signal_per_spad          → none
```

VL53L5CX vs VL53L8CX: same 8×8 architecture, same vendor, adjacent generation.
ST states cross-variant transfer degrades accuracy, so it is **not** valid as final
training data — but it is valid for **developing and debugging the dynamic pipeline**.

## 5.3 Not usable (checked, ruled out)

| Source | Why not |
|---|---|
| VIZTA / TiCaM (DFKI) | Kinect Azure ToF **camera**, in-cabin person/object detection, not hand gestures |
| FLAT (NVIDIA) | synthetic ToF **camera** multipath data |
| HaGRID | RGB images, 18 classes — useful only as a gesture-vocabulary reference |
| NVGesture, Briareo, DHG, SHREC, etc. | depth **cameras** ≥ 320×240 |

**Why downsampling a depth camera to 8×8 does not work:** a VL53L8CX zone is a 5.6°
cone with multi-target capability, per-zone status and per-SPAD signal. A depth pixel
has none of these. Downsampling reproduces shape only, and loses 2 of the 4 channels
ST's model uses.

## 5.4 Search coverage — so this is not re-litigated

Searched and found nothing further: Kaggle, Hugging Face, Zenodo, IEEE DataPort,
figshare, Papers With Code, arXiv surveys (×2), GitHub, Edge Impulse, Chinese-language
sources, other ToF vendors.

### ⚠️ TRAP #7 — search with the vendor's vocabulary

`"ToF gesture dataset"` → does **not** surface ST's dataset.
`"hand posture recognition dataset STM32 AI"` → surfaces it as the top result.

| We say | ST says |
|---|---|
| gesture (static) | **hand posture** |
| gesture (dynamic) | gesture / motion |
| driver | ULD |
| platform layer | porting layer |

Before concluding "it does not exist", search using the manufacturer's terms.

## 5.5 Reference implementations available

| Repo | Contents |
|---|---|
| `STMicroelectronics/STM32F4-GettingStarted-HandPosture` | Complete app for **NUCLEO-F401RE + X-NUCLEO-53LxA1**: prebuilt `.hex`, `app_sensor.c`, `app_network.c`, `app_comm.c`, converted model, ST Edge AI runtime |
| `DenissStepanjuk/UWB-Gestures-classification...` (advisor's link) | PyTorch CNN / RNN / Transformer on UWB radar. RNN 98.41%, Transformer 96.03%, CNN ~92.5% |
| UWB-Gestures dataset | **free on figshare** (not only IEEE): `figshare.com/articles/dataset/.../12652592` |

### ⚠️ TRAP #8 — do not copy the advisor's repo's split

```python
train_test_split(X, y, test_size=0.2)   # random, and no random_state
```
Two defects: not subject-independent (inflates accuracy), and not reproducible.
The loader already reads per-person (`zest(person, ...)`), so a subject-wise split is
easy. **This is an improvement our project can make, stated constructively.**

---

# 6. PAPERS IN PROJECT KNOWLEDGE

| Paper | Sensor | Setup | Key numbers | Why it matters |
|---|---|---|---|---|
| **Boner et al. 2022**, Tiny TCN, IEEE AICAS (ETH Zürich) | VL53L**5** | **4×4 @ 60 Hz**, 12 users, 6 dynamic + idle | **96.05%**, <100 kB, 5-frame window (0.25 s), TFLM int8, **DWT cycle counting**, 10-fold CV | Justifies 4×4@60Hz for dynamic; same measurement method as ours; 5-frame window means our 8 frames/gesture at 40.9 Hz is sufficient |
| **Wang et al. 2023**, IEEE Conf | P8864 8×8 | **STM32F411CEU6** — same MCU as ours | TPC-Net **98.47%**, cross-subject **91.04%** | Shows the accuracy drop when evaluation is subject-independent |
| **Bartoli et al. 2024**, IEEE Conf | VL53L8C**H** + F401RE | static, 4 classes | distance only **88%**, +signal **96%**, inference 5.34 ms (their CNN) | Source of the "signal matters" argument; **5.34 ms is their model, not ST's** |
| **Ahmed et al. 2021**, Scientific Data 8:102 | UWB radar | 8 volunteers, 12 gestures × 100 | 9,600 samples, 90×189 per sample | Best template for dataset design: records age/weight/BMI, includes an "Empty gesture" class, balanced, ships a viewer script |
| **Zabierowski et al. 2026**, IRS, Warsaw UT | **VL53L8CH + NUCLEO-L476RG** | ToF as low-cost radar substitute | velocity `V[n] = (x[n]−x[n−1])/Ts`, complexity **O(N)** vs radar O(MN log MN) | **Closest hardware match to ours.** The velocity formula is directly usable as a dynamic-gesture feature. Likely why the advisor is buying an L476RG |
| **Ahmed et al. 2026**, IEEE T-Radar | XeThru UWB ×3 | CNN-SVM + FPGA | 93.3% vs 92.2%, 75.5% fewer params, Vitis-AI INT8 | Structural template for a hardware-acceleration paper |

Also in Project Knowledge: Polito MSc thesis (dynamic, VL6180 ×3, 7 gestures, 17,500
samples, MLP 95%) — **useful as a thesis-writing template and for its variable-length
handling**: capture a fixed 100-frame window per gesture, pad short gestures with the
sensor's max range value (physically meaningful, since "no hand" reads max range).

---

# 7. ADVISOR (อ.Seal) — instructions and status

| # | Instruction (verbatim intent) | Status |
|---|---|---|
| 1 | Read section 4 (I2C) of DS14161 | ✅ |
| 2 | Write the M4 ↔ ToF connection driver yourself, not ST's demo | ✅ `my_platform.c` |
| 3 | Use interrupt for I2C data reception | ✅ INT pin via EXTI |
| 4 | Use interrupt for USART transmission | ✅ ring buffer |
| — | *"Interrupt ให้ใช้งานได้ก่อน ครับ"* → make interrupt actually work before considering DMA | ✅ verified §3.2 |
| 5 | Measure timing diagram with a logic analyser | ⏳ hardware ordered, ETA 29–31 Aug |
| — | *"ลองค้นหาข้อมูล data set จาก ToF ว่ามีบ้างมั้ย ครับ"* | 🟡 partially answered; the assistant initially missed ST's dataset |

**Advisor's stated position on DMA:** polling for experiments, interrupt for real use,
DMA only if the frequency is high. Our measurement supports deferring DMA — see below.

### ⚠️ TRAP #9 — DMA would not fix the current bottleneck

DMA frees the **CPU**, not the **bus**. Measured CPU idle is already 46% of the frame
period. The constraint is bus occupancy (12.18 ms of a 24.45 ms period), so DMA cannot
shorten the period. Fewer bytes, or SPI, can. State this when the topic comes up.

**Still unresolved with the advisor:** static vs dynamic gesture set (asked ~5 times,
no answer). The project can proceed: do static first using ST's dataset, dynamic second.

**Advisor's shopping cart (31 Aug)** — 6 items, 3 are microphone boards for a different
project. Relevant to us: `SATEL-VL53L8` (breakout, 2 boards, VL53L8CA — lets the sensor
be placed away from the board, and is what Bartoli used), `NUCLEO-L476RG` (1 MB flash,
ultra-low-power, Arduino-compatible so our shield fits, and the board used in the
ShortRange paper), `FRDM-MCXN947` (NXP, eIQ Neutron NPU — a possible AI-acceleration
comparison, but a different ecosystem and a port would be substantial work).

---

# 8. WHERE THE PROJECT STANDS

## Contributions — status

| # | Contribution | Novel? | Status |
|---|---|---|---|
| 1 | Measuring sensor→MCU transfer time | none of the 5 papers report it | ✅ done |
| 2 | Sensor/MCU serialisation model, slope 0.981, R² 0.9997 | not reported anywhere | ✅ done |
| 3 | 8-condition timing matrix, one variable per step, both modes | not reported anywhere | ✅ done |
| 3a | Mode-dependent effect of the same optimisation (headroom vs throughput) | not reported anywhere | ✅ done |
| 3b | Working on-board inference + 4-stage latency breakdown | — | ⬜ |
| 4 | Subject-independent + cross-dataset evaluation | ST did not do this — **verified in their source**, `data_loader.py:184-191` shuffles and splits with no user grouping | 🟡 loader + verification done 3 Sep; splitter and training still to write |
| 5 | First dynamic-gesture dataset for VL53L8CX | nothing comparable exists | ⬜ |

## Plan

```
Phase 0    lock the data format + clock + measurement matrix               DONE 2 Sep
Phase 1    logic analyser verification                                     1 day   (waiting on hw)
Phase 2    train on ST's dataset; subject-independent split                1 wk
Phase 3    deploy AI to board, instrument 4 stages                         1 wk    🏁 system complete
Phase 4    closed-loop check: our board's data → ST's model                2-3 d
Phase 5    collect our own static dataset (4-6 people)                     2-3 wk
Phase 5.5  train on our data, cross-dataset comparison, deploy             1-2 wk
Phase 6    collect dynamic dataset                                         3-4 wk
Phase 6.5  train temporal model (TCN/RNN), deploy                          2-3 wk
Phase 7    write up                                                        1 mo
```

Ordering principle: cheapest-and-reversible first, hardest-to-undo last.
Recruiting volunteers is the least reversible step, so it comes after the pipeline is
proven end-to-end.

## Immediate next action

Phase 0 is complete. Both modes now run at the hardware ceiling with no missed
ticks (`delta` constant, `dup = 0` across all eight conditions).

**Priority when the logic analyser arrives: Phase 1 first, ahead of Phase 2.**
It is advisor task 5 (advisor work outranks self-directed work), it takes one day
against Phase 2's week, and the hardware may have to be returned or shared —
whereas the dataset and loader sit on disk and wait indefinitely.

Ordered:

1. **Phase 1 — logic analyser** (when hardware arrives). Probe SCL/PB8, SDA/PB9,
   INT/PA4. Sample rate ≥ 4–8 MHz (≥10× the 400 kHz bus, or I2C will not decode).
   Capture ≈200 ms to span several frames in both modes (8×8 period 66.17 ms,
   4×4 period 16.55 ms). Besides satisfying task 5, this is the third independent
   clock needed to settle TRAP #12 — is the sensor's oscillator 0.7% fast, or the
   board's reference?
2. **Phase 2 — training.** Loader and verification are done (§5.1b). Next: write the
   splitter (`GroupKFold` / leave-one-subject-out on the `groups` array), then train
   the ST architecture under both split regimes and compare.
3. Set `MY_TOF_TIMING_MODE = 0`, capture F/S/G lines, confirm distances still
   match the earlier captures and that `G,` carries sensible signal values.

**Note on the repo:** commit `0f62812` (3 Sep, "Add matrix data, analysis script and
figure") has `MY_TOF_FAST_READ = 0` — i.e. rows C/G, the BSP path, not the D/H ceiling.
Not a bug, but set it to 1 before any run that is meant to reproduce D or H.

Firmware state after Phase 0: `USE_4X4` per experiment, `USE_INT=1`,
`FAST_READ=1`, `TIMING_MODE=1`, `DELAY_US=0`, 5 DISABLE macros active, HSE clock.

---

# 9. CORRECTION LOG — mistakes the assistant actually made

Kept so the same errors are not repeated. Each was caught by the student.

| # | Claim made | Reality | Root cause |
|---|---|---|---|
| 1 | "43.4° is wrong, it's 45°" (implying the advisor erred) | Both correct; FoI ≠ FoV | Did not open the datasheet |
| 2 | "The 291 mm reading is noise, filter it" | Real object; 100% of statuses valid | Assumed instead of measuring |
| 3 | "Cut the unused fields" | signal_per_spad is required by ST's model | Did not check what ML would need |
| 4 | "INT is slower than polling" (from n=1) | INT is slightly faster (n=14) | Concluded from a single data point |
| 5 | "No dataset matches our sensor" | ST publishes an exact-match dataset | **Never actually searched** |
| 6 | "I have searched exhaustively" | Two queries, both reusing terms from an already-found result | Searched to confirm, not to discover |
| 7 | "No dynamic dataset at all" | One exists (VL53L5CX, IEEE DataPort) | Did not check the list item by item |
| 8 | inference = 5.34 ms | 1.46 ms for the model we would deploy | Quoted a different paper's model |
| 9 | Deployment listed as one sub-bullet | It is the step that completes the whole contribution | Under-weighted in planning |
| 10 | Phases 5 and 6 ended at "collect data" | Collected data must then be trained and deployed | Plan written carelessly |
| 11 | Proposed `FAST_READ` without stating it requires INT | In polling mode it duplicates 43% of frames | Did not think through when the removed check mattered |
| 12 | Payload estimates: predicted 148 B (got 196), 536 B (got 580) | Consistently ~45 B low | Scaled by zone count, ignored per-block headers |
| 13 | Wrote "we exceeded the datasheet spec" | "up to 60 Hz" is a ceiling; +0.7% is an unattributed offset | Treated a favourable number as a result instead of questioning it |
| 14 | Claimed INT is what takes 4×4 to 60 Hz | INT changes nothing there (E→F: 40.57→40.49); the payload reduction does (F→G: +49%) | Generalised the 8×8 result to 4×4 without measuring |
| 14 | Predicted INT would lift 4×4 to 60 Hz | INT changed the rate by 0.08 Hz; the payload cut did it | Assumed the 8×8 finding transferred; it is a deadline, not a proportion |
| 15 | Predicted rows E–G would drop 1.36% with the new clock | Period is quantised to the sensor tick, so it barely moved | Applied a scale factor to a quantised quantity |
| 16 | Saw 4,177 ≠ 11,443 and explained it as *"the note confused frames with zones"* | The two numbers are from **two different datasets**; ST trained on an unpublished internal one. Nothing to do with units | **Invented a plausible-sounding explanation instead of tracing the number to its source.** The zone figures happened to fit, which made the wrong answer feel right |
| 17 | Reported "STSW-IMG035 is a new finding that may threaten contribution #5" | It was already in §5.2, researched in more depth than the fresh search | Read this file with a truncated view and did not notice the gap |

**Pattern:** answering from memory while presenting it as verified.
**Countermeasure:** the tagging rule at the top of this file.

**Second pattern, added 3 Sep — inventing explanations for discrepancies.**
When two numbers disagree, the only acceptable moves are: open the primary source, or
say `[ยังไม่ได้ตรวจ]` and ask. A hypothesis that "makes sense" is not an answer, and is
more dangerous than saying nothing, because it sounds finished. Error #16 would have
gone into the thesis if the student had not said *"go and find out"*.

---

*Last updated: 2026-09-04 · Phase 0 complete · Phase 2 loader written and verified
against ST 162/162 · three §5.1 figures corrected (TRAP #16–#18) · logic analyser
installed and its capabilities verified on the actual unit (§1.4, TRAP #19).*

---

# APPENDIX — duplicate copy of the 4×4 matrix

The rows below repeat §3.1 in a slightly different layout. Kept because the
per-step attribution text underneath is not duplicated anywhere else.

| **A** | 8×8@15Hz | all | polling | bsp | 115.8 | 1444 | **55.49 ms** | 66.17 | 15.11 Hz | **83.9%** | 10.68 |
| **B** | 8×8@15Hz | all | **int** | bsp | 4.0 | 1444 | **33.92** | 66.17 | 15.11 | 51.3% | 32.25 |
| **C** | 8×8@15Hz | **slim** | int | bsp | 4.0 | **580** | **14.37** | 66.17 | 15.11 | 21.7% | 51.79 |
| **D** | 8×8@15Hz | slim | int | **direct** | **1.0** | 580 | **13.44** | 66.17 | 15.11 | **20.3%** | 52.73 |
| **E** | 4×4@60Hz | all | polling | bsp | 13.5 | 532 | 13.33 | 24.65 | 40.57 | 54.1% | 11.32 |
| **F** | 4×4@60Hz | all | **int** | bsp | 4.0 | 532 | 13.32 | 24.70 | 40.49 | 53.9% | 11.38 |
| **G** | 4×4@60Hz | **slim** | int | bsp | 4.0 | **196** | **5.63** | **16.55** | **60.42** | 34.0% | 10.92 |
| **H** | 4×4@60Hz | slim | int | **direct** | **1.0** | 196 | **4.69** | 16.55 | **60.42** | **28.3%** | 11.86 |

**All eight rows use the HSE crystal clock.** `delta` was 4.0 in every 8×8 row,
1.5 in E and F, and 1.0 in G and H.

### Per-step attribution — 4×4 (the informative one)

| Step | Change | Rate | Note |
|---|---|---|---|
| E → F | polling → INT | 40.57 → 40.49 Hz | **no change** |
| F → G | disable unused fields | 40.49 → **60.42 Hz** | **+49.2%** |
| G → H | bypass redundant I2C | 60.42 → 60.42 Hz | already at the ceiling |

### ⚠️ TRAP #15 — at 4×4 it is the payload, not the interrupt, that buys the rate

The sensor's internal tick is 16.55 ms. To hit 60 Hz the MCU must finish inside
one tick **with margin**, not merely finish.

| | MCU work | Slack vs tick | delta |
|---|---|---|---|
| F | 13.32 ms | 3.23 ms | **1.5** — misses ticks |
| G | 5.63 ms | 10.92 ms | **1.0** — catches every tick |

3.23 ms of slack is not enough because per-frame time fluctuates slightly.
**"It fits" is not the same as "it fits reliably."**

### The same change has different effects in the two modes

| | 8×8 @ 15 Hz | 4×4 @ 60 Hz |
|---|---|---|
| Time available per frame | 66.17 ms | **16.55 ms** |
| What INT buys | **−39% MCU load**, rate unchanged | fewer I2C calls, rate unchanged |
| What payload reduction buys | −58% MCU load, rate unchanged | **rate +49%** |
| Binding constraint | the sensor | **our system** |

At 8×8 there is time to spare, so optimisation returns *headroom*.
At 4×4 time is tight, so the same optimisation returns *throughput*.
**This is why both modes had to be measured — reporting only one would have
given the wrong conclusion about what matters.**

