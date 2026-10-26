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
`[PAPER]` paper in Project Knowledge · `[STUDENT]` stated by Chin about the physical
setup or a conversation the assistant cannot observe directly

---

## QUICK STATUS — as of 26 Oct 2026

| Item | State |
|---|---|
| Phases 0–3 | ✅ complete (Phase 3 deployed on-board inference, 21 Sep 2026) |
| Phase 4 pre-flight check 6 (21-trial live test) | ✅ done 24–25 Sep 2026, 21/21 correct **with per-posture hand positioning** (§5.1e-R) |
| **Phase 4 first result (User 5 only)** | per-frame **66.3 %**, vote 67.6 %; five postures 70–100 %, **Like 0.3 % (→ Dislike), Love 0.0 % (→ BreakTime)**; P2 and H9 do not hold; n = 1, descriptive. Diagnosis: our hand sits ~1–2 rows off ST's position (§5.1e-E). **Revised 10 Oct: Love's 0 % is now attributed mainly to a validation-split bug in Phase 2 that starved Love (§5.1g-R, §9 #51); the deployed model and the Phase 2 LOSO numbers must be re-made** |
| **Phase 4 H8 evaluation, both subjects (20 Oct, deployed model with the §9 #51 bug)** | User 5 per-frame **66.3 %**, vote 67.6 %; User 6 per-frame **75.9 %**, vote 77.0 % (n = 2, descriptive). P1 and P3 hold; P2 and H9 do not (§5.1e-F). Like→Dislike is not shared (U5 Like 0.3 %, U6 Like 76.5 %) — hypothesis only |
| Phase 2 re-run (split fixed) | ST LOSO **85.28 %** (was 79.73); fixed production model on User 5: **68.8 %** — Love fixed (98.8), Like still 7.1, FlatHand fell to 6.8; strong seed dependence (§5.1h-R) |
| **Phase 5.5 result (20 Oct 2026, 90 models, commit `5c724d9`)** | Selection rule picks **arm B** (subject-balanced weights): ST-fold mean **B 85.61** vs A 84.17 vs C 81.05 %; B − A over 6 subjects +1.35 pp, Wilcoxon W = 5, p = 0.31 → **B not shown better than A**, chosen by the pre-registered rule. Arm B on our subjects: User 5 **82.8 ± 5.2 %**, User 6 **86.6 ± 6.0 %** (5 seeds). Q1 Q2 Q4 Q5 hold (Q5 by 0.11 pp), **Q3 fails** (arm A User 5 Like 3.2 %; arm B 23.1 %). Full record §5.1i-R |
| Error analysis of H8 (20 Oct) | hand position (vertical offset toward row 0, both subjects) is the best-supported cause of take-level failures, ρ = −0.58 over 28 takes, descriptive (§5.1j; full text `claude/ERROR_ANALYSIS_H8_2026-10-20.md`) |
| **Phase 5.5 final models (20 Oct 2026 21:18–21:32)** | arm B, all 6 subjects (36,409 frames), seeds 42–46 → `results_phase55_final\` (5 `.keras`, `manifest.json` with sha256), commits `e1f08ee` + `b0bcc01`. No accuracy (no held-out subject). Deployment rule + new-day test pre-registered §5.1l |
| **E5 built (20 Oct 2026 22:14, `pc_tools\build_ensemble.py`, commit `3fb36f0`)** | E5 = 22,555 params; E5 == mean of the 5 models (max |diff| 6.0e-08 over 8,448 frames); `E5.tflite` 100.9 KiB and fallback `seed42.tflite` 21.1 KiB, both via `from_keras_model`, same class as `.keras` on 2,000/2,000 frames (batch 1, max |diff| ≤ 3.0e-07); 200-frame board-check set + PC outputs saved in `results_phase55_final\deploy\`. No accuracy computed (check data is training data) |
| **E5 converted with X-CUBE-AI (26 Oct 2026, step 2.3, §5.1m)** | X-CUBE-AI 10.2.1 / ST Edge AI Core v2.2.0, TFLite, compression none, optimisation balanced, network name `network`, target stm32f4. Params 22,556 (= 5 × 4,511 + 1 averaging constant), MACC 73,762 (= 5 × 14,744 + 42), weights 88.16 KiB, Flash total 111.89 KiB, RAM total 11.21 KiB (activations 1.70 KiB). Validate on desktop: random data 10/10 (all one class — weak); **real data `board_check_inputs_200.npy` vs `board_check_E5_probs_200.npy`: 200/200 same class, all 7 classes (28–29 each), rmse 1.4e-08**. Code generated 15:50 into `ai_conversion_project\tof_ai_conversion\X-CUBE-AI\App\` (8 files). **Not yet on the board** — no on-board timing, no board-vs-PC check |
| **E5 deployed on the board (26 Oct 2026, step 2.4, §5.1n)** | All three §5.1l conditions met → **E5 is the deployed model**. Firmware build 0 errors / 0 warnings: Flash 247.4 KiB (48.3 %), RAM 26.6 KiB (20.8 %) by `arm-none-eabi-size`. Boot line `MY_AI: ... activations=1736 bytes` (= E5). **Per-frame work 29,306 ± 36 µs** (22 LAT windows × 30 = 660 inferences; inference 15,925 ± 35 µs; max window 29,324 ≤ 33,000). **Board vs PC 203/203** same class, 0 / 675 range-gate mismatches, max |prob diff| 1.1e-06 (`pc_tools\compare_board_pc.py`, live hand, all 7 postures). Firmware now has `MY_TOF_PRED_LINE = 1` (prints `Y,` lines) |
| **Phase 5.6, User 5 (26 Oct 2026 18:41–18:55, §5.1p)** | `data\phase56` 14/14 takes, 4,219 frames, 0 missing (v6 logger, firmware with Y+Q lines). **E5 93.0 % per-frame (vote 92.7 %) — measured on the board itself, identical to the PC**; 5 singles 89.0 ± 3.3 %; ST-only 68.2 %. Weak: Dislike 69.6 % (take 1 39 %), Like 81.4 % (take 2 63 %). Board vs PC **4,219/4,219**; per-frame work max **29,295 µs** (mean 29,204 ± 18), 0 frames > 33 ms. **B1–B7 all hold.** Caveat: hand centroids moved vs 9 Oct (Like +0.95 row toward ST) — seating change and/or having seen live predictions; cannot be separated. n = 1 subject, descriptive |
| **Next session (as of 26 Oct 2026, night)** | Phase 5.6 User 5 committed and pushed as **`9e5e7a4`** (37 files: firmware `MY_TOF_FRAME_TIMING`, `collect_own_data.py` v6, `evaluate_phase56.py`, `data\phase56\` 14 npz + 14 `_board.csv` + manifest, `results_phase56\eval_20261026_190627\`) `[ตรวจแล้ว: git push 19:20]`. Next: User 6 Phase 5.6 (same command with `--user 6`, he has not watched E5 predictions except a short live try on 26 Oct, §5.1m); A5 Love edge/centre live test; report/thesis writing |
| Phase 5.5a (shift augmentation) | **negative**: −12.1 pp vs retrained baseline, paired over 3 seeds (§5.1g-R); not adopted |
| Phase 4 data collection (`collect_own_data.py` **v5**) | ✅ **Done.** Chin (User 5) 42/42: main 14 (9 Oct) + extra 14 (9 Oct) + extra 14 (15 Oct, v5). Friend (User 6) 42/42: main 14 (14 Oct, v4) + extra 28 in one sitting (A10), file timestamps 18 Oct 2026 21:56–22:16 `[ตรวจแล้ว: xcopy listing 20 Oct]`; his first 14 extras (15 Oct) stay quarantined in `data\phase4_rejected` (A9, §9 #53). Manifest check of the 28 extras: missing frames 0, out-of-window 0, with warning 0 `[STUDENT]` (output pasted 20 Oct). `data\phase4_extra` holds 56 npz (28 + 28) + manifest = 57 files; `data\phase4` 28 npz + manifest = 29 files. Local backups on Chin's PC: `C:\Users\gaming\Documents\backup_phase4_20261020\phase4` (29) and `\phase4_extra` (57). Copy to Drive/USB: **not confirmed** |
| Collection protocol | guided arming: hand must hold inside **220–320 mm for 3 s** before recording starts; take discarded if the hand leaves the window for **8 consecutive frames**; no frame-level filtering (§5.1e-A8) |
| Pre-registration commit | Phase 5.5 pre-registration (§5.1i) written 14 Oct 22:55 **before** any friend extra data existed. Chin said "push แล้ว" on 15 Oct 00:31 `[STUDENT]` after being told to commit `PROJECT_FACTS.md`, `pc_tools\train_phase55.py`, `collect_own_data.py` (v5) and `quarantine_takes.py` and to leave out `results_phase55_SMOKE_TEST` and all `User6` data; the assistant did not check the remote (not needed, Rule 12). Friend's **main** set (collected 14 Oct, before the push) was first evaluated on 20 Oct 2026 (§5.1e-F) |
| Collection volume | 2 takes/posture/subject in `data\phase4` (H8 test set, as registered) **+ 4 extra takes/posture** in `data\phase4_extra` for Phase 5.5 training only. Chin's extras: **2 + 2 over two different days** (9 and 15 Oct, §5.1e-A7). Friend's extras: **one sitting** (Chin's decision 16 Oct, §5.1e-A10) — both subjects end with 3 × 14 = 42 takes |
| Subject pool | **hard limit n = 2 (Chin + 1 friend)**, stated by Chin 24 Sep 2026 — Phase 5 as originally planned (4–6 people) cannot happen (§8) |
| Deployed model file | **E5** (`results_phase55_final\deploy\E5.tflite` → X-CUBE-AI → board, since 26 Oct 2026, §5.1n); PC copy `E5.keras` sha256 `1a2268d3b5ab4fe7…`. Before 26 Oct: `results_phase2_final\production_model.keras` (§5.1d-F), files backed up in `C:\Users\gaming\Documents\backup_App_phase3_20261026\` |
| Logic-analyser probes | **still attached** as of 28 Sep 2026 `[STUDENT]` — not removed (§8, corrects the 11 Sep note) |
| Advisor message 28 Sep (ADC + logic analyser example) | **FYI only, not a task** — advisor's own reply 29 Sep 2026 (§7) |

---

# 1. HARDWARE SPECIFICATIONS

## 1.1 Sensor — VL53L8CX / VL53L8CA

| Fact | Value | Source |
|---|---|---|
| Chip actually on our shield | **VL53L8CA** (evaluation part, not the mass-production part) | `[UM3120]` |
| VL53L8CA relationship | ST support states it is a "super product" covering both VL53L8CX and VL53L8CH, and does support histograms — i.e. **not** a lesser or unrelated part; it is described as a superset of the two commercial variants | `[WEB]` ST community |
| ⚠️ Do not confuse with | A separate ST community reply warns that data from the **VL53L5CX** (a different, lower-resolution sensor *family*, not part of the VL53L8 line) does not transfer well to VL53L8CX-trained models. That warning is about a different product family — it is **not** a statement about VL53L8CA vs VL53L8CX. Conflating the two overstated Phase 4's risk in an earlier draft of this file (corrected 22 Sep 2026) | `[WEB]` ST community |
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

**Chip name check (21 Sep 2026, `[UM3120]` Rev 2 §1.6, p.4):** the manual states the
board "is equipped with a noncommercial VL53L8CA evaluation purposes only Time of
Flight sensor"; VL53L8CX and VL53L8CH are its equivalent orderable products. Progress
report 1 names the chip VL53L8CH — that is a related orderable part, not the chip on
our board. Progress report 2 uses **VL53L8CA**.

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

**No ADC is used anywhere in this project** — the firmware uses only I2C, the INT pin,
USART2 and (in Phase 1 only) TIM3 by direct register writes (§3.1b, §9 #17). The MCU
itself has an on-chip ADC; it is simply unused. (Checked 28 Sep 2026 against §1.3 and §4
when the advisor's ADC example arrived, §7.)

**Shield pin usage `[UM3120]` Figure 10–11 (opened 28 Sep 2026):** besides PB8/PB9 and PA4,
the shield routes several Arduino-header pins for SPI mode and for the optional left/right
satellite connectors. The schematic image is too small to read every label reliably, so
**which GPIO pins are genuinely free is `[ยังไม่ได้ตรวจ]`** — confirm against the project
`.ioc` before using any pin as a debug/marker output. PA5 (used as the marker pin in the
advisor's 28 Sep example) is also the NUCLEO user LED LD2 `[จำมา]`; whether the shield
loads it in I2C mode is not verified.

**Powering the NUCLEO without the PC (looked up in UM1724 §7.5, Oct 2026 `[WEB]`, not tried on the
bench):** external 7–12 V into **VIN** (CN6 pin 8, or CN7 pin 24), with **JP5 on pins 2–3** and
**JP1 removed**; connect the external supply **before** plugging USB. A USB phone charger/power bank
on the ST-LINK USB is the simpler option, but needs **JP1 fitted**. (The reason behind the JP1 rule
was not recorded here — re-read UM1724 §7.5 before relying on it.) Context: Chin asked about
battery power for the defence demo — an idea only, not decided. Re-open UM1724 for the exact jumper
labels before wiring anything.

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

**Probe wires — status corrected 28 Sep 2026 `[STUDENT]`:** the probes are **still attached**
to the board (§8 previously said they had been removed — that was wrong, §9 #45). The board has
since been moved between work locations, so **which channel sits on which pin must be
re-checked (photo or continuity) before any new capture**; do not assume the 10–11 Sep mapping
still holds.

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

The same rule applies to the board's serial port: close PuTTY / any serial monitor before
running `live_predict.py` or `collect_own_data.py`, or COM3 cannot be opened.

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

### What Phase 1 probed vs. the GPIO-marker technique (clarified 28 Sep 2026)

| | Phase 1 (done, 4–11 Sep) | GPIO-marker technique (advisor's 28 Sep example, §7) |
|---|---|---|
| Signals captured | Existing hardware nets: I2C SCL/SDA, sensor INT (PA4), plus a TIM3 1 kHz calibration output on PC8 | A GPIO the **firmware toggles itself** at an internal software event |
| What becomes visible | Sensor↔MCU traffic and frame timing | Timing of code running **inside** the MCU |
| Done in this project? | ✅ | ❌ not done — optional (advisor: "ให้ดูเฉยยย ครับ") |

If ever done, the natural use is to bracket the four on-board stages of Phase 3 (sensor read /
preprocessing / inference / argmax), currently measured only with DWT, to get an independent,
visual timing diagram. It would not fix a known problem — DWT on this board is already verified
three ways (§3.1b). Estimated effort half a day to a day `[ประเมิน]`; put it behind a new
compile-time switch (e.g. `MY_TOF_GPIO_MARK`) so the default build is unchanged.

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

**Addendum — verified in `my_tof.c` on 21 Sep 2026:** with `MY_TOF_FAST_READ = 1` the code
copies `RawData.target_status[k]` straight from the ULD (`vl53l8cx_get_ranging_data`), so
`g_status`, the `S,` serial line and the on-board AI all see **raw** status (5 = valid).
With `FAST_READ = 0` they would see the BSP-remapped value (0 = valid); both
`live_predict.py` and the on-board preprocessing (`valid = status == 5 || status == 9`)
would then treat every zone as invalid. **Keep `FAST_READ = 1` for anything that feeds the model.**

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

## 2.4 PC-side environment — verified 24–25 Sep 2026 (Chin's Windows PC)

| Item | Value | Source |
|---|---|---|
| Python | 3.11 (`...\Programs\Python\Python311\`) | `[MEASURED]` pip output |
| TensorFlow / Keras | **2.21.0** / **3.15.1** | `[MEASURED]` pip output |
| pyserial / numpy | 3.5 / 2.4.6 | `[MEASURED]` pip output |
| TF start-up messages (oneDNN, CPU instructions, "GPU support is not available on native Windows") | **informational only, not errors** — inference runs on CPU, fast enough for this model | `[MEASURED]` |
| Repo location | `C:\Users\gaming\Documents\GitHub\tof-gesture-recognition` | `[MEASURED]` `dir /s /b` |

**Duplicate scripts in the repo (found 24 Sep 2026):** `live_predict.py` exists both at the repo
root and in `pc_tools\`. The one this file has always referred to is **`pc_tools\live_predict.py`**.
The two copies have **not** been compared `[ยังไม่ได้ตรวจ]` — diff them, then delete the root copy,
to avoid the "edited one file, ran the other" failure (same family as §9 #38). Check
`collect_own_data.py` for the same duplication before its first run.

**Command that works (run from the repo root):**
```
python pc_tools\live_predict.py --port COM3 --model results_phase2_final\production_model.keras
```

**Windows shell note:** very long single-line commands pasted into Command Prompt were split
across two lines and failed (24 Sep 2026). `cd` to the repo root first and use short relative paths.

---

# 3. OUR OWN MEASUREMENTS

All measured with DWT cycle counter (11.9 ns resolution) on the actual board.
Method note: Boner et al. 2022 (ETH Zürich) use the same DWT technique — citable.

## 3.0 Hypothesis register — every question asked in Phase 1, and its verdict

Written here so the record is complete and auditable. Predictions were recorded
**before** each run; results were not allowed to edit the prediction. Two of the
seven predictions were wrong, and both are kept visible rather than quietly fixed.

| ID | Question / hypothesis | Prediction on record | Result | Verdict |
|---|---|---|---|---|
| **H1** | Which clock runs fast — the sensor or the board? | The board clock was the suspect (TRAP #12 as originally framed) | **The sensor is the fast one**, not the board clock. A crystal is ±50 ppm and cannot produce the observed +7000 ppm; the sensor's RC oscillator can | ✅ **Answered — original suspicion overturned** |
| **H2** | Is the DWT cycle counter on the board trustworthy? | — (verification question, no numeric prediction recorded) | **DWT measures correctly**, confirmed three independent ways | ✅ Answered |
| **H3** | What fraction of the I²C bus is idle? | — (measurement question, no numeric prediction recorded) | **72.8 % idle (4×4), 79.9 % idle (8×8)** | ✅ Answered |
| **H4** | What frequency does SCL actually run at? | Nominal 400.000 kHz | **399.947 kHz**, 132 ppm from nominal | ✅ Answered |
| **H5** | How long from the INT edge to the start of the conversation? | — (measurement question; the *definition* was initially wrong, see §9 #20) | **6.61 µs, SD 0.039**, measured to the I²C START condition, identical in both modes | ✅ Answered after the definition was corrected |
| **H6** | Is the true 4×4 vs 8×8 difference ≠ 0 once drift is removed? (A-B-A design; H0 = the whole difference is drift) | **1,000–1,400 ppm** after removing drift | **700.2 ± 11.3 ppm, t = 61.7.** H0 rejected. Independently corroborated at **698 ppm** from the period ratio | ⚠️ **Direction confirmed, magnitude wrong** — the real value fell 300 ppm below the predicted floor. See §9 #32 |
| **H7** | Cold-start drift: monotonic exponential rise to a plateau? | Start 6000–6200 → end 6850–6950 ppm; total **+700 to +950 ppm**; τ = 20–40 min | Start **7139.8**, end **6985.2**, total **−154.6 ppm** (opposite sign), three-phase curve, τ = 56.7 ± 4.3 min | ❌ **Rejected on all five counts.** See §9 #30 and `data/phase1_drift_v2/RESULTS_phase1_drift.md` |

**Why H6 and H7 both missed the same way.** Both predictions were produced by
extrapolating a curve whose shape had not been measured over the range being
predicted. H6 extrapolated drift linearly through a 28-minute gap; H7 extrapolated
the drift shape from a 92-minute window that happened to contain only the decay
phase. The countermeasure is in §9 under the third pattern: measure the shape over
the range you intend to predict, or do not predict a number.

Full method and raw data: `data/phase1_int/RESULTS_phase1.md` (H1–H6),
`data/phase1_drift_v2/RESULTS_phase1_drift.md` (H7).

## 3.1 The measurement matrix — 8 conditions, one variable at a time

**All eight rows use the HSE crystal clock. dup = 0 in every row**, verified
against the sensor's own `streamcount`.
Raw data: `pc_tools/timing_8x8_matrix.csv`. Analysis: `pc_tools/plot_matrix.py`.

> ⚠️ **Period and Rate columns are DWT values, ±1013 ppm.** Valid for comparing
> the eight rows against each other — one instrument, one session — but not the
> authoritative frame rates. Those are in §3.1b, and they carry a timestamp
> because the sensor drifts (§3.1d).
>
> **The MCU-work column is confirmed accurate.** DWT on this board was verified
> against an independent calibrated clock three ways (§3.1b, TRAP #12).

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
not sufficient. Report margin, not just the mean.

> ### ⚠️ OPEN QUESTION — the old explanation no longer fits
>
> This section used to attribute F's missed ticks to "per-frame jitter". The
> logic-analyser campaign measured that jitter directly: **200 µs at 4×4**
> (§3.1e). A 3.23 ms margin is **16× that** — jitter alone cannot explain why F
> missed ticks so often (delta 1.5).
>
> Something else is involved. Candidates not yet tested:
> - the 13.32 ms figure is a mean and the per-frame maximum is much larger
> - blocking somewhere outside the measured window, e.g. UART back-pressure
> - the sensor's own timing shifting during the run (§3.1d drift)
>
> **[ยังไม่ได้ตรวจ]** Do not repeat the jitter explanation until it is tested.
> The *conclusion* of TRAP #15 — payload, not interrupt — is unaffected and is
> now confirmed independently below.

### ✅ Confirmed 10 Sep 2026 — INT is definitively not the bottleneck

Measured on the wire: **INT falling edge → I2C START = 6.61 µs, SD 0.039**,
identical in 4×4 and 8×8 (§3.1c).

The frame period is 16.55 ms. The MCU begins the transaction **~2500× faster**
than frames arrive. Interrupt latency was never capable of limiting the rate,
which is why switching to INT moved 4×4 by only 0.08 Hz.

The bus is idle 72.8% of the time at 4×4 (§3.1c) — direct evidence that the
constraint is the sensor's production rate, not the MCU's ability to consume.

### Headroom for inference — ST's published 1.46 ms (SUPERSEDED, kept for the record)

The 1.46 ms is **ST's published figure** for its shipped 8-class st_cnn2d on
NUCLEO-F401RE @ 84 MHz (model-zoo README, §5.1) — not something we measured.

| | Load with AI | Inferences fitting in idle |
|---|---|---|
| A | 86.1% | 7 |
| **D** | **22.5%** | **36** |

**Superseded by Phase 3 (21 Sep 2026):** inference measured on our own board is
**3.197 ms** (7-class `production_model`, X-CUBE-AI 10.2.1, STM32F411RE @ 84 MHz, HSE,
DWT cycle counter). Total per-frame work at condition D with the model running is
16.58 ms = ~25.1% of the 66.17 ms budget. Use these, not the table above.
**Open question:** why 3.197 ms vs ST's 1.46 ms — see §8.5 D2. Do not explain the gap
until it is tested.

## 3.1b Frame rate vs. datasheet — RESOLVED 10 Sep 2026

Superseded by the logic-analyser campaign. The DWT/`HAL_GetTick` figures below
were the best available until then, but the `R` line has ±1013 ppm of
quantisation (it counts whole milliseconds over a ~1 s window), which is why the
two modes appeared to overshoot by the *same* 0.7%. They do not.

**Authoritative values** — logic analyser, calibrated, least-squares fit over all
edges, interleaved A-B-A design so drift cannot be confused with mode. Full
method and raw data in `data/phase1_int/RESULTS_phase1.md`.

| Mode | UM3109 spec | Measured | Period | Over spec |
|---|---|---|---|---|
| 8×8 | 15 Hz | **15.1150 Hz** | **66.1595 ms** | **+0.767%** |
| 4×4 | 60 Hz | **60.4178 Hz** | **16.5514 ms** | **+0.696%** |

Difference between modes (**this is H6**, see §3.0): **700.2 ± 11.3 ppm, t = 61.7**
— highly significant, H0 ("the whole difference is drift") rejected. The prediction
on record was 1,000–1,400 ppm, so H6's *direction* held but its *magnitude* did not
(§9 #32).

> **These are snapshot values.** The sensor drifts (§3.1d). Quoting a frame rate
> without saying when it was measured is now an error.

`delta` is exact and `dup = 0`, so the system reads **every** frame the sensor
produces in both modes. There is no remaining rate to recover.

What is left to optimise is negligible: of D/H's remaining MCU work, the I2C
payload read is 96% (4.51 of 4.69 ms at 4×4) and is bounded by the 400 kHz bus.
Eliminating everything else would save 0.18 ms and change no rate.

### ✅ TRAP #12 — RESOLVED: the sensor is the fast one

**Still do not write "we exceeded the datasheet"** — the datasheet says *"up to"*,
a configuration ceiling, not a guaranteed exact rate.

But the source of the overshoot is now settled. The logic analyser's crystal was
calibrated against the board's HSE by having the board emit an exact 1000.000000 Hz
square wave (TIM3 CH3 on PC8, PSC=0 ARR=41999, 0 ppb rounding error):

```
LA vs board clock:  -95.408 ppm   SD 0.349   n = 3
```

Two independent crystals agreeing to 95 ppm cannot both be wrong by the same
amount by chance. That same instrument then measured the sensor at **+7000 ppm**
above nominal — **73× larger** than the agreement between the two clocks.

A crystal is ±0.005% (50 ppm). It cannot produce 7000 ppm. **The sensor's RC
oscillator is the source.**

**The mode difference clinches it.** If the board clock were at fault, both modes
would be off by exactly the same amount. They differ by 700 ppm.

**Consequence: every figure in the §3.1 timing matrix is trustworthy.** DWT on
this board is confirmed correct by three independent routes — the 1 kHz
calibration (95 ppm), the frame period, and the SCL clock (132 ppm from nominal).

### ⚠️ The "one tick" model is an approximation, not a mechanism

The old text claimed a single internal tick shared by both modes. Precise
measurement shows this is wrong.

```
period ratio 8×8 / 4×4  =  3.99721,  not 4.00000   (-0.070%)

internal measurement period (streamcount steps by 1 at 4×4, by 4 at 8×8):
  4×4:  16.5514 ms   →  60.4178 ticks/s
  8×8:  16.5399 ms   →  60.4599 ticks/s
  difference: 698 ppm
```

698 ppm from the ratio and 700.2 ± 11.3 ppm from the A-B-A experiment are the
same number reached two ways. **The sensor sets a different internal measurement
period in each mode.**

`8×8` still consumes 4 internal measurements per output frame — that part holds,
and is independently confirmed by the `anomaly` counter (§4, 62 consecutive
frames, zero anomalies) and by the jitter scaling (√4 = 2×, §3.1e). What is wrong
is only the claim that the tick *period* is identical across modes.

## 3.1c Bus utilisation — measured directly, 10 Sep 2026

Previously inferred from the dose-response experiment (§3.5). Now measured.

| Mode | Bus busy | **Bus idle** | SD | Bytes/transaction |
|---|---|---|---|---|
| 4×4 | 27.197% | **72.803%** | 0.124 | 199 (196 payload + 4 overhead) |
| 8×8 | 20.107% | **79.893%** | 0.319 | 583 (580 payload + 4 overhead) |

The 4 overhead bytes are visible in the decoded capture: address-write,
register address (2 bytes), address-read.

**SCL runs at 399.947 kHz** (calibration applied), 132 ppm from the 400.000 kHz
setting — so the nominal figure may be quoted as-is. It is exact because
PCLK1 = 42 MHz, fast mode duty 2, period = 3 × CCR × Tpclk1, giving CCR = 35
exactly. [จำมา — RM0383 formula not re-opened this session]

**INT to I2C START: 6.61 µs, SD 0.039**, identical in both modes. The MCU
responds ~2500× faster than the frame period, so INT is nowhere near being the
bottleneck (see TRAP #15).

> Measure to the **START condition** (SDA falling while SCL is high), not to the
> first SCL rising edge. The latter is one clock period later and overstates the
> latency by 39%.

**INT pulse width: 100.4 µs**, identical in both modes despite a 4× difference in
frame period and a 3× difference in read time. The sensor releases INT long
before the MCU finishes its 4512 µs read.

## 3.1d ⚠️ TRAP #20 — the sensor frame rate is NOT stable over time

Measured 10 Sep 2026, same sample rate, same wiring, only time differing:

| Elapsed | 4×4 (ppm over 60 Hz) |
|---|---|
| 0 min | 6038 |
| 47 min | 6946 |
| 57 min | 6911 |
| 75 min | 6890 |
| 92 min | 6844 |

**+908 ppm in the first hour, then a slow decline of ~100 ppm.** 8×8 shows the
same shape (−122 ppm over the same later window).

Within a settled 17-minute window the drift is 46.1 ppm, t = 4.69 — significant.

**Cause not established — deferred, see §8.5 D1** for the designed follow-up.
Temperature is the leading suspect: the sensor uses an
RC oscillator, which has a large temperature coefficient, and rise-then-settle is
the shape of thermal equilibration. **No temperature was measured**, so this is
not proven.

Consequences:
- A frame rate quoted without a timestamp is incomplete.
- Comparing two conditions measured minutes apart requires an interleaved design.
  The naive comparison of the two modes gave 1678 ppm; A-B-A gave 700 ppm. Drift
  accounted for 975 ppm of the apparent difference.
- Phase 5–6 dataset collection will not have a constant frame rate across a
  session. Plan for it.

## 3.1e Instability by timescale — new characterisation

| Timescale | Magnitude |
|---|---|
| per frame (16 ms) | 200 µs edge jitter = 12 000 ppm |
| per minute | ~30 ppm |
| ten minutes | ~46 ppm |
| one hour | ~900 ppm |

Edge jitter is 200 µs at 4×4 and 100 µs at 8×8 — a factor of 2 while the period
differs by 4. Two independent checks agree with this:
- period SD should be √2 × edge jitter if edges are independent: predicted
  283/141 µs, measured 311/152 µs (9% and 7% off)
- √4 = 2 is the signature of random jitter accumulating over 4 internal
  measurements per 8×8 frame — the same conclusion as the streamcount evidence,
  reached from different data

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
| Sensor | **VL53L8CX**; ours is **VL53L8CA**, a noncommercial evaluation part (`[UM3120]` §1.6, §1.1 above) — same family, not an identical chip. Whether ST-trained models transfer to it is Phase 4's question (§5.1e) |
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

**`None` is not an empty scene.** Inspected 3 Sep with a single sample frame — it
contains *more* occupied zones than `Fist`. That single-frame comparison used a
FlatHand figure (155 mm) that turned out to be unrepresentative — see the
corrected, full-dataset numbers in §5.1c below (measured 15 Sep across all 11,061
frames): FlatHand's true median is 278.5 mm, not 155 mm. The occupied-zone point
about `None` (45/64 vs Fist's 16/64) still stands; only the distance figure was wrong.

`None` means *"something is there, but it is not one of the 7 postures"* — an open-set
reject class. That is why it needs **more** subject diversity than any other class,
not less: it must cover each person's scene, idle posture and transitions.

**Decision for Phase 2:** drop `None`, train on **7 classes**. Record the reason in the
thesis; it is a defensible methodological choice, not a shortcut. Collect `None` from
all volunteers in Phase 5 and restore the 8th class there.
*(Amended 5 Oct 2026: with the subject pool fixed at n = 2 (§5.1e-A1), there is no separate
Phase 5 collection. Whether to collect `None` from our two subjects during Phase 4 is
**undecided** — `collect_own_data.py` currently records only the 7 postures `[CODE]`.)*

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
from ST and the comparison stays fair. (Consequence for hand position: §5.1f.)

> ⚠️ **OPEN (20 Oct 2026, §9 #56): is our architecture really the same as ST's?** `train_st_cnn2d.build_model` uses
> `Conv2D(8, 3, padding="same")` `[ตรวจแล้ว: train_st_cnn2d.py]` → Flatten input 4×4×8 = 128 → **4,511 parameters for 7 classes**
> (152 + 4,128 + 231; 4,544 for 8 classes) `[คำนวณ]`. ST's published count is **2,752** (8 classes, above), which matches
> `padding="valid"` exactly (152 + 2,336 + 264 = 2,752; Flatten input 3×3×8 = 72) `[คำนวณ]`. The padding in ST's `.keras` was
> never recorded here (Keras' default is "valid"). Until ST's file is inspected, "same architecture, only the split differs" is
> **unverified**. Possibly relevant to the 3.197 ms vs 1.46 ms gap (§8.5 D2) — untested. Check: read the Conv2D config / `model.summary()`
> of ST's `.keras`.
>
> **Added 20 Oct 2026 (night) `[WEB]` `[คำนวณ]`:** the Hugging Face card CNN2D_ST_HandPosture (commit 6f883ae) lists FLOAT32 weights
> flash **10.75 KiB** = 11,008 B = exactly **2,752 × 4 B**, so ST's shipped model has 2,752 parameters → consistent with `padding="valid"`.
> Strengthens, but does not close, §9 #56 — the `.keras` config itself is still unread. Phase 5.5's A/B/C comparison stays valid **within
> our pipeline**; the thesis must not say "same architecture as ST".

### ⚠️ TRAP #5 — inference cost was overstated by 3.7×

The assistant used **5.34 ms** (from Bartoli 2024's own CNN) for weeks.
The correct figure for ST's shipped model is **1.46 ms** (ST's published number). **Measured on our own board in Phase 3: 3.197 ms** for our 7-class model — see §9 #40 and §8.5 D2.
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

**ST's numeric label codes** (`HAND_POSTURE_DICT` in `st_dataset.py`, from ST's
`utils/handposture_dictionnary.py`) `[CODE]`: None 0, FlatHand 20, Like 21, Love 24,
Dislike 25, BreakTime 27, CrossHands 28, Fist 32. `st_dataset.py` orders class indices by
these codes; `train_st_cnn2d.py` and `my_tof.c` use a different order (§9 #38) — always map by name.

Why we did not simply edit ST's `data_loader.py`: it never records which user a frame
came from, and the advisor's standing instruction is to write our own layer rather than
adapt vendor demo code (§7 item 2).

## 5.1c Real-hardware deployment test — first live test with an unseen hand (15 Sep 2026)

First end-to-end test: gesture → sensor → trained model → prediction, with a real
hand the model never saw during training (the student's own — a stand-in for "user 5").
Full chain: `train_st_cnn2d.py --split production` (trains on all 11,061 frames, no
held-out set, saves `production_model.keras`) feeding `live_predict.py` (new tool,
`pc_tools/live_predict.py`), which reads live `F,`/`S,`/`G,` serial lines, applies the
same preprocessing as training, and reports a majority vote over a 15-frame window.

**Bug caught before it corrupted the test:** the board was still running the 4×4
firmware config left over from Phase 1 (16 zones per line), while the model requires
8×8 (64 zones, matching the training data's spatial resolution) — confirmed directly
by counting comma-separated values in a live `F,` line. Fixed by setting
`MY_TOF_USE_4X4 = 0` in `my_tof.h`, rebuilding, and reflashing. **Lesson: after any
Phase 1 experiment that changed firmware config, verify it's back to the config a new
test actually needs — do not assume.**

**Environment check before testing:** the board was initially propped against a
corner, so half the sensor's field of view hit a nearby wall (~150–320 mm) while the
other half hit open ceiling (~1950 mm) — confirmed from a photo. Moved to open space;
confirmed clear via the minimum distance across a full frame being >1500 mm (>1000 mm
past the 400 mm threshold that matters), safe for testing.

**Corrected fact — true per-class collection distance (measured 15 Sep, new tool
`pc_tools/measure_typical_distance.py`, full 11,061-frame dataset, nearest valid zone
per frame):**

| Class | Median | Mean | p10 | p90 | n |
|---|---|---|---|---|---|
| Like | 241.0 mm | 246.0 mm | 168.0 mm | 337.4 mm | 1,580 |
| Dislike | 257.5 mm | 257.7 mm | 190.0 mm | 323.0 mm | 1,686 |
| Fist | 262.5 mm | 262.7 mm | 191.0 mm | 334.0 mm | 2,274 |
| CrossHands | 271.0 mm | 265.6 mm | 190.0 mm | 332.0 mm | 1,266 |
| Love | 278.0 mm | 272.6 mm | 194.0 mm | 343.0 mm | 1,274 |
| FlatHand | 278.5 mm | 277.4 mm | 185.0 mm | 356.3 mm | 1,738 |
| BreakTime | 301.0 mm | 293.7 mm | 235.0 mm | 347.0 mm | 1,243 |

This **supersedes** the misleading single-frame FlatHand figure in TRAP #18 (155 mm).
All 7 classes cluster in a **240–300 mm** band — ST's volunteers did not use
meaningfully different distances per posture. **270 mm falls inside the p10–p90 band
of all 7 classes simultaneously** (checked directly) and is the recommended single
test distance — no need to vary distance by gesture.

**Live test results, own hand, no prior calibration:**

| Attempt | Distance | Result |
|---|---|---|
| FlatHand | ~20 cm | **100% vote agreement, 98–100% confidence, sustained >100 consecutive frames** once the hand settled |
| Love (single-hand ASL "I love you": thumb+index+pinky extended) | ~20 cm | Consistently misclassified as **FlatHand** (100% agreement, high confidence) — a confusion pair never seen in the ST-4-user subject-independent results, where Love scored 89–98% |
| Love (two hands pressed together instead) | ~20 cm | Reported accurate by the student |
| All 7 gestures | **~27 cm (270 mm)**, positioned carefully | **All recognised correctly** — student's own report after switching to the corrected distance |

**Working hypothesis for the Love/FlatHand confusion (not yet confirmed):** at 8×8
resolution the sensor sees an occupied-zone footprint, not individual fingers. The
single-hand "I love you" shape (three digits splayed in different directions) may
produce a spread-out footprint similar to a fully open palm at this resolution — a
geometric confusion of the same kind already documented for Fist/Dislike/Like/FlatHand
(§8, ASL M/N/T discussion), not a distance artefact: Love's and FlatHand's own
training-data median distances are only 0.5 mm apart, so ST's own data already has
them at essentially the same distance, yet the model still separated them well among
the original 4 users. The confusion appearing only with an unseen hand suggests it is
a generalisation-gap effect, consistent with the ~18-point random-vs-subject-independent
gap already measured (§5.1d below), not a new failure mode.
*(Follow-up 24–25 Sep 2026: the two-hand Love is also position-sensitive — see §5.1e-R and §5.1f.)*

**Practical conclusion:** hand positioning (distance from sensor) is the dominant
source of the "sometimes accurate, sometimes not" experience reported during testing.
Once corrected to ~270 mm, every gesture attempted was recognised correctly with the
student's own hand. This matters directly for the oral exam: a physical distance guide
(a 27 cm ruler or card) handed to whoever tests the system live is a low-cost,
no-advance-calibration-needed way to keep committee-member testing inside the
distribution the model was actually trained on.
*(Refined 24–25 Sep 2026: distance is not the only positional factor — lateral position of
the hand over the sensor also matters, differently per posture, §5.1e-R / §5.1f.)*

**Tools added, all smoke-tested with synthetic data before being handed off:**
- `pc_tools/live_predict.py` — live serial → prediction, majority vote, wrong-resolution warning
- `pc_tools/measure_typical_distance.py` — per-class true distance statistics from real data
- `train_st_cnn2d.py` extended with `--split production` (trains on all data, saves a deployable `.keras` model) and `--equalize-train` (isolates whether a fold's low accuracy is a data-quantity artefact — see §5.1d)

## 5.1d Phase 2 training results — random split vs subject-independent (15 Sep 2026)

Architecture exactly as ST's (§5.1), only the final Dense layer changed from 8 to 7
units (the `None` class dropped, TRAP #18). Trained with `pc_tools/train_st_cnn2d.py`
on the full 11,061-frame public dataset. Every configuration trained **3 times with
different seeds** (42/43/44) to separate training-run randomness from genuine
data/subject effects — a single run per fold gave wildly inconsistent numbers on the
first attempt and was not trustworthy (see correction log, this section informed by
that failure).

**Random split (reproducing ST's own method, TRAP #8 — shuffle, no subject grouping):**

| Seed | Test accuracy |
|---|---|
| 42 | 97.56% |
| 43 | 96.88% |
| 44 | 97.29% |
| **mean** | **97.24% ± 0.34 pp** |

**Subject-independent (leave-one-user-out, 3 seeds per fold):**

| Held-out user | Mean ± SD | Range | n_train |
|---|---|---|---|
| User1 | 68.91% ± 3.34 pp | 65.60–72.27% | 6,477 |
| User3 | 78.31% ± 3.67 pp | 74.08–80.48% | 8,889 |
| User4 | 81.97% ± **9.47 pp** | 72.50–91.43% | 8,890 |
| User2 | 89.71% ± 0.61 pp | 89.08–90.30% | 8,927 |
| **overall** | **79.73% ± 8.63 pp** | — | — |

**Gap vs. random split: 17.51 percentage points** — compare Wang et al. 2023's
98.47% → 91.04% (a 7.43-point gap). Our gap is **2.36× larger**. Plausible reason:
only 4 subjects here vs. presumably more diversity in Wang's setup, and static
posture vs. their dynamic-gesture task may respond differently to subject variation
— this comparison should be stated carefully in the thesis (see §8, "do not compare
accuracy numbers across static and dynamic tasks directly").

**Diagnostic: does User1's low score come from less training data, or genuinely
harder generalisation?** Re-ran subject-independent with `--equalize-train`, which
subsamples every fold's training set down to the smallest fold's size (6,477 — which
is User1's own fold, so it was untouched) before training. User1's fold's accuracy
was unchanged at 65.60% even when User2/3/4's folds lost ~2,400 training frames each.
**Conclusion: the gap is not primarily a data-quantity artefact** — User1 (and to a
lesser extent User3) are genuinely harder for the model to generalise to.

**Confusion pattern — reproducible across all 4 folds, not noise:**

| Group | Classes | Behaviour |
|---|---|---|
| "Hard" | Fist, FlatHand, Dislike, Like | Confused with each other in every fold's top-5 confusion pairs |
| "Easy" | Love, CrossHands, BreakTime | 88–100% per-class accuracy in nearly every fold |

Per-user gap between the two groups' average accuracy: User1 38.2 pp, User3 20.1 pp,
User4 6.8 pp, User2 1.1 pp. **This is the same physical principle already identified
for ASL M/N/T** (§8): Fist/Dislike/Like differ mainly by thumb position, close to the
8×8 sensor's spatial resolution limit; FlatHand's presence in the same confused group
is new information not predicted by that earlier ASL analysis and is not yet fully
explained — plausibly related to the Love/FlatHand confusion found in the live-hand
test (§5.1c), both involving spread-out finger silhouettes.

**Tools:** `pc_tools/train_st_cnn2d.py` (`--split random|subject|both|production`,
`--equalize-train`, `--n-seeds`), `pc_tools/analyze_confusion.py` (prints confusion
matrices with class names and top confusion pairs from a saved `phase2_results.json`).
Both smoke-tested with synthetic data matching the real file format before being
handed off — real npz structure differed from documentation in one place (see
correction log): `glob_data` is a numeric code, not the class name as text; the
class name is read from the folder name instead (`dataset_dir/<ClassName>/log__.../npz/`).

### 5.1d-F Which `production_model.keras` is the deployed one — verified 24 Sep 2026

The repo contains **two** files named `production_model.keras` `[MEASURED]` `dir /s /b`:

| Path (relative to repo root) | Status |
|---|---|
| **`results_phase2_final\production_model.keras`** | **The deployed model.** `convert_to_tflite.py` sets `MODEL_PATH = ...\results_phase2_final\production_model.keras` and writes `production_model.tflite` next to it `[CODE]` (read with `findstr`, 24 Sep 2026). This is the model converted for X-CUBE-AI and running on the board. **Use this for Phase 4.** |
| `pc_tools\results_phase2\production_model.keras` | Provenance unknown `[ยังไม่ได้ตรวจ]` — not referenced by the conversion script. Do not use without first establishing what it is. |

Other result folders at the root: `results_phase2\` (contains **no** `production_model.keras`),
`results_phase2_equalized\` (from `--equalize-train`). Neither is the deployed model.
**Never infer the right model from a folder name** ("_final" happened to be right here; the
proof is the conversion script). Record the model file's sha256 in every Phase 4 run log (§5.1e).

## 5.1e Phase 4 — pre-registration (registered 21 Sep 2026, BEFORE any Phase 4 data was collected)

**Status: PILOT.** n = 2 subjects (Chin + one other person). That is below the n ≥ 3 rule,
so the results are descriptive; no conclusion about transfer may be drawn from them alone.

**Question (H8):** does `production_model.keras` (trained on ST's data only) transfer to
our board, whose sensor is a VL53L8CA rather than ST's VL53L8CX?

| Item | Setting |
|---|---|
| Independent variable | data source: ST held-out users vs. our board |
| Dependent variable | per-subject accuracy (7 classes) |
| Controls | one model, ST's preprocessing (§5.1), 8×8, ~270 mm (inside every class's p10–p90, §5.1c), test-only (no training on our data) |
| Baselines | chance = 14.3% (1 of 7); ST leave-one-user-out per-user 68.91 / 78.31 / 81.97 / 89.71% (mean 79.73%, §5.1d) |
| Unit of inference | the **subject**, not the frame (frames within a take are highly correlated) |
| Subjects | **n = 2** (Chin + 1) — pilot |
| Collection per subject | 7 classes × 2 takes × ~20 s (~300 frames per take at ~15 Hz), `collect_own_data.py` |
| Prediction | see below (written by Claude at Chin's request) |

**Prediction — written by Claude on 21 Sep 2026 because Chin had none and asked Claude to
predict ("คุณทำนายเลย").**
- **P1:** the mean per-subject per-frame accuracy of the two subjects lies in **60–85%**, central guess ~72%, i.e. slightly below ST's leave-one-user-out mean of 79.73%. Subjective probability that the mean exceeds 79.73%: ~30%; that it falls below 60%: ~15%.
- **P2:** for each subject, the mean recall of Love / CrossHands / BreakTime is higher than that of Fist / FlatHand / Dislike / Like (in ST's leave-one-user-out the gap was positive for all 4 users, 1.1–38.2 pp, §5.1d).
- **P3:** for each subject, the 15-frame majority-vote accuracy is ≥ the per-frame accuracy.
- **Reasons:** (i) unseen hardware unit (VL53L8CA vs CX — same family; ST describes VL53L8CA as a superset covering both CX and CH, §1.1, so this is a smaller risk than initially framed) plus unknown acquisition settings (frequency/integration time/sharpener not published) and orientation not yet empirically checked (§5.1e item 6) → mild downward pressure, not a known failure mode; (ii) ST leave-one-user-out is already ~80% with a 69–90% per-user spread; (iii) `production_model` saw 4 users (leave-one-user-out models saw 3) → slightly upward; (iv) Chin's single-hand live test at ~270 mm was correct on all 7 (n = 1, §5.1c) → moderately upward.
- **Weakness on record:** the two earlier numeric predictions (H6, H7) failed when extrapolated from unmeasured shapes (§3.0). Treat the central guess as weak; the interval and P2/P3 are what is being registered.

**Pre-flight checks (do before any subject):**
1. `F,` line carries 64 values (8×8), not 16 (§9 #35). *`collect_own_data.py` checks this automatically.*
2. Status is raw (5 = valid). **Verified in `my_tof.c` 21 Sep 2026** for the current switches (`FAST_READ=1`, §2.1 addendum). *The collector also warns if status looks remapped.*
3. Class order taken from `train_st_cnn2d.py` (§9 #38). **Verified 21 Sep 2026:** `train_st_cnn2d.py`, `live_predict.py` and `my_tof.c` use the same order (Fist, FlatHand, Dislike, Like, Love, CrossHands, BreakTime).
4. Sensor field of view clear of walls/obstacles (§5.1c).
5. One end-to-end dry run with Chin (one gesture, ~5 s). **Done 24–25 Sep 2026 with `live_predict.py`** (Fist, normal output, no 4×4 warning) — this verifies the live-prediction chain, **not** `collect_own_data.py`, which still needs its own dry run (§5.1e-P).
6. **Sensor orientation parity with ST — CONFIRMED SOURCE, not geometrically verified.** Chin opened
   `hand_posture/docs/README_DATASETS.md` on 22 Sep 2026 `[WEB]`. It states in its own words that
   the public dataset was recorded with a specific sensor orientation, shown as a photo of the
   sensor board held at an angle with a diagonal arrow labelled "Up" (toward the upper right, roughly
   where a raised hand's fingers point) and "Down" (lower right) — i.e. it tells the person recording
   a posture which direction to raise their hand relative to the board, not a literal 90°/180° rotation
   of the board itself. A rough photo comparison cannot measure this precisely. **Chin's live test on
   15 Sep (§5.1c, n = 1, all 7 postures correct) is the stronger check** — do the n = 3-per-posture
   repeat test (21 trials) proposed alongside this pre-registration before treating orientation as
   settled; if most trials are correct, orientation is not a live concern.
   Also confirmed from the same page: the recording tool is **STSW-IMG035_EVK ("Gesture EVK")**, used
   with a **"8x8" preset** in its sensor-driver dropdown, and it saves each take as
   `log_<gesture>_<user>_<comment>_..._8x8_<timestamp>/npz/*.npz` under a folder per class — the same
   `<Class>/log__.../npz/*.npz` layout `collect_own_data.py` already writes (§5.1e). **Not found:**
   any published ranging frequency, integration time, or sharpener value — the tool's GUI exposes only
   the "8x8"/"4x4" preset, not these lower-level settings; still open.
   **→ Done 24–25 Sep 2026: 21/21, but with per-posture hand positioning — see §5.1e-R for what this
   does and does not show.**

**Decision rules for the pilot (proposed):**
- report each subject's accuracy separately;
- any subject < 60% → stop and diagnose (status, distance, orientation, sensor settings) before Phase 5;
- both ≥ 69% (the lowest ST user) → consistent with ST's spread; still add a third subject (Phase 5 adds subjects anyway) before claiming transfer;
- otherwise → check preprocessing and orientation first.
*(Amended 5 Oct 2026: no third subject is available — §5.1e-A1. "Before Phase 5" now reads
"before Phase 5.5". A transfer claim cannot be made from n = 2 under any outcome.)*

**Secondary outcomes:** accuracy with the 15-frame majority vote (what the deployed system uses); per-class recall; top-5 confusion pairs; **H9:** Fist / FlatHand / Dislike / Like remain the four lowest-recall classes (§5.1d).

**Statistics (pilot, n = 2):** no rank test is possible with n = 2. Report the two per-subject values, their range and their difference from ST's per-user values. Do not write "significant". **This pilot cannot be cited as evidence in a paper or thesis by itself — it is a pre-flight check, not a result.**

**Publication-quality standard for Phase 5 (write here so it is decided before, not after, data collection):**
- **Minimum n:** at least 8–10 subjects, chosen because ST's own reference dataset used only 4 — matching or exceeding that with a more diverse subject pool is itself a contribution, and 8–10 is roughly the smallest n at which a two-sided Mann–Whitney U comparison against ST's 4 per-user values can produce a subject-matter-meaningful p-value (with 4 vs 4 the smallest attainable two-sided p is ≈0.03, achieved only if every value on one side exceeds every value on the other — not a realistic detection floor; n = 8–10 gives room for a real but partial overlap to still be detectable).
- **Test:** Mann–Whitney U (two-sided) comparing our per-subject accuracies to ST's 4 leave-one-user-out values; report the U statistic, exact or asymptotic p, and an effect size (rank-biserial correlation or Cliff's delta) — not the p-value alone.
- **Error analysis:** every misclassified frame categorised as model error / label error / sensor noise / out-of-distribution (wrong distance, wrong orientation, obstruction) before writing any results section — already required by §5.1e's checklist; restated here because a paper reviewer will ask for it.
- **Full reproducibility log per subject:** firmware commit + switches, model sha256, clock source, minutes since power-on, distance, orientation note, date/time — `collect_own_data.py`'s manifest already captures all of these.
- **Report both numbers, never one:** subject-independent-on-ST's-own-data (79.73%) and cross-sensor-on-our-board — a paper claiming "the model works on our hardware" without reporting both, and without the statistical test above, would not survive review.
*(Amended 5 Oct 2026: this standard is **not attainable** — the subject pool is fixed at n = 2,
§5.1e-A1. It is kept here unedited as the record of what was intended; see §5.1e-A1 for what
replaces it.)*

**Error analysis (before writing results):** per-class, per-subject; show 5–10 misclassified frames; categorise as model error / label error / sensor noise / out-of-distribution (e.g. wrong distance or orientation).

**Log per run:** firmware commit + switch settings (currently 8×8, `USE_INT=1`, `TIMING_MODE=0`, `FAST_READ=1`, timing budget 30 ms), clock (HSE), model file sha256, data checksum, minutes since power-on (§3.1d drift), distance, orientation, subject id, date/time.

**Output and consumer (plan rule):** our-board data saved in ST-compatible layout (`User<N>` folders, ST label codes, raw status) so Phase 5.5 can train on it. Evaluate with `st_dataset.load_dataset()` (verified against ST, has the 100 mm lower bound like `my_tof.c`) and map classes by **name**, never by index (index orders differ between `st_dataset.py` and `train_st_cnn2d.py`, §9 #38).

**Open items from the 21 Sep 2026 web search** `[WEB]` — 2 searches, about 3 of the 6 angles (vendor GitHub, ST case-study page, ST community). Not covered yet: the recording tool's own manual, other papers' datasets, other languages. "Not found" here does **not** mean "does not exist":
- ST's dataset README (https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/hand_posture/datasets/README.md) says the public dataset was recorded with the ST datalog tool **STSW-IMG035_EVK (Gesture EVK)** and "with the following sensor orientation" (figure not visible in the search excerpt).
- ST's acquisition settings (ranging frequency, integration time, sharpener, target order) were **not found**; ST's case-study page says only "frequency aligned with the application and reactivity needed".
- An ST community reply says the dataset was captured with the VL53L8CX and that a **different sensor family, VL53L5CX** (lower resolution, not part of the VL53L8 line), will not give good results with VL53L8-trained models. **This is not a statement about VL53L8CA** — see §1.1 for the correction of an earlier draft that conflated the two. CA/CX/CH are the same family; ST separately describes CA as a superset covering CX and CH. Whether that translates to near-identical *ranging* behaviour is still Phase 4's open, empirical question — but there is no published ST statement suggesting it will not work.
- ST's case-study page describes ST's own dataset there as private (5 users, 7 postures), consistent with §5.1 TRAP #16.

### 5.1e-R Result of pre-flight check 6 — the 21-trial live test (24–25 Sep 2026)

**When:** night of 24–25 Sep 2026. Setup photos were taken 22:46 and 22:56 on 24 Sep (Unix
timestamps in the uploaded file names); the `live_predict.py` session used for the trials started
00:10 on 25 Sep (TensorFlow log timestamp). *An earlier chat message dated this test 23 Sep —
wrong, §9 #46.*

**Setup:**
- Board flat on the table, sensor window facing up, ST-LINK/USB end toward the laptop; hand held
  above the sensor, coming from Chin's side. Chin confirmed this is the same configuration as the
  15 Sep test (§5.1c) `[STUDENT]` and photo.
- Model `results_phase2_final\production_model.keras` (§5.1d-F) via `pc_tools\live_predict.py`
  (15-frame sliding majority vote), firmware 8×8 data mode.
- Distance ~270 mm, kept by an arm/elbow landmark rather than measured each trial (a hand in mid-air
  cannot be taped).
- **Order randomised before the test** with Python `random.seed(42); random.shuffle(CLASSES * 3)`:
  CrossHands, CrossHands, Fist, Love, Dislike, BreakTime, FlatHand, Love, BreakTime, CrossHands,
  Like, Like, FlatHand, Love, Dislike, Dislike, Fist, FlatHand, Fist, Like, BreakTime.
- Between trials the hand was removed from the field of view.

**Reading rule (proposed by Chin, adopted — better than the assistant's first rule, §9 #43):**
a trial's result is the vote label once it has stayed the same for ≈1 s (≈15 printed lines at
~15 Hz) **and** agreement is ≥ 70–80 %; if it never settles within ~8–10 s, record "unstable"
rather than right/wrong. Reason: the vote window is a sliding 15-frame buffer that is **not reset**
between postures (§5.1f), so the first lines after a change still contain the previous posture.

**Result: 21 / 21 correct, all stable.** Each trial showed a sustained run of 100 % agreement
before the hand was withdrawn — 9 to 36 consecutive printed lines (shortest: trial 10, where the
label still never left CrossHands until the hand moved away).

| Posture | Trials | Result | Notes |
|---|---|---|---|
| CrossHands | 1, 2, 10 | 3/3 | — |
| Fist | 3, 17, 19 | 3/3 | — |
| Love (two hands, heart) | 4, 8, 14 | 3/3 | needs the **edge** of the heart over the sensor centre, see below |
| Dislike | 5, 15, 16 | 3/3 | — |
| BreakTime | 6, 9, 21 | 3/3 | trial 6 first attempt **void** — operator performed CrossHands by mistake (model correctly said CrossHands); repeated |
| FlatHand | 7, 13, 18 | 3/3 | trial 13 first attempt: hand slightly too high → label held "FlatHand" but agreement fell to 60 % while single frames alternated Fist/FlatHand; repositioned → 100 % for 17 lines (counted). Most height-sensitive posture |
| Like | 11, 12, 20 | 3/3 | — |

**⚠️ What this result does NOT show — disclosed by Chin after trial 7:** Chin moved his hand, for
**every** posture, to the position where the model recognised it, and then returned to that
remembered position for the repeats. So the test answers:

> "For each posture, is there a **reproducible position** at which the model is reliable?" → **yes, 21/21.**

and **not** the question check 6 was written for:

> "Does **one** fixed orientation/position work for all 7 postures?" → **not tested; the evidence points to no.**

**Love — the clearest case `[STUDENT]`:** with the two-hand heart, the model was correct only when
the **edge** of the heart (not its visual centre) was over the sensor centre; with the heart's
centre over the sensor it gave unstable / wrong predictions. Chin's interpretation: the centre of
the heart presents a flat, palm-like surface, so it looks like FlatHand at 8×8. **Not verified** —
the centre-position log was not saved, so whether the wrong label was specifically FlatHand is
unknown. Consistent with the §5.1c working hypothesis.

**Conclusions allowed:**
- Board orientation is **not** the live concern for Phase 4 (the configuration that worked on 15 Sep
  works again, posture by posture).
- **Hand position within the field of view is** — the model is position-sensitive, differently per
  posture (mechanism in §5.1f).
- n = 1, informed subject, positions found by trial and error → descriptive only; this is **not**
  evidence about transfer (H8).

**Data kept:** terminal excerpts of all 21 trials exist only in the chat (trials 1–3 also as uploaded
text files). Nothing was saved to the repo. If these are wanted as thesis figures, re-run a few trials
and screenshot (§5.1e-P, figures list).

### 5.1e-E First H8 evaluation — User 5 (Chin) only, 9 Oct 2026 23:40

**Run:** `pc_tools\evaluate_phase4.py` on `data\phase4` (14 takes, User 5), model
`results_phase2_final\production_model.keras` sha256 `f4be0cda420393d0…`, preprocessing
`pc_tools\st_dataset.py` (ST rules, 100–400 mm); cross-check against the training code's
preprocessing: same within float32 rounding (max 3.1e-07, normalised units). Manifest: 14 rows,
missing frames 0, brief out-of-window 0, files match. Output folder
`results_phase4\eval_20261009_234054\`. `[MEASURED]` (report text pasted by Chin, this session).

| Metric | User 5 |
|---|---|
| Frames evaluated | 4,220 of 4,220 (none removed by the 100–400 mm rule) |
| **Per-frame accuracy** | **66.3 %** (chance 14.3 %; ST LOSO per-user 68.9–89.7 %) |
| 15-frame vote accuracy | 67.6 % |
| Per-take accuracy | mean 66.3 %, SD 44.6, range 0.0–100.0 (n = 14 takes) — bimodal, not noise |
| Recall | Fist 97.8 · FlatHand 99.3 · Dislike 96.2 · **Like 0.3** · **Love 0.0** · CrossHands 70.1 · BreakTime 100.0 |
| Top confusions | **Love → BreakTime 601** · **Like → Dislike 572** · CrossHands → FlatHand 180 · Like → Fist 30 · Dislike → Fist 23 |

**Against the pre-registration (n = 1 subject — descriptive only, no conclusion):**
- P1 (60–85 %): inside the band (66.3 %), below the central guess of 72 %.
- P2 (Love/CrossHands/BreakTime recall > the other four): **does not hold** (56.7 vs 73.4 %).
- P3 (vote ≥ frame): holds, but barely (+1.3 pp) — the errors are systematic, so voting cannot fix them.
- H9 (Fist/FlatHand/Dislike/Like are the four lowest): **does not hold** — the lowest are Love, Like,
  CrossHands, Dislike.
- Decision rule: 60–68.9 % → "check preprocessing and orientation first". Preprocessing is checked
  (two independent implementations agree; labels verified against `glob_data`; every frame inside
  220–320 mm). Orientation/position is the open item.

**What the pattern suggests (hypotheses, not findings):** the error is concentrated in two postures
and is almost total for each — not a general degradation. That matches the position sensitivity
documented in §5.1e-R/§5.1f: the deployed model (Flatten, no global pooling) recognises a posture
only at the positions it saw in ST's data, and Chin collected at his **natural** position (A2).
- **Love:** Chin's natural Love puts the heart's centre over the sensor (A5). On 24–25 Sep that
  position was "unstable / wrong" but the wrong label was not logged; it is now measured: **BreakTime,
  601 of 601 frames**. Chin's earlier guess (FlatHand) is **not** supported.
- **Like:** correct 3/3 on 24–25 Sep with per-posture positioning, now 0.3 % at the natural position,
  almost all → Dislike. Like/Dislike are vertical mirror images, but Dislike itself is 96 % correct,
  so a simple vertical flip of the sensor cannot explain it (a flip would also turn Dislike into Like).
- To separate "position" from "orientation" without new data: compare the per-class hand footprint
  (mean distance/valid-zone map) of our takes with ST's, and test the model on spatially shifted /
  flipped copies of our frames. Not done yet.
- **n = 1.** The friend (User 6, naive) is needed before saying anything about transfer. Phase 5.5
  (training on pooled data that includes our natural positions) is the expected remedy — to be
  tested, not assumed.

**Diagnosis run 9 Oct 2026 23:56 (`pc_tools\diagnose_position.py`, output
`results_phase4\diag_20261009_235613\`; ST data at `C:\ds\ST_VL53L8CX_handposture_dataset`) `[MEASURED]`:**

Hand centroid (row | col on the stored 8×8 grid, row 0 = first row as stored — physical "up" not verified):

| Posture | ST, range over 4 users | User 5 | Outside ST range |
|---|---|---|---|
| Fist | 4.54–5.59 \| 2.54–3.96 | 4.06 \| 2.61 | row |
| FlatHand | 3.89–4.52 \| 3.29–3.62 | 3.45 \| 2.43 | row, col |
| Dislike | 3.82–4.54 \| 3.03–4.13 | 3.94 \| 3.48 | no |
| Like | 4.64–5.29 \| 2.82–3.93 | 3.55 \| 2.42 | row, col |
| Love | 4.29–4.96 \| 3.36–3.49 | 2.55 \| 3.49 | row |
| CrossHands | 4.50–4.70 \| 3.14–3.54 | 3.28 \| 3.47 | row |
| BreakTime | 3.47–4.27 \| 3.55–3.87 | 3.14 \| 3.57 | row |

Accuracy of User 5 frames after transforming them (exploratory; best transform chosen after seeing data):

| Transform | Fist | FlatHand | Dislike | Like | Love | CrossHands | BreakTime | Overall |
|---|---|---|---|---|---|---|---|---|
| none | 97.8 | 99.3 | 96.2 | 0.3 | 0.0 | 70.1 | 100.0 | 66.3 |
| shift 2 rows "down" (dy=+2) | 90.4 | 44.8 | 0.5 | 82.1 | 69.9 | 100.0 | 100.0 | 69.7 |
| flip up-down | 0.0 | 100.0 | 55.0 | 99.2 | 56.8 | 100.0 | 0.0 | 58.7 |
| rotate 180 | 0.3 | 100.0 | 44.5 | 100.0 | 41.7 | 100.0 | 0.0 | 55.2 |
| flip left-right | 100.0 | 77.6 | 100.0 | 3.6 | 0.0 | 99.2 | 100.0 | 68.6 |

**Reading (n = 1, diagnostic, not a result):**
- Our hand sits **systematically ~1–2 rows nearer row 0 than ST's** in 6 of 7 postures (Dislike is the
  only one inside ST's range). Columns mostly overlap.
- Moving our frames 2 rows toward ST's position recovers **Like (0.3 → 82 %) and Love (0 → 70 %)** but
  breaks **Dislike (96 → 0.5 %)** and FlatHand (99 → 45 %). The model's Like/Dislike decision therefore
  depends strongly on **where** the hand mass sits vertically, not only on its shape: in ST's data Like
  sits lower (4.64–5.29) than Dislike (3.82–4.54); our Like (3.55) sits where ST's Dislike does. Our
  Love (2.55) sits nearer ST's BreakTime rows than ST's Love rows.
- **Orientation is not the explanation:** flipping up-down or rotating 180° fixes Like but destroys
  Fist and BreakTime (→ 0 %). Left-right flip changes little, as expected from the training-time
  horizontal-flip augmentation (sanity check).
- **Preprocessing is not the explanation** (two implementations agree, §5.1e-E above).
- *Revised 10 Oct 2026 (§5.1g-R): for **Love** this is superseded — correctly-trained baselines get Love
  92–100 % on the same frames; the deployed model's Love failure is attributed mainly to the Phase 2
  validation-split bug (§9 #51). The position offset itself is still measured and real.*
- **Working hypothesis for the report:** the transfer failure for Like/Love is a **hand-position
  distribution shift** combined with a **position-sensitive model** (Flatten, no pooling, §5.1f), not a
  VL53L8CA-vs-CX sensor difference. Must be checked on the friend (User 6, naive) before it is stated.
- **Not yet checked:** what physically produces the offset (hand height vs sensor aim vs ST's forearm
  entering the bottom rows) — look at `footprint.png`; whether row 0 is physically "up" on our mount.
- **Implication for Phase 5.5 (to pre-register before training):** add random spatial-shift
  augmentation (±1–2 zones) and include our data; predict that Like/Love recall rises without Dislike
  collapsing. Keeping ST's architecture keeps the comparison fair; global pooling is an alternative
  that changes the architecture and would need to be reported as such.

### 5.1e-F H8 evaluation — User 5 + User 6, 20 Oct 2026 15:24 `[MEASURED]`

**Run:** `pc_tools\evaluate_phase4.py --data-dir data\phase4 --model results_phase2_final\production_model.keras`
(28 takes, User 5 and User 6). Model = the **deployed** one, sha256 `f4be0cda420393d0…`, i.e. the model
**with the §9 #51 validation-split bug**. Preprocessing `pc_tools\st_dataset.py`; cross-check against the
training code: same within float32 rounding (max 3.1e-07). Manifest: 28 rows, missing 0, brief out-of-window
0, files match. ST baseline printed: LOSO 66.3–94.3 % (mean 85.3 %, corrected). Output folder
`results_phase4\eval_20261020_152457\` (report.txt, summary.json, per_take.csv, per_frame.csv, confusion PNGs).
Report text pasted by Chin, 20 Oct 2026.

| Metric | User 5 | User 6 |
|---|---|---|
| Frames evaluated | 4,220 | 4,228 |
| Per-frame accuracy | 66.3 % | 75.9 % |
| 15-frame vote accuracy | 67.6 % | 77.0 % |
| Per-take accuracy (n = 14 takes) | mean 66.3, SD 44.6, range 0.0–100.0 | mean 75.9, SD 28.5, range 24.5–100.0 |

Per-class recall (take 1 / take 2 in brackets for User 6):
- User 5: Fist 97.8 · FlatHand 99.3 · Dislike 96.2 · **Like 0.3** · **Love 0.0** · CrossHands 70.1 · BreakTime 100.0
- User 6: Fist 98.3 (99.3/97.4) · FlatHand 59.1 (31.1/87.1) · Dislike 71.2 (79.1/63.2) · Like 76.5 (76.5/76.5) · Love 62.3 (100.0/24.5) · CrossHands 100.0 (100/100) · BreakTime 63.9 (27.8/100.0)

Top confusions (frames): User 5 — Love→BreakTime 601, Like→Dislike 572, CrossHands→FlatHand 180.
User 6 — Love→BreakTime 225, BreakTime→FlatHand 218, FlatHand→Fist 205, Dislike→FlatHand 131, Like→Dislike 112.

**Determinism check:** the User 5 block is identical to §5.1e-E (9 Oct) in every reported number — the script has no randomness.

**Against the pre-registration (n = 2, descriptive; no significance claim, A1):**
- P1 (60–85 %): **holds** — 66.3 and 75.9 both inside; mean of the two 71.1 % (central guess ~72 %, which §5.1e already calls weak).
- P2 (Love/CrossHands/BreakTime recall > Fist/FlatHand/Dislike/Like, per subject): **does not hold** — User 5 56.7 vs 73.4; User 6 75.4 vs 76.3 (−0.9 pp).
- P3 (vote ≥ frame): **holds, barely** — +1.3 pp (U5), +1.1 pp (U6); errors are systematic, voting cannot fix them.
- H9 (four lowest = Fist/FlatHand/Dislike/Like): **does not hold** for either (U5 lowest: Love, Like, CrossHands, Dislike; U6: FlatHand, Love, BreakTime, Dislike).
- Decision rule: User 5 66.3 % vs ST's lowest 66.33 % (0.03 pp below — rounding-level) → "check preprocessing and orientation first": both already checked (§5.1e-E and its position diagnosis). User 6 ≥ 66.33 % → consistent with ST's per-user spread.

**Caveats:** (1) the deployed model has the §9 #51 bug, so per-class numbers (Love especially) are not a clean test of the sensor/board; (2) one model = one seed (§5.1h-R: per-class results for unseen subjects depend strongly on the seed); (3) within-class take-to-take spread for User 6 reaches 75.5 pp (Love), 72.2 (BreakTime), 56.0 (FlatHand) `[คำนวณ]`, as large as the between-subject gap of 9.6 pp, so the order of the two subjects is not interpretable; (4) n = 2 subjects, 2 takes per posture.

**Hypothesis, not finding:** Like→Dislike is not shared between subjects (U5 Like 0.3 %, U6 Like 76.5 %), which fits a subject-specific hand-position shift rather than a VL53L8CA-vs-CX sensor effect (§5.1e-E working hypothesis). Not tested: footprint of User 6 vs ST (`diagnose_position.py` not run on User 6).

**Not done:** error analysis (categorise misclassified frames from `per_frame.csv`); split-fixed model / multi-seed evaluation on User 6.

### 5.1e-A Amendments after registration — dated, not merged silently into the text above

**A1 — Subject pool fixed at n = 2 (Chin stated 24 Sep 2026 `[STUDENT]`): "only 2 people, no way
to get more."** Chin also stated the advisor knows there are 2 people. Consequences:
- **Phase 5 as planned (4–6 people) will not happen.** Phase 4's data *is* our whole own dataset;
  Phase 4 and Phase 5 collapse into one collection. (Schedule gets ~2–3 weeks shorter.)
- **The publication-quality standard above (n = 8–10, Mann–Whitney U) is unattainable.** With 2 vs 4,
  any comparison with ST's per-user values stays **descriptive**. Do not report a p-value or the word
  "significant" for H8.
- **The two uses of "6 people" are different and must not be mixed:**

  | Use | Groups | What it answers | n that matters |
  |---|---|---|---|
  | H8 comparison (Phase 4) | ours (2) **vs** ST (4), kept separate | does the ST-only model transfer to our board? | the smaller group, **2** |
  | Leave-one-subject-out on pooled data (Phase 5.5) | ST 4 + ours 2 = **6**, pooled | how well does a model trained on everyone else generalise to an unseen person? | **6 folds** |

- **Caveat for the 6-fold LOSO:** in the two folds that hold out one of *our* subjects, "new person"
  and "different sensor unit (CA vs CX)" are confounded — report those two folds separately from
  ST's four.
- **Where the main quantitative claim moves:** Phase 5.5's 6-subject LOSO (vs ST's 4) plus the
  engineering contributions (§8 table) that do not depend on n. The H8 pilot is reported as a
  case study.
- **To do:** say this to the advisor once, in words, at the next normal contact (e.g. the next
  progress report): "n = 2, so the cross-sensor result is a descriptive case study." Not a blocker.

**A2 — Natural positioning protocol for collection (decided 24 Sep 2026):** both subjects perform
each posture **as they naturally would**; no positional coaching, no "trick" positions, and the
subject must **not see live predictions** during collection (prevents auto-correcting toward the
model). Basic controls stay: ~270 mm, clear field of view, 8×8 firmware. Rationale: H8 asks whether
the ST-only model transfers to a normal user on our board, not whether it works after coaching.

**A3 — Chin is an informed subject.** After the 21-trial test Chin knows the per-posture positions,
so his "natural" data may drift toward them unconsciously. Label his data **informed/self**; the
friend is the **naive** subject. Report the two separately. Chin's proposal to *simulate* a naive
user with his own hand instead of recruiting the friend was **rejected as a substitute** (same hand
anatomy → still n = 1; the same reason ST's random split inflates accuracy, §5.1b) — it may be run as
an *extra* condition, never counted as subject 2.

**A4 — One collection, two uses.** Phase 4 data is first used **test-only** on the deployed model
(H8), then reused unchanged as training data in Phase 5.5. No second session is needed.

**A5 — Love position (decided 5 Oct 2026):** collect Love at each subject's natural position (for
Chin: heart centre over the sensor). Do **not** deliberately add edge-position Love — ST's data
already contains whatever taught the current model the edge position, and it stays in the Phase 5.5
pool. **After Phase 5.5, re-test Love live at both edge and centre with the new model** (before/after
comparison against 24–25 Sep: old model edge = correct, centre = wrong), and confirm the edge
position was not lost.

**A6 — Labels come from intent, never from the model.** A take is labelled with the posture the
subject was asked to perform, whatever the model predicts at the time. Moving the hand until the
model agrees and then labelling it would only confirm what the model already knows (circular) —
acceptable for the 21-trial "does a working position exist" check, **not** for training data.

**A7 — Extra takes for training (proposed by the assistant 8–9 Oct 2026; Chin: "เอาที่คุณคิดว่าดีที่สุด"
`[STUDENT]`, read as agreement).** The registered H8 test set stays exactly as registered: **2 takes ×
7 postures × 20 s per subject** in `data\phase4` (User 5 = Chin, User 6 = friend). In addition, each
subject records **4 extra takes per posture** into `data\phase4_extra` (same script, same settings,
`--takes 4 --out-dir ...\data\phase4_extra`). The extras are **used only for Phase 5.5 training** and
never enter the H8 evaluation. Reason: Chin asked whether 2 takes per subject is too little to train
well — it is thin for training, but changing the registered test set after seeing nothing would still
be a protocol change, so the extra data goes into a separate folder instead.
*Revision 9 Oct 2026 (assistant proposed, Chin: "โอเค" `[STUDENT]`):* the 4 extra takes are split
**2 on one day + 2 on another day** (`--fw-note "... extra day1"` / `"... extra day2"`). Reason: more
takes from the same hand in one sitting add little; a second session adds natural day-to-day variation
(posture, seating, fatigue), which is what helps generalisation. Between takes the hand is withdrawn
and re-placed naturally, never deliberately matched to the previous take. Whether 6 takes per subject
is enough is **unknown until Phase 5.5** — if our two LOSO folds come out poor, collect more then.

**A8 — Guided arming and take-level discard (Chin's spec 9 Oct 2026, implemented in
`collect_own_data.py` v4).** Chin wanted only "clean" data: no frames from moving the hand in/out, and
the hand at a controlled distance throughout.
- **Window:** 270 ± 50 mm → **220–320 mm**, measured as the nearest valid zone (status 5/9) per frame.
- **Arming:** after Enter, nothing is recorded; a single live line shows distance, an IN RANGE / TOO
  FAR / TOO CLOSE / NO HAND tag and a hold bar. Recording starts automatically once the hand has been
  inside the window **continuously for 3 s** (Chin chose 3 s over the assistant's 1 s); leaving the
  window resets the count to zero. 60 s without arming → retry/skip prompt.
- **Discard:** during recording, if the hand is outside the window for **8 consecutive frames
  (~0.5 s)** the whole take is discarded (nothing written) and retried automatically.
  Chin first asked for discard on a single out-of-window frame; the assistant pushed back — the
  measured distance jitters by a few mm and hands tremble, so a 1-frame rule would discard nearly every
  take near the window edges. Chin accepted; `--abort-frames 1` remains available.
- **No frame-level filtering:** brief excursions (< 8 frames) are **kept** in the data and counted in
  the manifest column `out_of_window_frames`. The only frame filter anywhere is ST's 100–400 mm rule at
  evaluation time (§5.1), so our data is processed exactly like ST's.
- Recording continues from the same serial stream as arming (no buffer flush in between), so there are
  no stale frames and no gap at the start of a take; `stale_dropped` is therefore 0 in real takes.
- Manifest gained `out_of_window_frames, window_lo_mm, window_hi_mm, hold_s, abort_frames` — manifests
  from older script versions are refused (use a new `--out-dir`).
- Screen text is **English** (9 Oct): the Windows Terminal font has no Thai glyphs, so leading vowels
  (เ แ ไ) were spaced out in screenshots meant for the report.

**A9 — Operator confirms the posture after every take (`collect_own_data.py` v5) + quarantine
procedure (15 Oct 2026).** *Trigger:* the friend's first extra set (15 Oct 00:01–00:10, v4) contained
one wrong posture. v4 checks only distance and frame loss, and a take with no `[WARNING]` was saved
without any prompt (`[CODE]` the `else: action = "save"` branch). Chin noticed only afterwards that
one take was wrong, not which. *Fix:* v5 asks `Posture correct the whole time?  y = save, r = redo,
s = skip` after every take, so the operator must watch all 20 s. Operator rule: `r` for any
posture-**shape** error or doubt; **never** `r` because of hand position (A2, A8 — natural
positioning). Tested on the fake board (10 cases + main loop, incl. an unknown answer being re-asked);
seen live on Chin's board 15 Oct (the question appeared every take). *Quarantine:*
`pc_tools\quarantine_takes.py` **moves** takes (never deletes) to `data\phase4_rejected\` and appends
`quarantine_log.csv` (moved_at, class, user, take, recorded_at, src, dst, reason); dry run by default,
`--apply` to move; refuses to overwrite. The training loader reads only `data\phase4` and
`data\phase4_extra`. Applied once, ~15 Oct 00:19: 14 takes of User 6 (`--since 20261014_233700`),
reason "User6 extra day1: subject reported a wrong posture in one take, take unknown"; the dry-run list
was checked before `--apply` (14 takes, 2 per posture, all User 6, none of User 5, none from the main
set). The **whole** set was removed because guessing which take was wrong could leave the wrong one
in. *Caveat:* `manifest_collect.csv` in `data\phase4_extra` still has those 14 rows (the script only
appends); anything that joins takes to that manifest must also read `quarantine_log.csv`.
`train_phase55.py` does not read a manifest `[CODE]`, and `evaluate_phase4.py` uses the manifest of
`data\phase4`, which was not touched.

**A10 — Friend's extras in one sitting (Chin's decision, 16 Oct 2026 23:27 `[STUDENT]`).** A7's
2 + 2 split is a reasoned preference and §5.1e-P step 5 says "if possible"; it is not a hard rule.
Chin's own 9 Oct evening was 28 takes in one sitting (main 14 + extra 14; board on 19:47, last take
23:11), and he wants equal counts: both subjects 3 sets × 14 = 42 takes. Command: `--user 6 --takes 4
--out-dir ...\data\phase4_extra --fw-note "c88c0d2 not re-verified; extra day1+day2 single sitting
(deviation from A7)"`. *Cost:* the friend's extras carry **no between-day variation** (Chin's carry
two days) → thesis limitation. The operator may pause between postures (the script waits at "Press
Enter to start"). *Status 20 Oct:* **collected** — 28/28 saved in one sitting (file timestamps 18 Oct 2026 21:56–22:16); discarded attempts Dislike 2, BreakTime 1; manifest missing 0 / out-of-window 0 / warnings 0; both subjects now 42 takes. *(17 Oct: not collected yet.)* If the friend cannot sit for 28 takes, fall
back to two sittings on different days, which restores A7. Assistant's first answer on this treated
the split as required and guessed fatigue (§9 #55).

### 5.1e-P Execution plan for Phase 4 (as of 17 Oct 2026)

1. ✅ **Located** (8 Oct 2026, `dir /s /b`): one copy only,
   `C:\Users\gaming\Documents\GitHub\tof-gesture-recognition\pc_tools\collect_own_data.py`.
2. **Firmware check:** `MY_TOF_USE_4X4=0`, `MY_TOF_TIMING_MODE=0`, `MY_TOF_FAST_READ=1`,
   `MY_TOF_USE_INT=1`; the script's 3-second pre-check stops on 4×4 / no frames / remapped status.
3. ✅ **Dry runs (do not count, kept as evidence):**
   - `data\phase4_dryrunk` — 8 Oct, **found the stale-frame bug** (§9 #47): "missing" 291 / 3683, first
     frames of FlatHand copied Fist's 217/215 mm. Keep as bug evidence.
   - `data\phase4_dryrun2` — 8 Oct, after the fix: 75 frames, missing 0, first frames 258–262 mm.
     Old manifest format (cannot be appended to by v4).
   - `data\phase4_dryrun3` — **9 Oct, v4, User 99, 1 take × 10 s: 7/7 saved, each 150 frames
     (15.0 Hz), missing 0, out-of-window 0, medians 266–282 mm; discard triggered on the board twice
     (Fist in an earlier run at ~406 mm, Dislike — Chin pulled the hand out deliberately `[STUDENT]`).**
     Pre-check: 45 frames / 3 s. Screenshot of the summary saved as
     `report_figures\phase4\dryrun3_summary.png` (Thai-label version).
   n = 1 take per posture — enough to show the mechanism works, not a reliability figure.
4. **Chin's collection** (from the repo root):
   `python C:\Users\gaming\Documents\GitHub\tof-gesture-recognition\pc_tools\collect_own_data.py --port COM3 --user 5 --minutes-since-poweron <min> --fw-note "c88c0d2 not re-verified" --out-dir C:\Users\gaming\Documents\GitHub\tof-gesture-recognition\data\phase4`
   — defaults: 7 postures × 2 takes × 20 s, window 220–320 mm, hold 3 s, discard after 8 consecutive
   out-of-window frames (A8). Natural positioning (A2). Then the extras (A7): same command with
   `--takes 4` and `--out-dir ...\data\phase4_extra`. `--fw-note`: the last pushed firmware commit is `c88c0d2` (Phase 3); whether the
   board still runs exactly that build is `[ยังไม่ได้ตรวจ]`.
   **✅ Done for Chin (User 5), 9 Oct 2026** (board powered on 19:47, USB from the PC; `--fw-note
   "c88c0d2 not re-verified"`):

   | Set | Folder | Takes | Time (file timestamps) | minutes-since-poweron | Discarded | Per take |
   |---|---|---|---|---|---|---|
   | Main (H8 test) | `data\phase4` | 14/14 (7 × 2 × 20 s) | 21:11:38 → 21:35:43 | 71 | 1 (Dislike) | 301–302 frames (15.1 Hz), 0 missing, 0 out-of-window, medians ≈ 270–275 mm in the takes shown |
   | Extra day 1 (training only) | `data\phase4_extra` | 14/14 | → 23:11:17 | 134 | 0 | same pattern in the takes shown |

   Board check before the main set: 45 frames / 3.0 s = 14.8 Hz [PASS]. Only some per-take Result
   blocks were seen by the assistant; "0 missing" for every take is supported by the absence of any
   `[WARNING]` prompt (the script warns on any mid-take gap) `[STUDENT]` — confirm from
   `manifest_collect.csv` when the evaluation script is written.
   **Observation (Chin, 9 Oct):** Dislike is awkward to form ("งอมือยาก") — the one discarded take was
   Chin losing the posture mid-take. Possible explanation to check (not a conclusion) if Dislike later
   scores worse than other postures; watch whether the friend finds it hard too. **Update 14–15 Oct:** the friend also found Dislike hard — 3 discarded attempts in his main set (wrist rotation, hand left the window) `[STUDENT]`; Chin lost 1 attempt on 9 Oct and none in his 15 Oct extras. Only n = 2 subjects → a pattern to mention, not a conclusion. Saved Dislike takes are the ones the subject managed to hold, so they are a survivor sample (selection effect → thesis limitation).
4b. ✅ **Chin extra day 2 — done 15 Oct 2026** (v5; board on since ≈ 23:00 on 14 Oct, `--minutes-since-poweron 92`; file timestamps 00:35:20 → 00:48:17; `--fw-note "c88c0d2 not re-verified; extra day2 (v5)"`): **14/14 saved** in `data\phase4_extra`. 1 take auto-discarded (FlatHand take 1: the hand left the window at 2.4 s, 14 frames out). 1 take redone by the operator (BreakTime take 1: 291 frames, 14.5 Hz, **11 frames missing mid-take**, `[WARNING]` → `r`; cause **not determined**, n = 1, Chin was asked what was running and did not answer). Every saved take in the pasted log: 302 frames (15.1 Hz), 0 missing, 0 out-of-window, median 263–274.5 mm. Board check 14.8 Hz `[PASS]`; its `[WARNING] 0 %` only means the hand was not yet in front of the sensor (that check is not recorded). The first answer typed with the Thai keyboard (`ั`) was rejected as "Unknown answer" (§9 #54); nothing was lost. Screenshots `collect_v5_confirm.png` and `frame_loss_warning.png` were requested; capture **not confirmed**. Laptop charging pause: board unplugged ≈ 00:56 15 Oct, replugged 16 Oct ≈ 23:20 (list_ports first showed no ports, then COM3 — cause not diagnosed, probably the cable).
5. **Friend's collection (User 6), naive (A2–A3):**
   - **Main — ✅ 14/14 in `data\phase4`, 14 Oct 2026** (v4; board plugged ≈ 23:00; `--minutes-since-poweron 3`; `--fw-note "c88c0d2 not re-verified"`; first file `log__20261014_231812__User6__take1`; finished before 23:36). Board check 14.8 Hz `[PASS]`, `[WARNING]` median 157 mm (his hand was closer than the window during the unrecorded check). Fist take 1: 302 frames, median 265.5 mm; take 2: 302 frames, median 252.5 mm; both 0 missing / 0 out-of-window. Collection Summary: all 7 postures 2/2 saved; **Dislike had 3 discarded attempts** (Chin: the friend found rotating the wrist hard and the hand left the window `[STUDENT]`), the other postures 0. The assistant saw only the two Fist Result blocks and the Summary — the other 12 per-take Results were not pasted `[ยังไม่ได้ตรวจ]`; confirm from `manifest_collect.csv` when evaluating.
   - **First extra attempt — quarantined** (15 Oct 2026 00:01:52 → 00:10:06, v4): 14/14 saved, Like had 2 discards; Chin reported one wrong posture, which take unknown → all 14 moved to `data\phase4_rejected` (A9).
   - **Extras — ✅ 28/28 in one sitting (A10), 18 Oct 2026 21:56–22:16** (file timestamps) into `data\phase4_extra`. Collection Summary (pasted 20 Oct): discarded attempts Dislike 2, BreakTime 1, others 0. Manifest check: missing 0, out-of-window 0, with warning 0.
6. **Evaluation script — ✅ written (`pc_tools\evaluate_phase4.py`, §9 #50; ST baseline updated to the corrected LOSO 14 Oct); original plan:** load `data/phase4` with `st_dataset.load_dataset()`,
   run `results_phase2_final\production_model.keras`, report per subject: per-frame accuracy,
   15-frame vote accuracy, per-class recall, confusion matrix — mapping classes **by name** (§9 #38).
7. Compare with the decision rules and P1–P3/H9 (descriptive), then Phase 5.5 (§8).

**Figures to capture for the progress report / thesis / defence (Chin's standing request):**

| When | Capture | Used for |
|---|---|---|
| Setup | photo of board + hand position (the 24 Sep photos exist) | Methodology — experimental setup |
| Start of collection | the settings box (Subject, Postures, Distance window, Hold, Recording time, Discard rule) + `Board check ... [PASS]` | Methodology — collection protocol |
| During collection | one live `Distance ... Hold ████░░` line, one `● REC` line, and one take's **Result** block | Methodology — arming / data-quality evidence |
| During collection | one `[DISCARDED]` message followed by `Attempt 2` | Methodology — take-level discard rule (A8) |
| End of collection | the **Collection Summary** table (Saved / Discarded / Skipped per posture) | Methodology — completeness |

**Captured 9 Oct 2026 (stored in Chin's Discord `[STUDENT]`):** `user5_1_settings.png` (settings box +
Board check PASS, command line cropped — the minutes-since-poweron value is not to appear in the next
progress report, Chin's request), `user5_2_take.png` (Fist take 1: Hold 3.0/3.0, REC, Result, SAVED),
`user5_5_summary.png` (Collection Summary 14/14, Dislike discarded 1). Dry-run screenshots
(`dryrun3_summary.png` Thai labels, `dryrun3_english_screen.png`) can stand in for the
`[DISCARDED]` figure, captioned as dry run.
| After collection | `data/phase4/` folder tree + `manifest_collect.csv` opened | Methodology — reproducibility |
| Evaluation | per-subject accuracy vs ST's 4 users (table/chart) | **Results — the main Phase 4 figure** |
| Position sensitivity | two terminal excerpts side by side: Love edge (stable 100 %) vs Love centre (unstable) — **re-run and save; not kept on 24–25 Sep** | **Discussion — the unexpected finding** |
| After Phase 5.5 | same Love edge/centre test with the new model | Results — before/after |

## 5.1f Model properties that explain the position sensitivity — verified from code 24–25 Sep 2026

| Property | Evidence | Consequence |
|---|---|---|
| **No translation invariance** | Architecture `Conv2D(8,3×3) → MaxPool 2×2 → Dropout → Flatten → Dense(32) → Dense(7)`, no global pooling `[CODE]` `train_st_cnn2d.py:176–187` | After one 2×2 pool, `Flatten` feeds a 4×4×8 map to `Dense`, which weights each grid position separately — the same hand shape shifted by 1–2 zones (≈30–60 mm at ~300 mm, §1.1) gives a different input to the classifier |
| **No re-centring in preprocessing** | `st_dataset._preprocess_session` and `live_predict.preprocess` keep the raw zone layout; only background cut + normalisation `[CODE]` | Where the hand lands in the 8×8 grid goes straight into the model |
| **Forced choice among 7** | `None` dropped (TRAP #18); prediction is `argmax` with **no confidence threshold** `[CODE]` `train_st_cnn2d.py:221`, `live_predict.py` | The model always names one of the 7, even when unsure; the printed "มั่นใจ X %" is just the winning softmax value |
| **Vote window not reset** | `deque(maxlen=15)` appended every frame, printed every frame once full `[CODE]` `live_predict.py:127–144` | Output scrolls at ~15 lines/s; first ~15 lines after a posture change mix old and new posture |

**Not decided — options if the position sensitivity must be reduced**, cheapest first:
(1) user practice + a positioning guide for the live demo; (2) train-time positional augmentation
(random shifts); (3) centroid re-centring in preprocessing — must be implemented identically in
training and in the firmware; (4) replace `Flatten` with global average pooling — needs retraining,
re-conversion, and re-measurement of latency and footprint. **Decide from Phase 4 / 5.5 evidence, not
in advance.** In the thesis this is a limitation + discussion point (not a hidden flaw); in the
defence, the presenter performs the demo.

**Open check (not run):** do ST's own users show per-class position offsets? Proposed: for each
class and user in ST's data, the mean (row, col) of the nearest valid zone. If ST's Love already sits
off-centre, the "edge" position is ST's convention; if not, it is specific to our setup.

## 5.1g Phase 5.5a — shift-augmentation ablation: PRE-REGISTRATION (written 10 Oct 2026 00:15, before any run)

**Question:** does random spatial-shift augmentation **alone** (no data from our board in training)
fix the Like/Love failure on User 5 found in §5.1e-E? This separates "make the model position-robust"
from "show the model our hand positions" (the latter is Phase 5.5 proper, after the friend's data).

| Item | Setting |
|---|---|
| Independent variable | training augmentation: **baseline** (random horizontal flip only, as in `train_st_cnn2d.py`) vs **shift** (same flip + random shift dy, dx ∈ {−2…+2} zones, vacated zones filled with background = distance 4000 / signal 0 before normalisation) |
| Training data | ST public dataset, 4 users, 7 classes (None dropped), loaded with `train_st_cnn2d.load_dataset()` — identical preprocessing to the deployed model |
| Architecture / optimiser | `train_st_cnn2d.build_model()` unchanged (Conv2D 8 → MaxPool → Dropout 0.2 → Flatten → Dense 32 → Dense 7; Adam 0.01; batch 32; early stopping on val loss, patience 30, max 1000 epochs) |
| Augmentation form | static: each training frame copied **K = 5** times, each copy with its own random flip (both arms) and own random shift (shift arm only) — both arms see the same number of samples per epoch |
| Validation | the same random 10 % of ST frames in both arms (seeded), un-augmented; used only for early stopping. *Differs from production, which used Keras' last-10 % split — so the baseline arm is re-trained here rather than borrowed from the deployed model.* |
| Test set | `data\phase4`, User 5 only, preprocessed with `st_dataset` exactly as in §5.1e-E. **Never used for training, early stopping or model choice.** `phase4_extra` is not used. |
| Seeds | 42, 43, 44 per arm (3 runs each, 6 models) — report mean ± SD over seeds |
| Metrics | User 5 per-frame accuracy, 15-frame vote accuracy, per-class recall (Like, Love, Dislike in focus), confusion pairs |

**Predictions (Claude, written before running; Chin had none of his own):**
- **P4:** shift-arm mean Like recall > 30 % (baseline arm expected near 0 %, like the deployed model).
- **P5:** shift-arm mean Love recall > 30 %.
- **P6:** shift-arm mean Dislike recall stays ≥ 70 % (the shift test in §5.1e-E showed moving frames can
  break Dislike; augmentation should teach the model not to rely on position).
- **P7:** shift-arm mean per-frame accuracy exceeds the baseline arm by ≥ 5 pp.
- **Sanity check S1:** baseline-arm accuracy on User 5 should land near the deployed model's 66.3 %
  (±10 pp). If not, the re-training pipeline itself changes results and that must be explained first.
- Subjective confidence: P7 ~65 %; P4 ~50 %; P6 ~60 %. Main risk on record: at 8×8 the thumb
  direction may carry too little shape information — removing the position cue could then leave Like
  and Dislike **both** near 50 % instead of fixing Like.
- **Interpretation limits:** n = 1 test subject; seeds give training variance only, not subject variance.
  Frames are correlated, so no frame-level significance test is reported. A positive result says
  "shift augmentation helps for this subject", not "for new users".
- **Deliverable rule (Rule 6):** if P7 holds, shift augmentation becomes part of the Phase 5.5 training
  recipe (with our data added) and is then evaluated by 6-subject LOSO; if it fails, Phase 5.5 proceeds
  without it and this run is reported as a negative ablation.

### 5.1g-R Result — run 10 Oct 2026 00:15–00:33 (`results_phase55a\run_20261010_001458\`) `[MEASURED]`

ST training frames 11,061 (9,955 train × 5 copies, 1,106 val); test User 5, 4,220 frames; settings
exactly as pre-registered.

| Seed | baseline acc | shift acc | baseline Like / Love / Dislike | shift Like / Love / Dislike |
|---|---|---|---|---|
| 42 | 83.3 | 73.4 | 6.6 / 100.0 / 100.0 | 21.9 / 86.2 / 100.0 |
| 43 | 92.7 | 83.6 | 69.9 / 92.5 / 100.0 | 46.2 / 100.0 / 79.4 |
| 44 | 76.6 | 59.4 | 22.8 / 100.0 / 100.0 | 9.6 / 57.0 / 79.6 |
| **mean ± SD** | **84.2 ± 8.1** | **72.1 ± 12.1** | 33.1 ± 32.8 / 97.5 ± 4.3 / 100 ± 0 | 25.9 ± 18.6 / 81.1 ± 22.0 / 86.3 ± 11.8 |

Shift recall for Fist fell to 42.1 ± 22.9 (baseline 90.4 ± 7.1).
**Paired by seed** (same seed, same flips, same validation split in both arms): shift − baseline =
−9.9, −9.1, −17.2 pp; mean **−12.1 pp**, paired t(2) = −4.68, p = 0.043 (two-sided). n = 3 seeds on
one test subject — this is training-run variability only, not evidence about other people.

Checks: **S1 NO** (baseline 84.2 % vs deployed 66.3 %, outside ±10) · P4 no (25.9 %) · P5 yes (81.1 %) ·
P6 yes (86.3 %) · **P7 no (−12.1 pp)**. Per the deliverable rule: shift augmentation (static, ±2 zones,
K = 5) is **not** adopted for Phase 5.5; report as a negative ablation. Do not re-tune the shift size
after seeing this (a new pre-registration would be needed). A plausible reason, untested: ±2 zones
is a quarter of the 8×8 field and destroys position cues that are genuinely informative (Fist dropped).

**S1 failure → a bug in Phase 2 found (§9 #51):** the retrained baselines differ from the deployed model
mainly in Love (92.5–100 % vs 0 %). The two recipes differ in the validation split.
`train_st_cnn2d.py` passes `validation_split=0.1` to `model.fit()`; Keras takes the validation set
from the **last** 10 % of the arrays **before** shuffling — **confirmed 10 Oct 2026 on Chin's PC** (Keras 3.15.1, `keras\src\trainers\data_adapters\array_slicing.py`, `train_validation_split`: *"Split arrays into train and validation subsets in deterministic order. The last part of data will become validation data."*; `split_at = floor(batch_dim × (1 − validation_split))`).
`train_st_cnn2d.load_dataset()` reads `sorted(rglob("*.npz"))`, i.e. class folders in alphabetical
order — BreakTime, CrossHands, Dislike, Fist, FlatHand, Like, **Love** — so the last 10 %
(≈1,106 of 11,061 frames) is almost all **Love** (Love has 1,274 frames, §5.1). Consequences, to be
verified:
- the deployed `production_model.keras` trained on only ≈170 Love frames, and its early stopping
  watched a validation set that was ≈100 % Love → plausibly why Love = 0 % on User 5;
- the Phase 2 subject-independent runs (79.73 %, §5.1d) used the same `validation_split` on
  index-ordered training folds → Love was starved there too; **the H8 baseline numbers are affected**;
- the random-split runs were not affected (`train_test_split` shuffles first).
**§5.1e-E's "position" explanation for Love is therefore withdrawn as the main explanation**; Like's
failure is seed-dependent (6.6–69.9 % across correctly-trained baselines) and remains open.
**Next:** (1) ~~confirm Keras' behaviour~~ done, confirmed; (2) fix `train_st_cnn2d.py` (shuffled or stratified validation split), pre-register, re-run Phase 2
(LOSO + production); (3) re-evaluate H8 with the corrected production model.

## 5.1h Phase 2 re-run with the validation-split fix — PRE-REGISTRATION (written 10 Oct 2026 00:50, before any run)

**Question:** does the class-ordered validation split (§9 #51) explain the deployed model's Love = 0 %
on User 5, and how much did it distort the Phase 2 numbers?

**Single change:** shuffle the training arrays (seeded with the run's seed) **before**
`model.fit(validation_split=0.1)`. Everything else is the original `train_st_cnn2d.py` code, imported
unchanged: `load_dataset`, `build_model`, `augment_flip`, seeds 42/43/44, GroupKFold LOSO, batch 32,
early stopping patience 30. Script: `pc_tools\train_phase2_fixed.py`; output `results_phase2_fixed\`.
The original `train_st_cnn2d.py` and `results_phase2_final\` are kept untouched as the record of the bug.

**Why this and not Phase 5.5a's baseline:** 5.5a's baseline differed from Phase 2 in two ways (random
validation split **and** K = 5 copies), so S1's failure cannot be attributed to the split alone.

**Important prior evidence (§5.1d):** in the original LOSO runs Love was already an "easy" class
(88–100 %) despite the bug — so the bug did not stop the model learning ST-style Love. The hypothesis
is narrower: with only ≈ 170 Love training frames (production) the model's notion of Love was too narrow
to cover Chin's Love.

**Predictions (Claude, before running):**
- **R1 (mechanism check):** the script prints the class mix of the validation set. Original order:
  ≈ 100 % Love. Fixed: each class within ±3 pp of its share of the training data. *Must hold, otherwise
  the fix is not doing what it claims.*
- **R2:** corrected LOSO overall mean is within ±5 pp of 79.73 % (Love was already easy there).
- **R3:** corrected production model on User 5 (evaluated with `evaluate_phase4.py`): Love recall ≥ 80 %.
- **R4:** corrected production model on User 5: per-frame accuracy ≥ 75 %.
- **R5:** Like recall on User 5 stays < 50 % (the Like problem is not explained by this bug).
- Confidence: R3 ~75 %, R4 ~65 %, R2 ~70 %, R5 ~65 %.
- **If R3 fails** (Love still low with the split fixed): the 5.5a baselines' Love success came from the
  K = 5 copies / different epochs, not from the split, and §5.1g-R's attribution must be withdrawn.
- **Limits:** production model = 1 seed (as originally); User 5 = 1 subject. LOSO = 4 folds × 3 seeds.

### 5.1h-R Result — 10 Oct 2026 00:53–01:08 `[MEASURED]` (`results_phase2_fixed\`, `results_phase4\eval_20261010_010753\`)

**R1 holds:** validation mix, original order → **Love 100 %**; shuffled → Fist 18, FlatHand 17, Dislike 16,
Like 14, Love 11, CrossHands 12, BreakTime 12 %. The bug is confirmed on the real data.

**Corrected ST leave-one-user-out (3 seeds per fold):**

| Held-out | Original (bug) | Fixed mean ± SD | Difference |
|---|---|---|---|
| User1 | 68.91 | 66.33 ± 1.33 | −2.58 |
| User2 | 89.71 | 92.39 ± 1.41 | +2.68 |
| User3 | 78.31 | 88.09 ± 2.41 | +9.78 |
| User4 | 81.97 | 94.30 ± 1.40 | +12.33 |
| **Overall** | **79.73 ± 8.63** | **85.28 ± 12.90** | **+5.55** (paired t(3) = 1.64, p = 0.20) |

Mean recall over folds (fixed): Fist 81 · FlatHand 86 · Dislike 82 · Like 84 · Love 97 · CrossHands 91 ·
BreakTime 94. **R2 does not hold** (+5.55 pp, just outside ±5). Seed-to-seed SD per fold fell from up to
9.47 pp to 1.3–2.4 pp — early stopping on a Love-only validation set made the original runs erratic.
**The ST LOSO baseline for H8 is now 85.28 % (range 66.3–94.3), replacing 79.73 %.** The random-split
97.24 % is unaffected (train_test_split shuffles).

**Corrected production model** (`results_phase2_fixed\production_model.keras`, sha256 `639d5453d9ff9717…`,
seed 42, 56 epochs) **on User 5**:

| | Deployed (bug) §5.1e-E | Fixed |
|---|---|---|
| Per-frame accuracy | 66.3 | **68.8** |
| 15-frame vote | 67.6 | 68.9 |
| Fist / FlatHand / Dislike | 97.8 / 99.3 / 96.2 | 70.0 / **6.8** / 100.0 |
| Like / Love | 0.3 / **0.0** | **7.1** / **98.8** |
| CrossHands / BreakTime | 70.1 / 100.0 | 99.2 / 100.0 |
| Top confusion | Love → BreakTime 601 | **FlatHand → Fist 562**, Like → Dislike 422 |

- **R3 holds (Love 98.8 %)** — the deployed model's Love failure is explained by the validation-split bug.
- **R4 does not hold** (68.8 % < 75 %). **R5 holds** (Like 7.1 % < 50 %).
- **New: FlatHand collapsed to 6.8 % (→ Fist)** with the fixed model, after being 99.3 % with the buggy
  one. Together with 5.5a's baselines (User 5 accuracy 76.6–92.7 % and Like 6.6–69.9 % across 3 seeds of
  the same recipe), this shows that **for an unseen subject, per-class results depend strongly on the
  training seed**; a single production model is one draw from a wide distribution.
- **Robust across every model tried so far: Like → Dislike** for Chin's natural Like.
- Implications: (1) evaluate User 5 / User 6 with **several seeds** of the production recipe and report
  mean ± SD, not one model; (2) consider a seed **ensemble** (average the softmax of N models) for
  deployment — N × the measured 3.197 ms inference is still far inside the ~66 ms frame budget — to be
  pre-registered before testing; (3) Phase 5.5 (our data in training) remains the main remedy for Like
  and FlatHand.
- `evaluate_phase4.py` still prints the old ST baseline (68.9–89.7 %, mean 79.7 %) — update its constants
  to the corrected LOSO values before the next report screenshot.

## 5.1i Phase 5.5 — train on ST + our two subjects: PRE-REGISTRATION (written 14 Oct 2026 22:55, before the friend's data exists)

**Question:** does adding our own subjects' data to training make the model work for a person it has
not seen — in particular for our subjects' natural Like (→ Dislike so far) and FlatHand — without
hurting ST's users?

**Data:** ST 4 users (11,061 frames, `train_st_cnn2d.load_dataset`, split-bug fix applied) + User 5 and
User 6, each `data\phase4` (main, 14 takes) + `data\phase4_extra` (28 takes; Chin 2 + 2 days, friend one sitting — §5.1e-A10) ≈ **12,600 frames
per subject** vs ST's ~2,100–4,600 per user. Pooled naively our two subjects would be ~70 % of training.

**Folds — leave-one-subject-out over 6 subjects:**
- held-out **ST user**: train on the other 3 ST users + both our subjects (main + extra); test on that user.
- held-out **our subject**: train on 4 ST users + the other subject (main + extra); test on the held-out
  subject's **main set only** (`data\phase4`, the registered H8 test set). Their own extras are not used
  anywhere in that fold.

**Arms (one change each vs A):**

| Arm | Model | Loss weighting |
|---|---|---|
| **A** | ST architecture (`train_st_cnn2d.build_model`) | none |
| **B** | ST architecture | **subject-balanced**: each training subject gets the same total weight (per-frame weight ∝ 1 / that subject's frame count) |
| **C** | ST architecture with **Flatten → GlobalAveragePooling2D** (position-invariant head) | none |

Everything else as the fixed Phase 2 recipe: horizontal-flip augmentation, shuffle before
`validation_split=0.1`, Adam 0.01, batch 32, early stopping patience 30, max 1000 epochs.
**5 seeds** (42–46) per arm × fold → 3 × 6 × 5 = 90 models. Script `pc_tools\train_phase55.py`, resumable.

**Selection rule (fixed now, so the test subjects cannot pick the method):** the arm with the highest
mean accuracy over the **4 ST-user folds** (20 models each) is selected; if the best is within 1 pp of A,
A is selected (simplest, closest to ST). Our two subjects' folds are then reported for **all** arms, with the
selected arm's numbers as the headline. No other arm, augmentation or hyper-parameter may be added after
seeing any fold result without a new pre-registration.

**Predictions (Claude, before any run):**
- **Q1:** arm A, mean over ST folds is within −3 pp of the fixed Phase 2 LOSO (85.28 %) — adding our data does
  not hurt ST's users. Confidence ~70 %.
- **Q2:** arm A, User 5 fold, mean per-frame accuracy > 68.8 % (the fixed ST-only model on User 5).
  Confidence ~70 %.
- **Q3:** arm A, User 5 fold, mean Like recall > 30 % (ST-only models: 6.6–69.9 %, mean 33 in 5.5a's
  baselines; 7.1 % for the fixed production model). Confidence ~45 % — the main risk is that thumb
  direction is not resolvable at 8×8.
- **Q4:** arm B ≥ arm A on the ST folds (B stops our ~70 % share from dominating). Confidence ~55 %.
- **Q5:** arm C < arm A on the ST folds by ≥ 3 pp (GAP after 2×2 pooling leaves only 8 features).
  Confidence ~60 %.
- **Statistics:** arms compared per subject (6 folds), paired over subjects; report mean ± SD over seeds per
  fold, the per-subject differences, and a paired test across the 6 subjects (Wilcoxon signed-rank,
  exact) with its statistic — n = 6 subjects, so no claim beyond "for these subjects".
- **Report table:** subject × class recall for the selected arm (Wang et al. 2023 Table V template, §10.2).
- **Deliverable rule:** the selected arm is then trained on all 6 subjects × 5 seeds → production
  candidates; ensemble vs single model is a separate, later pre-registration (Phase 5.5b), then
  X-CUBE-AI conversion and on-board timing (Rule 6).

### 5.1i-R Result — real run 20 Oct 2026 (started 17:12 by the TensorFlow log timestamp; finished before Chin pasted the report at 20:38), `results_phase55\` (report.txt, runs.jsonl, summary.json), commit `5c724d9` `[MEASURED]`

**Run:** `python pc_tools\train_phase55.py --st-data C:\ds\ST_VL53L8CX_handposture_dataset --ours data\phase4 data\phase4_extra`
(full paths in the commit). Loaded: User1 4,584 · User2 2,134 · User3 2,172 · User4 2,171 · **User5 12,664** (main 4,220 + extra
8,444) · **User6 12,684** (main 4,228 + extra 8,456) frames. 90/90 models, 0 resumed. Smoke test run first (passed, not committed).
Settings exactly as pre-registered (§5.1i). Seed-level numbers re-computed by Claude from the pasted log: they reproduce the report
table to 0.01 pp `[ตรวจแล้ว: log pasted by Chin 20 Oct 20:38]`.

**Mean ± SD over 5 seeds (per-frame accuracy, %; our subjects tested on their MAIN set only):**

| Held-out | Arm A | **Arm B** | Arm C | B seed range |
|---|---|---|---|---|
| User1 | 69.9 ± 0.9 | 71.4 ± 2.0 | 75.4 ± 2.5 | 68.8–73.5 |
| User2 | 92.5 ± 1.4 | 92.4 ± 1.1 | 84.0 ± 2.8 | 91.3–94.1 |
| User3 | 83.0 ± 4.7 | 87.2 ± 2.4 | 82.2 ± 5.5 | 85.1–91.3 |
| User4 | 91.2 ± 3.1 | 91.6 ± 3.5 | 82.6 ± 3.1 | 85.9–95.1 |
| User5 | 78.8 ± 3.0 | 82.8 ± 5.2 | 81.1 ± 7.6 | 76.4–90.4 |
| User6 | 88.3 ± 5.9 | 86.6 ± 6.0 | 89.8 ± 2.9 | 80.7–94.1 |
| **ST-fold mean (selection)** | 84.17 | **85.61** | 81.05 | |
| mean of 6 | 83.96 | 85.30 | 82.53 | |

**Selection (pre-registered rule):** best ST-fold mean = B, 1.44 pp above A (> the 1 pp tie margin) → **arm B selected**.

**Paired over the 6 subjects (Wilcoxon signed-rank, exact):**
- B − A per subject: +1.44, −0.18, +4.12, +0.40, +4.02, −1.74 pp; mean **+1.35 pp**, W = 5.0, **p = 0.312**. B higher for 4 of 6.
- C − A per subject: +5.48, −8.54, −0.84, −8.52, +2.32, +1.50 pp; mean −1.44 pp, W = 9.0, p = 0.844.
- n = 6 subjects: the smallest attainable two-sided exact p is 0.031. **No arm is shown to differ from A.** B is used because the rule
  fixed in advance says so, not because it was shown to be better. Thesis wording: *"arm B was selected by the pre-registered rule; its
  difference from arm A was not statistically distinguishable for these six subjects."*

**Recall per class, arm B (mean over 5 seeds, %):**

| | Fist | FlatHand | Dislike | Like | Love | CrossHands | BreakTime |
|---|---|---|---|---|---|---|---|
| User1 | 66.2 | 70.5 | **45.9** | 78.1 | 96.9 | 91.3 | 93.8 |
| User2 | 85.0 | 92.6 | 94.9 | 93.1 | 98.2 | 94.4 | 88.1 |
| User3 | 90.9 | 93.5 | 86.4 | 68.5 | 91.9 | 95.8 | 83.1 |
| User4 | 97.8 | 97.4 | 75.2 | 73.8 | 99.7 | 98.3 | 99.2 |
| User5 | 79.4 | 91.0 | 99.6 | **23.1** | 86.7 | 100.0 | 100.0 |
| User6 | **67.5** | 98.7 | 84.6 | 81.3 | 93.1 | 99.8 | 80.8 |

**Pre-registered predictions:**

| | Prediction | Result | Verdict |
|---|---|---|---|
| Q1 | arm A ST-fold mean ≥ 85.28 − 3 = 82.28 | 84.17 (−1.11 pp vs the fixed Phase 2 LOSO) | ✅ holds |
| Q2 | arm A User 5 > 68.8 | 78.8 | ✅ holds |
| Q3 | arm A User 5 Like recall > 30 | **3.2** (arm B 23.1) | ❌ **does not hold** |
| Q4 | arm B ≥ arm A on ST folds | 85.61 ≥ 84.17 | ✅ holds |
| Q5 | arm C ≤ arm A − 3 on ST folds | 81.05 ≤ 81.17 | ✅ holds — **by 0.11 pp only** |

**Reading (for these 6 subjects only):**
- Adding our data: User 5 rises from 68.8 % (fixed ST-only production model, 1 seed, §5.1h-R) to 82.8 ± 5.2 % (arm B); User 6 from
  75.9 % (deployed model **with bug #51**, §5.1e-F — not a clean baseline) to 86.6 ± 6.0 %. ST users barely move (Q1). Caveats: the
  User 5 baseline is one seed vs five here; no split-fixed multi-seed ST-only baseline exists for User 6 (open, §8.6).
- **Like for User 5 is not fixed by more data** (A 3.2 %, B 23.1 %), consistent with the position explanation of §5.1j: Chin's natural
  Like sits where ST's and the friend's Dislike sits. Data volume is not the bottleneck for this class.
- Other weak cells: User 1 Dislike 45.9 %, User 6 Fist 67.5 %, User 1 overall ≈ 70 % in every arm (as in Phase 2).
- **Seed spread is large** (User 6 arm B 80.7–94.1 %; User 5 76.4–90.4 %): one trained model is one draw → the choice of the deployed
  model needs its own rule or an ensemble (5.5b).
- Arm C (GAP, 671 params) is worse on ST's users 2 and 4 by ≈ 8.5 pp but better on User 1 and our two; not selected.
- Unchanged limits: n = 6 subjects; our two folds confound "new person" with "different sensor unit" (§5.1e-A1); our model is not ST's
  exact network (§9 #56).

**Deliverable (Rule 6):** train arm B on all 6 subjects × 5 seeds and **save** the models → 5.5b pre-registration (selection rule /
ensemble) → X-CUBE-AI conversion + on-board timing → A5 Love edge/centre live test.

### 5.1j Error analysis of the H8 evaluation — 20 Oct 2026 (full text: Project Knowledge `claude/ERROR_ANALYSIS_H8_2026-10-20.md`)

Input: `results_phase4\eval_20261020_152457\per_frame.csv` (8,448 frames), `per_take.csv` (28 takes) and the 28 `data\phase4` npz files
(sent as `phase4_for_claude.zip`). Model = deployed (bug #51). n = 2 subjects → descriptive; frames within a take are not independent,
so no frame-level p-values. Method check: the centroid code reproduces §5.1e-E's User 5 table exactly.

- **Errors cluster by take, not frame:** User 6 FlatHand 31.1 / 87.1 %, BreakTime 27.8 / 100 %, Love 100 / 24.5 % (take 1 / take 2).
- **Errors come in long runs:** 72.5 % of wrong frames are in runs ≥ 30 frames (2 s) → the 15-frame vote cannot repair them.
- **Confidence:** User 5's wrong frames are confident (mean 0.893; 69.7 % ≥ 0.9) → a confidence threshold is not a general fix
  (User 6 kept-frame accuracy 75.9 → 86.7 % at 0.9, User 5 66.3 → 61.1 %).
- **Distance:** inconclusive (within-take IQR ≈ 6 mm; direction inconsistent).
- **Hand position — strongest signal:** row offset of each take's hand centroid outside ST's 4-user range vs take accuracy, Spearman
  **ρ = −0.58** (column −0.11; 28 takes, descriptive). Inside ST's row range: 6 takes, mean 90.0 %; outside: 22 takes, 65.9 %.
  **Every** out-of-range take is offset toward row 0, for both subjects; User 6 good-vs-bad take pairs 3/3 in the same direction.
  Exceptions: Fist, BreakTime (U5) and FlatHand (U5) tolerate 0.3–0.6 rows of offset.
- **Reading:** a shared direction for two people points to **setup geometry** (how the board sits relative to the hand) rather than
  individual habit — not tested. Whether row 0 is physically "up" is still unverified.
- **Implications:** hand re-centring is the best-supported 5.5c candidate; a position check in `collect_own_data.py` for any future data
  (protocol change → pre-register); a positioning guide for the demo (§5.1f option 1).

### 5.1k Literature / web update — 20 Oct 2026 (full text: `claude/LITERATURE_UPDATE_2026-10-20.md`) `[WEB]`

Coverage: ~16 searches + 10 page fetches; Rule 3 angles 5 of 6 fully, citation-chasing partial. "Not found" = not found by these
searches. No clone, no download.
- **STSW-IMG050** (ST, active): AI hand-posture package for VL53L7CX/VL53L8CX on NUCLEO-F401RE; **no dataset or pre-trained model**
  listed; workflow = define postures, collect from multiple users, retrain with the model-zoo script — the same thing Phase 5.5 does.
- **ST moderator (ST Community, 18 Nov 2024):** the normalisation constants 295/196, 281/452 were computed on ST's public dataset and
  should be recomputed for a new dataset → round-2 candidate (needs firmware parity, §5.1f).
- **Hugging Face card:** 10.75 KiB = 2,752 × 4 B → ST's model is the `padding="valid"` network (§5.1 note, §9 #56). MACs: ours
  13,536 vs ST's 7,744 = **1.75×**; measured time ratio 3.197 / 1.54 = **2.08×** → padding is the leading, **untested** D2 candidate.
- **Oberweger, Wohlhart, Lepetit (arXiv 1502.06807):** fixed-size metric cube centred on the hand's centre of mass, normalised, to be
  invariant to distance — supports Chin's 20 Oct idea (use distance to recover physical hand size) and re-centring; shown on
  high-resolution depth cameras, not 8×8.
- **GlobalAveragePooling2D** is listed in ST Edge AI Core's Keras operator page; not confirmed for X-CUBE-AI 10.2.1 via `.tflite`.
  Moot unless arm C is revived.
- Nothing new on a VL53L8 hand-posture paper or dataset after Sep 2026; no source on per-subject loss weighting for ToF LOSO (arm B is
  our own experiment).

### 5.1l Phase 5.5b + 5.6 — deployed model and new-day test: PRE-REGISTRATION (written 20 Oct 2026 22:05, before any new-day data exists)

**Status of inputs:** final models trained 20 Oct 2026 21:18–21:32 with `pc_tools\train_phase55_final.py` (arm B, all 6 subjects,
36,409 frames, seeds 42–46, epochs 83/68/87/85/76, 4,511 params each), folder `results_phase55_final\` (models + `manifest.json`
with sha256), commits `e1f08ee` (models) and `b0bcc01` (script). No accuracy exists for them (no held-out subject). Their val_loss
was deliberately not displayed and is **not** used for any choice below. Chin was asked for his own predictions and deferred to
Claude ("เอาทางที่ดีที่สุด ชัวร์สุด ละเอียดสุด", 20 Oct 22:03) `[STUDENT]`.

**Questions:** (1) for our two (known) subjects on a new day, how accurate is the final recipe, and is User 5's Like fixed?
(2) does a 5-model ensemble beat a single model? (3) does the converted model behave on the board as on the PC?

**Deployed model — decided now:**
- **Primary: ensemble E5** = one Keras model whose output is the mean of the 5 models' softmax outputs (single network for X-CUBE-AI).
- E5 is deployed **if and only if** all hold on the board: X-CUBE-AI conversion and build succeed; **total per-frame work** (sensor
  read + preprocessing + inference + decision, DWT, HSE, 8×8 data mode) **≤ 33 ms** (half the 66.17 ms frame period); and board vs
  PC predicted class agree on **200 / 200** samples (Phase 3 used 50).
- Otherwise: **single model seed 42**, fixed now, with the same 200-sample agreement check.
- This choice does **not** depend on the new-day results. E5 is deployed whatever B3 shows, if it fits.

**Test set (Phase 5.6):** `data\phase56`, `collect_own_data.py` v5, same protocol as the H8 main set: 7 postures × 2 takes × 20 s per
subject, window 220–320 mm, hold 3 s, discard after 8 consecutive out-of-window frames, natural positioning, **no live predictions
shown** (A2), operator confirms posture every take (A9). Collected on a **day different from every earlier collection**; log
minutes-since-power-on and `--fw-note`. User 5 alone first is allowed; User 6 when available. **`data\phase56` is never used for
training, early stopping, hyper-parameters, model choice or normalisation constants — for any later phase.** If it is ever used
that way, a new test set must be collected.

**Models evaluated on `data\phase56` (same preprocessing as `evaluate_phase4.py`, classes mapped by name):**
1. Baseline 1 — ST-only fixed production model `results_phase2_fixed\production_model.keras` (sha256 `639d5453d9ff9717…`, §5.1h-R).
2. Baseline 2 — each of the 5 single arm-B models; reported as mean ± SD over the 5.
3. E5 (primary).

**Metrics per subject:** per-frame accuracy, 15-frame vote accuracy, per-class recall, top confusion pairs, per-take accuracy.

**Predictions (Claude, before any data):**
- **B1:** User 5 Like recall, mean of the 5 single models, **> 50 %** (23.1 % when User 5 was unseen, §5.1i-R). Confidence ~55 %.
  Main risk: Chin's natural Like sits where Dislike sits in other people's data (§5.1j).
- **B2:** E5 per-frame accuracy **≥ 85 %** for each subject. Confidence ~60 %.
- **B3:** E5 ≥ the mean of the 5 single models, for each subject. Confidence ~75 %.
- **B4:** E5 ≥ Baseline 1 (ST-only) by **≥ 10 pp** for User 5. Confidence ~70 %.

**Statistics:** n = 2 subjects (known users), 2 takes per posture → **descriptive only**; no p-values (frames within a take are
correlated). Report per-subject numbers, the per-take spread, and the 5-model SD. These numbers describe **known users on a new day**,
not new people — the new-person figure stays the LOSO 85.3 % (§5.1i-R).

**Live check after deployment (A5):** Love, two-hand heart, **5 trials at the edge position and 5 at the centre position**, order
randomised with `random.seed(55); random.shuffle(["edge"]*5 + ["centre"]*5)`; reading rule of §5.1e-R (label stable ≈ 1 s and
agreement ≥ 70–80 %, else "unstable"); log every trial. Reference: deployed old model, edge correct / centre wrong (24–25 Sep).
Descriptive (n = 1 subject).

**Deliverable (Rule 6):** the deployed model running on the board, its measured per-frame timing, footprint, the 200-sample
agreement, the new-day accuracy table, and the Love live check. If B1 fails badly (Like < 30 %), round 2 (5.5c, re-centring) is the
next pre-registration, evaluated by the same 6-subject LOSO and then on `data\phase56`.

### 5.1m Step 2.3 result — X-CUBE-AI conversion of E5 (26 Oct 2026, 15:12–15:54) `[MEASURED]`

**Tool:** STM32CubeMX with X-CUBE-AI **10.2.1** (runtime lib v10.2.0, `NetworkRuntime1020_CM4_GCC.a`), which calls
**ST Edge AI Core v2.2.0** (`stedgeai.exe`). Project: the separate conversion project
`C:\Users\gaming\Documents\GitHub\tof-gesture-recognition\ai_conversion_project\tof_ai_conversion\tof_ai_conversion.ioc`
(STM32F411RETx / NUCLEO-F411RE; located with `dir C:\Users\gaming\*.ioc /s /b`). No migration dialog appeared. A
"Non optimal boards parameters … automatically fix?" dialog was answered **No** (this project is never flashed; change one thing only).
Model file: `results_phase55_final\deploy\E5.tflite`. Settings left exactly as found from Phase 3: name `network`, TFLite,
STM32Cube.AI MCU runtime, compression **None**, optimisation **Balanced**. The model previously in the project was the Phase 3
`production_model.tflite` — `.ai\network_production_model.tflite_c_info.json` dated 21 Sep 2026 `[ตรวจแล้ว: dir 26 Oct]`; its
CubeMX summary: RAM 1.48 KiB, Flash 32.39 KiB, 14,744 MACC `[ตรวจแล้ว: screenshot]`.

**Analyze** (42.1 s; `network_analyze_report.txt`, console text pasted by Chin) `[ตรวจแล้ว]`:

| Item | Value | Note |
|---|---|---|
| model hash | `0xb3193840e74ed083a5f621d5790b9e6a` | identifies the converted file |
| params | **22,556** | = 5 × 4,511 + 1 (the ÷5 constant of the averaging layer); Keras counts 22,555 |
| input / output | f32(1×8×8×2) 512 B / f32(1×7) 28 B, both inside activations | |
| MACC | **73,762** | each branch 14,744 (conv 9,224 + relu 512 + pool 512 + dense 4,128 + 32 + dense 231 + softmax 105) = the Phase 3 model's 14,744 exactly; + 4 adds 28 + scale 14 |
| weights (ro) | 90,276 B = **88.16 KiB** | +52 B (+0.1 %) vs the float model |
| activations (rw) | 1,736 B = 1.70 KiB | |
| runtime library | Flash 24,300 B (23.73 KiB), RAM 9,740 B (9.51 KiB) | |
| **total** | **Flash 114,576 B = 111.89 KiB** of 512 KiB · **RAM 11,476 B = 11.21 KiB** of 128 KiB | CubeMX's panel shows 112.61 KiB Flash — 0.72 KiB gap **not explained** `[ยังไม่ได้ตรวจ]`; the build's own size output (step 2.4) is authoritative |
| compute split | Conv ≈ 69 % of MACC (5 × 10,248 ops), Dense ≈ 28 % | where to cut if speed is ever needed |

Prediction written before Analyze (Claude): Flash 88–100 KiB. Weights matched (88.16); **total missed** (111.89) because the
runtime library was left out of the estimate. Conclusion unchanged: fits easily.

**Validate on desktop** (the C model compiled for the PC and compared with the reference; not the board):

| Run | Inputs | Reference | n | acc | rmse | Class coverage |
|---|---|---|---|---|---|---|
| 1 | random, seed 42, range (0, 1) | the `.tflite` itself | 10 | 100 % | 6e-09 | **all 10 in class 1 (FlatHand)** — weak test |
| **2** | `board_check_inputs_200.npy` (200, 8, 8, 2) f32, range −0.62…18.90 | `board_check_E5_probs_200.npy` (PC/tflite outputs, §5.1l build) | **200** | **100 %** | **1.4e-08** (C vs our file) · 6e-09 (tflite vs our file) · 1.5e-08 (C vs tflite) | 28 / 29 / 29 / 28 / 29 / 29 / 28 — all 7 classes |

Run 2 shows the C code reproduces the PC model on real frames of every class. **It is agreement, not accuracy** — the 200 frames
come from `data\phase4`, which is training data, and the reference is the model's own output. The 0.061–0.075 ms "duration" in these
logs is the **PC's** CPU time (AMD), never to be quoted as board latency (Rule 5).

**Generate** (15:50): success dialog; `.ioc` saved. New files, all dated 26 Oct 2026 15:50, in
`ai_conversion_project\tof_ai_conversion\X-CUBE-AI\App\`: `network.c` 64,485 B · `network.h` 9,347 · `network_config.h` 1,553 ·
`network_data.c` 3,516 · `network_data.h` 2,331 · `network_data_params.c` 246,738 · `network_data_params.h` 1,765 ·
`network_generate_report.txt` 60,174. `.ai\network_E5.tflite_c_info.json` 80,863 B (Phase 3's: 18,404 B). Runtime library
`Middlewares\ST\AI\Lib\NetworkRuntime1020_CM4_GCC.a` unchanged (20 Sep 2026).

**Prediction for step 2.4, written before any on-board measurement (Claude, 26 Oct 2026):** E5 is the Phase 3 network run five times,
so inference ≈ 5 × 3.197 ≈ **16 ms**; total per-frame work ≈ 16.58 − 3.20 + 16 ≈ **29–30 ms**, i.e. **≤ 33 ms → E5 deployable, with
little margin**. Caveats: 3.197 ms was measured on `production_model` at HSE (§3.1, §9 #40) — same per-branch architecture, so the
5× scaling is reasonable but not exact (layer kernels, cache, call overhead). If the measured total exceeds 33 ms, §5.1l already says:
deploy `seed42.tflite` instead — no new decision.

**Live demo observation, 26 Oct 2026 (PC only, before conversion) `[STUDENT]`:** Chin (User 5) and the friend (User 6) each tried
`pc_tools\live_demo.py` with `E5.keras` and report that every posture was predicted correctly, including User 5's Like, Dislike and Love.
Not counted (no n, no per-posture tally) → anecdotal, not a result. Both are **known users** (in E5's training data), so this does not
contradict the LOSO Like 23.1 % (§5.1i-R, model had not seen User 5). Useful only as: same users, a different day, a new hand placement.
**Caveat for Phase 5.6:** both subjects have now watched live E5 predictions before `data\phase56` exists; A2's reason (no auto-correcting
toward the model) is weakened for both. Mitigation: no live predictions during collection; compare phase56 hand centroids with
phase4's (§5.1j method) to see whether hand placement moved; disclose in the thesis.

**Screenshots taken during step 2.3 (Chin's Win+PrtScn, default names `Screenshot (n).png`, to be zipped at report time):**
CubeMX opened on `tof_ai_conversion.ioc`; X-CUBE-AI panel with the old model (14,744 MACC) + the "Non optimal" dialog; same panel after
"No"; `network` tab settings (old path); `network` tab with `E5.tflite`; Analyze in progress (20 %, shows the `stedgeai` command);
**Analyze result (100 %, ⭐)**; validation inputs/outputs set to the board-check files; Generate success. The three console texts
(Analyze, Validate random, **Validate real ⭐**) were pasted as `.txt` and are the stronger evidence.


### 5.1n Step 2.4 result — E5 on the board (26 Oct 2026, 16:05–17:46) `[MEASURED]`

**Files.** In the main firmware the model code lives in `53L8A1_SimpleRanging\STM32CubeIDE\X-CUBE-AI\App\` (8 files, Phase 3 versions
dated 21 Sep 00:06) `[ตรวจแล้ว: dir]`; `STM32CubeIDE\Debug\X-CUBE-AI\App\` holds build objects only. Library: `.cproject` line 68 and
`Debug\objects.mk` line 8 link `NetworkRuntime1020_CM4_GCC` from `../Middlewares/ST/AI/Lib` (same 10.2.0 runtime as the generated code;
the file is `lib…`-prefixed, so `dir NetworkRuntime*` does not find it). Old 8 files backed up to
`C:\Users\gaming\Documents\backup_App_phase3_20261026\`; **all 8** new files copied over (including the generate report, so the folder
never shows an old report next to new code — §5.1m said 7). The compiled `my_tof.c` is
`STM32CubeIDE\Application\User\my_tof.c` (`Debug\Application\User\my_tof.d` line 1); **`53L8A1_SimpleRanging\Inc\my_tof.c` is a stray copy
that is not compiled** — compare and delete later (same family as §2.4's duplicate `live_predict.py`). `main.c` loop:
`my_tof_read_frame → my_tof_send_frame → my_tof_infer → my_tof_send_prediction` (lines 146–150); `app_tof.c`'s ST demo loop is disabled
(`main.c` lines 92, 141). `my_tof.c` needed **no change** for E5: buffers use `AI_NETWORK_*` macros, weights `NULL` (§9 #37), class order
correct (§9 #38), preprocessing = ST rules.

**Build** (Clean + Build, 16:26, 0 errors, 0 warnings): `text 243,340 · data 9,964 · bss 17,252` → **Flash 247.4 KiB (48.3 %)**,
**RAM 26.6 KiB (20.8 %)** (Phase 3: 166.2 / 18.9 KiB). Prediction 230–250 / 24–30 KiB held. Whether the stack/heap reservation is inside
these numbers is `[ยังไม่ได้ตรวจ]`; same method as Phase 3, so the comparison holds. **All files are compiled with `-O0`** (Debug
configuration) — a D2 candidate (§8.5), not changed now (one variable at a time). Device macro still `STM32F401xE` (§9 #36 note).

**On-board check.** Boot: `MY_TOF: init OK (8x8 @ 15 Hz, budget 30 ms, INT/async, … read=direct, status=raw(5=valid), clk=HSE)`,
`CLK,84000000`, `MY_AI: network ready (in=128 floats, out=7 classes, activations=1736 bytes, …)` — 1,736 B = E5's activations, so the
board runs E5 `[ตรวจแล้ว: PuTTY screenshot 16:35]`.

**Per-frame timing** (`LAT` lines, DWT, E5, F411RE @ 84 MHz HSE, X-CUBE-AI 10.2.1, `-O0`; log `C:\Users\gaming\Documents\e5_lat_20261026.log`;
n = 22 windows × 30 inferences):

| Stage | mean ± SD (µs) | min–max of window means | Phase 3 (production_model) |
|---|---|---|---|
| sensor read | 13,254 ± 0 | 13,254 | 13,254 |
| preprocess | 124 ± 2 | 123–128 | 126 |
| **inference** | **15,925 ± 35** | 15,823–15,940 | 3,197 |
| decision | 3 ± 1 | 2–6 | 2 |
| **total** | **29,306 ± 36** | 29,202–**29,324** | 16,580 |

Total ≤ 33,000 µs → condition met, margin 3.7 ms; 44 % of the 66.17 ms frame period. Inference is 4.98 × Phase 3 for 5.00 × the MACC:
**≈ 216 ns per MACC for both models** → E5 runs exactly as 5 copies of the Phase 3 network. The prediction on record (§5.1m: ≈ 16 ms,
≈ 29–30 ms) held. AI is now 54 % of per-frame work (sensor 45 %); in Phase 3 it was 19 % (sensor 80 %). Limits: each LAT line is a
30-frame mean, so the single-frame maximum is not known `[ยังไม่ได้ตรวจ]` (spread ±35 µs vs 3,676 µs margin); in the last windows the
frame gap grew (47, 106 frames per 30 inferences) and inference fell by ~115 µs — inference time depends slightly (< 1 %) on the input,
cause unknown `[ยังไม่ได้ตรวจ]`.

**Field-of-view episode.** During the timing run 3 corner zones read 240–280 mm with no hand: Chin's own **head** (board face-up in front
of him, he leaned over it) `[STUDENT]`. It made the board infer continuously (harmless for timing — cost is content-independent) but would
corrupt hand input. With the board at arm's length and no leaning, 13 frames read 1,469–2,001 mm in every zone. Protocol for demos/tests:
board at arm's length, do not lean over it. *Hypothesis only:* body/head position relative to the board may relate to the shared row-0
offset of both subjects (§5.1j "setup geometry") — not tested.

**Board-vs-PC agreement (the §5.1l 200/200 condition) — method B.** The pre-planned `board_check_inputs_200.npy` route (method A) would feed
already-preprocessed frames to the board and so would **not** test the firmware's preprocessing, the riskiest part. Method B: new firmware
switch **`MY_TOF_PRED_LINE`** in `my_tof.c` (1 = every frame prints `Y,<frame>,<class 0–6>,<p0..p6 %.6f>`, or `Y,<frame>,-1` when the hand is
outside 100–400 mm, `Y,<frame>,-2` on a run error; printed after t3 so LAT is unchanged; `#error` if combined with `MY_TOF_TIMING_MODE = 1`;
boot line shows `pred_line=1`). Old file backed up as `backup_App_phase3_20261026\my_tof_before_predline.c`. New script
**`pc_tools\compare_board_pc.py`**: pairs `F/S/G/Y` by frame number, re-preprocesses the raw frame with **`st_dataset._preprocess_session`**
(the training function itself), predicts with `E5.keras`, compares frame by frame; guided 7 postures × 29 predicted frames. Pass rule fixed
before the run: ≥ 200 compared, 0 class mismatches, 0 range-gate mismatches, 0 run errors. Tested beforehand on a simulated board (PASS;
6 injected mismatches all caught; `pred_line=0` stops the run).

**Result, run `results_phase55_final\board_check_live\run_20261026_174016\`** (report.txt, frames.csv, raw_serial.log), `E5.keras` sha256
`1a2268d3b5ab4fe7…`: frames logged 675 (18 incomplete dropped); **compared 203, class agree 203/203; range-gate mismatch 0 of 675; run
errors 0; max |prob diff| 1.10e-06, median 4.17e-07** (≈ the 6-decimal print resolution) → **PASS**. Prediction (203/203, 0 gate
mismatches, ≤ 1e-5) held. Boot lines were not captured in this run (RESET pressed after the 20 s window); firmware identity rests on the
16:35 / 17:15 PuTTY screenshots (`activations=1736`, `pred_line=1`). Chin = operator, one session, frames correlated within a posture —
this is an agreement check, **not accuracy**. The board-predicted class per prompted posture (Fist 26/29, FlatHand 28, Dislike 28, Like 23,
Love 21, CrossHands 13, BreakTime 29) came from frames with no hold window, a known user, and a subject who had seen live predictions —
**must not be reported as accuracy**; `data\phase56` is the place for that.

**Commit:** `84bfea1` (pushed 26 Oct 2026 17:58, `3fb36f0..84bfea1`). `.gitignore` now ignores `*_SMOKE_TEST/`; an unrelated `dog_vs_cat.docx` was moved out of the repo to `C:\Users\gaming\Documents\`.

**Decision (§5.1l rule, applied as written):** conversion and build ✅ · total per-frame work ≤ 33 ms ✅ (29.3 ms) · board vs PC 200/200 ✅
(203/203) → **E5 deployed**. `seed42.tflite` not needed.


### 5.1o Phase 5.6 amendment A11 — board-side logging during collection (written 26 Oct 2026 ≈ 18:35, BEFORE any `data\phase56` data exists)

**Why.** Chin asked for "the best, surest, most detailed" Phase 5.6 (18:18–18:24). §5.1l measures accuracy on the PC only and relies on
§5.1n's 203/203 to speak for the board. With the board already predicting and timing every frame during collection, logging those outputs
gives on-board accuracy directly, board-vs-PC agreement on every collected frame, and the full per-frame timing distribution — at no change
to what the subject does or sees.

**What changes (logging only):**
- Firmware: new switch `MY_TOF_FRAME_TIMING = 1` in `my_tof.c` → every predicted frame prints
  `Q,<frame>,<sensor_us>,<preprocess_us>,<inference_us>,<decision_us>,<total_us>` (same numbers that feed LAT, not averaged; printed after t3);
  boot line adds `frame_timing=1`. `MY_TOF_PRED_LINE = 1` stays (Y lines).
- `pc_tools\collect_own_data.py` **v6**: a `BoardTap` wrapper records Y/Q lines; each saved take gets `<take>_board.csv` (one row per frame of
  the take: board class / 7 probabilities / 5 timings) next to its `.npz`. Board check prints how many frames had Y/Q lines.
- **Unchanged:** `.npz` content and layout, `manifest_collect.csv` columns, window 220–320 mm, hold 3 s, 20 s takes, discard after 8
  out-of-window frames, the y/r/s confirmation, natural positioning, **no predictions shown** (the screen shows only Y/Q line counts).
  Tested on a simulated board 26 Oct (14/14 takes, row counts match the `.npz` frame counts, manifest header identical to v5).

**Added analyses (on top of §5.1l, which is unchanged):**
1. On-board E5 accuracy per subject: per-frame (board class vs posture label) and 15-frame vote computed from the board's own per-frame classes
   (same rule as `my_tof_send_prediction`: majority over the last 15 frames, "none" counts as a vote).
2. Board vs PC (E5.keras on the same raw frame, `st_dataset` preprocessing): class and range-gate agreement on **every** collected frame.
3. Per-frame on-board timing over all predicted frames: mean, SD, median, p99, **max**, count > 33,000 µs.

**Predictions (Claude, before any data):**
- **B5:** board vs PC class agreement = **100 %** of frames where both predict; range-gate mismatches = 0. Any mismatch → stop and diagnose
  before reporting accuracy.
- **B6:** on-board E5 per-frame accuracy equals the PC E5 accuracy (follows from B5).
- **B7:** per-frame total work: mean 29.3 ± 0.1 ms, **max ≤ 30.0 ms**, **0 frames > 33 ms** (LAT window spread was ±36 µs, §5.1n).
- Statistics as §5.1l: n = 1–2 subjects, descriptive; timing numbers are per-frame counts (no inference beyond this session).

### 5.1p Phase 5.6 result — User 5, new day (26 Oct 2026) `[MEASURED]`

**Collection** (`collect_own_data.py` v6, `--fw-note "84bfea1+frame_timing (uncommitted), E5, v6 logger; phase56 day1; poweron ~14:30 est"`,
`--minutes-since-poweron 247`): 14/14 takes saved 18:41–18:55, **0 discarded**, 301–303 frames per take (15.1 Hz), 0 missing, 0 out-of-window,
medians 267–272 mm; board log Y and Q present for every frame of every take (the board predicted every frame). The 3-s board check showed
Q 0/45 only because no hand was held during it. Love take 1 was confirmed with a Thai-keyboard `ัy` (accepted). Firmware boot line before
collection: `pred_line=1, frame_timing=1` (screenshot 18:33). Data never used for anything but this evaluation (§5.1l).

**Evaluation** `pc_tools\evaluate_phase56.py` → `results_phase56\eval_20261026_190627\` (report.txt, summary.json, per_frame.csv,
confusion_E5_User5.png). n = 4,219 frames, all inside 100–400 mm.

| Model (User 5) | per-frame % | 15-frame vote % |
|---|---|---|
| ST-only baseline (`results_phase2_fixed\production_model.keras`) | 68.2 | 68.1 |
| 5 single arm-B models | 89.0 ± 3.3 (84.2–92.1) | 89.1 ± 3.4 |
| **E5 on the PC** | **93.0** | **92.7** |
| **E5 on the board** (its own Y lines) | **93.0** | **92.7** |

Recall % (Fist / FlatHand / Dislike / Like / Love / CrossHands / BreakTime): ST-only 76.2 / 11.1 / 8.1 / 85.4 / 97.7 / 100 / 99.0 ·
E5 100 / 100 / **69.6** / **81.4** / 100 / 100 / 100 · singles mean 97.2 / 100 / 51.4 / 74.6 / 100 / 100 / 100.
E5 top confusions: Dislike→FlatHand 124, Like→Dislike 112, Dislike→Fist 46, Dislike→Like 13. **Errors sit in two takes:** Dislike take 1
39 % (take 2 100 %), Like take 2 63 % (take 1 100 %); the other 12 takes are 100 %. Voting does not help (92.7 vs 93.0) — errors come in runs.

**Board.** Board vs PC: **4,219 / 4,219** same class, 0 gate mismatches, max |prob diff| 7.4e-07. Per-frame work over all 4,219 frames:
sensor 13,255.1 ± 0.3 µs, preprocess 125.2 ± 3.1, inference 15,822.0 ± 18.2 (max 15,911), decision 2.1, **total 29,204 ± 18 µs, p99 29,274,
max 29,295, 0 frames > 33,000** → closes the "single-frame worst case" item (§5.1n). Inference with a hand present is ~100 µs (0.7 %) shorter
than in the §5.1n LAT run (head in view) — content-dependent, as noted there; cause still unknown.

**Pre-registered predictions:** B1 (singles' mean Like > 50 %) 74.6 ✅ · B2 (E5 ≥ 85 %) 93.0 ✅ · B3 (E5 ≥ singles) +4.0 pp ✅ ·
B4 (E5 − ST-only ≥ 10 pp) +24.8 ✅ · B5 (board = PC 100 %) ✅ · B6 (board accuracy = PC) ✅ · B7 (mean 29.3 ± 0.1 ms, max ≤ 30.0, none
> 33) ✅ (mean 29.20 ms — inside the band by 4 µs). Decision rule: Like ≥ 30 % → **5.5c is not triggered**.

**Hand position (centroid row | col, phase56 vs phase4 main set 9 Oct):** Fist 4.35|2.77 vs 4.06|2.61 · FlatHand 3.52|2.57 vs 3.45|2.43 ·
Dislike 4.07|2.43 vs 3.94|**3.48** · **Like 4.50|2.71 vs 3.55|2.42** · Love 2.59|3.42 vs 2.55|3.49 · CrossHands 3.67|3.36 vs 3.28|3.47 ·
BreakTime 2.98|3.37 vs 3.14|3.57. Like moved **+0.95 row toward ST's range** (4.64–5.29); Dislike moved −1.05 column. **Two explanations are
confounded and cannot be separated with these data:** (a) a different seating (board at arm's length, no leaning — adopted today after the
head-in-view episode, §5.1n); (b) Chin had watched live E5 predictions earlier the same day (§5.1m) and knew about the Like/Dislike position
problem. Like's improvement (23.1 % in LOSO → 81.4 % here) may therefore partly reflect hand placement, not only the model; Dislike's drop
coincides with its column shift. **Added 26 Oct 19:15 `[STUDENT]`: Chin collected all his `data\phase4` / `phase4_extra` takes STANDING, and `data\phase56` SITTING** — so explanation (a) is a real, documented protocol difference (arm/wrist angle), not just board distance. Whether the friend (User 6) stood or sat is `[ยังไม่ได้ตรวจ]`. The direction of its effect on accuracy is unknown (12 of 14 takes were already 100 %); "standing would score higher" is a hypothesis. A standing new-day session would be a **separate, pre-registered condition** (e.g. `data\phase56_stand`), never a replacement for this one. For the live demo, stand as in the training data. **Thesis wording:** report 93.0 % with these two caveats, n = 1 subject, one session, two takes per
posture; it is "a known user on a new day", not a new user (LOSO 85.3 % stays the new-user figure).

### 5.1q A5 Love edge/centre live test with E5 — setup and prediction (written 26 Oct 2026 19:26, before the first trial)

Protocol as §5.1l / §5.1e-A5: two-hand heart Love, 5 trials with the heart's **edge** over the sensor centre and 5 with its **centre** over
the sensor centre; order from `random.seed(55); random.shuffle(["edge"]*5 + ["centre"]*5)` (Python 3.11, computed 26 Oct):
**centre, edge, centre, centre, edge, centre, centre, edge, edge, edge**. Reading rule §5.1e-R (label stable ≈ 1 s and agreement ≥ 70–80 %,
else "unstable" if not settled in ~8–10 s). Tool (changed 19:35 at Chin's request, still before the first trial): **`pc_tools\live_demo.py` v3 `--source both`** — the
**board** predicts (its Y lines drive the display and the vote) and the PC re-runs `E5.keras` on every frame and shows "board = PC a/b".
Vote/arming (220–320 mm, hold 1 s, 15-frame vote) is done by the PC script from the board's per-frame classes. Chin **stands** (as in the training data, §5.1p note). n = 1 subject, descriptive.
Reference (old deployed model, 24–25 Sep): edge correct, centre unstable/wrong (→ BreakTime, §5.1e-E).
**Prediction (Claude):** centre **5/5 Love** (E5 trained on Chin's natural centre-position Love; phase56 Love recall 100 %), edge **≥ 4/5 Love**
(ST data, where the old model learned the edge position, is still in E5's training pool).
**Informal first round (after 19:35, not counted) `[STUDENT]`:** using live_demo v3 `--source both`, Chin reported Love was predicted
correctly "almost 100 % of the time, at every position, centre or edge". No log was kept and nobody else read the screen, so it is
anecdotal and is **not** the A5 result.
**Amendment A12 — recorded A5 (pre-registered 26 Oct 19:58, before any recorded trial; requested by Chin):** the 10 trials are recorded
instead of read off the screen, so the scoring is automatic and auditable.
- Tool: `pc_tools\collect_own_data.py` **v6.1** (adds `--classes` and `--take-notes` only; v6 behaviour otherwise unchanged) with
  `--classes Love --takes 10 --seconds 10 --take-notes centre,edge,centre,centre,edge,centre,centre,edge,edge,edge --out-dir data\a5_love`;
  the take header shows CENTRE/EDGE so the order cannot be mixed up. Arming as in collection (220–320 mm window, **3 s** hold — the collector default; "1 s" first written here was wrong, corrected 20:15 before any analysis), then 10 s
  recorded; F/S/G plus the board's own Y/Q lines per frame (`*_board.csv`). Chin **stands**, board at arm's length.
- Scoring (`pc_tools\a5_eval.py`): per take, over in-range frames, the **board's** most frequent per-frame class is the take's result;
  result = **Love** if that class is Love with **≥ 80 %** of frames; **unstable** if the top class has < 80 %; otherwise the other class.
  Report centre x/5 and edge y/5 (Love only), Love share per take, 15-frame-vote Love share, hand centroid (row, col) per take and per
  position (mean ± SD) as a check that edge and centre placements really differ, compared with Chin's phase4 Love centroid and ST's
  Love rows 4.29–4.96. Also board vs PC (`E5.keras`) on every frame and per-frame timing.
- A discarded/retaken take keeps its take number (and therefore its position). Prediction unchanged: centre 5/5, edge ≥ 4/5.
- The recorded `data\a5_love` set is a test set: never used for training or model selection (§5.1l).
- Limits: n = 1 subject, 5 takes per position, frames correlated → descriptive only. The centre/edge contrast in the recording is only
  as large as Chin's placement — the centroid check reports how large it actually was.

### 5.1r A5 result — Love edge/centre, predicted on the board (26 Oct 2026, 20:07–20:14 collection, 20:16 analysis) `[MEASURED]`

Data `data\a5_love` (10 takes, 150–151 frames each, 0 missing, 0 out-of-window, median distance 261–279 mm; Y and Q lines on every
frame). Analysis `pc_tools\a5_eval.py` → `results_a5\eval_20261026_201630\`. Chin stood; firmware `pred_line=1, frame_timing=1`, E5.
- **Centre: Love 5/5. Edge: Love 5/5.** Every take had **100 %** of its frames predicted Love by the board (1,501/1,501 frames);
  15-frame-vote Love share 100 % in every take. Prediction (centre 5/5, edge ≥ 4/5): **both met**.
- **Board = PC: 1,501/1,501** frames, gate mismatches 0, max |Δp| 7.35e-07. Per-frame work 29,188 ± 19 µs (mean ± SD), max 29,283 µs,
  0 frames over 33 ms.
- **Placement check (hand centroid, per take):** the two placements differed **vertically (row), not horizontally**: centre row
  2.58 ± 0.83 (takes 2.65, 2.49, 2.10, 3.93, 1.75), edge row 3.78 ± 0.36 (4.36, 3.90, 3.55, 3.53, 3.57); columns 3.43 vs 3.45.
  Edge − centre = +1.20 rows (mean), +1.08 (median). Mann-Whitney U = 4, exact two-sided p = 0.095, rank-biserial r = 0.68 (n = 5 + 5).
  Centre take 6 (row 3.93) sits **inside the edge range** — that take was effectively an edge placement. Centre placements match
  Chin's training Love (phase4 centroid row 2.55, col 3.49). Edge placements reach ST's Love rows (4.29–4.96) only in take 2 (4.36).
- **Interpretation:** E5 recognised Love at Chin's natural position and when shifted ≈ 1.2 rows (≈ 1 zone) toward ST's position.
  The old model's centre failure (24–25 Sep) did **not** recur. Caveats: n = 1 subject, 5 takes per position; the placement contrast
  was smaller than intended and not separated statistically (p = 0.095), so this shows robustness over about one zone of vertical shift,
  not over the full range; the old-model reference was an informal screen reading on another day, not a controlled comparison; Love
  was already the easiest posture (phase56 recall 100 %). Position sensitivity of the other postures (Like/Dislike) is untested live —
  that is **A6** (§5.1s).

### 5.1s A6 — 1-zone hand shift, all 7 postures, on the board: PRE-REGISTRATION (written 26 Oct 2026 20:33, before any A6 data)

**Why (Rule 6 chain):** §5.1j's best-supported error cause is hand position; A5 only covered Love. A6 measures, live on the board, how
each posture tolerates a controlled 1-zone shift. Its recorded frames become a **fixed position-shift test set**: a later 5.5c model
(re-centring) can be run on the *same* frames on the PC (`a6_eval.py --extra-model`) for a paired before/after — **report only**;
model selection stays with LOSO (§5.1l). Chin asked for "the best, surest, most detailed" option and for Claude to decide on merit,
not to please him (20:27) `[STUDENT]`; Claude chose 5 positions × 2 takes.
**Design.** Subject User 5 (Chin), standing, board at arm's length, E5 firmware `pred_line=1, frame_timing=1`.
- Positions per posture: **natural, row+1, row−1, col+1, col−1**. Natural = Chin's mean hand centroid for that posture in `data\phase4`
  (§5.1j method); a shift = **1 zone** (≈ 2.8 cm at 270 mm: 2·270·tan 22.5° / 8 = 28 mm, from FoV 45° §2). Directions are named by
  grid row/column, not up/down, because "row 0 = physically up" is still unverified (§5.1j).
- 2 takes per position → 10 per posture, 70 in all; order within each posture shuffled with `random.Random(66 + posture index)`;
  postures in the standard order. Plan file made **once** by `pc_tools\a6_plan.py --user 5` → `data\a6_shift\a6_plan_User5.json`,
  committed before collection (its sha256 is printed in every A6 report).
- Collection: `collect_own_data.py` **v6.2** `--plan` (position guide): the live line shows the hand centroid (mean of last 5 frames)
  against the take's target; the 3 s hold starts only when distance is 220–320 mm **and** the centroid is within **±0.35 zone** of the
  target on both axes; then 10 s recorded. The board's predictions are **never shown** during A6. A take that cannot be armed after
  two 60 s timeouts may be skipped; skips are reported as missing, not retried later. Tolerance is not changed during A6.
- Sessions: two blocks with a 10–15 min rest — block 1 Fist, FlatHand, Dislike, Like; block 2 Love, CrossHands, BreakTime
  (`--classes`), same out-dir. A short practice with the guide in `data\a6_practice` (not analysed) may precede block 1 so Chin learns
  which hand movement raises row/column; he reports that mapping `[STUDENT]` — this also answers §5.1j's "is row 0 up?".
- Scoring (`pc_tools\a6_eval.py`): A5's rule per take (board's top class over in-range frames, ≥ 80 % → that class, else unstable);
  correct = equals the performed posture. Report posture × position (x/2), per position (x/14), per posture (x/10), natural vs shifted
  (takes and mean correct-frame share), achieved shift (Δrow, Δcol from natural, mean per position; worst miss from target), every
  wrong take with its frame counts, board vs PC on every frame, per-frame timing.
- `data\a6_shift` is a test set: never used for training or model selection.
**Predictions (Claude, before data; confidence in brackets):**
- P1 natural ≥ 12/14 correct (70 %). P2 shifted mean correct-frame share < natural's (75 %).
- P3 **row−1** (further toward row 0, away from ST's range — the failure direction of §5.1j) is the worst position, ≤ 11/14 (60 %).
- P4 row+1 (toward ST) ≥ natural − 1 take (55 %). P5 each col shift ≥ 12/14 (55 %).
- P6 most wrong takes are Like or Dislike (60 %). P7 Love 10/10 (65 %, A5 tolerated +1.2 rows).
**Decision rule (deliverable):** if any position scores ≤ 10/14 **or** shifted correct-frame share is ≥ 10 pp below natural, 5.5c
re-centring is pre-registered next (selection by 6-subject LOSO; A6 then re-scored with `--extra-model` as report-only before/after).
Otherwise 5.5c stays optional and the thesis reports E5 as tolerant to 1-zone shifts for this subject.
**Limits:** n = 1 subject (a known user), 2 takes per cell, frames correlated → descriptive only; the guide makes placement more
controlled than real use, so A6 measures sensitivity to a known shift, not everyday accuracy.

### 5.1t Amendments A13 + A14 (26 Oct 2026 21:50, before any A6 or standing data)

**A13 — live A6 with the on-screen guide is abandoned; replaced by a simulated shift (A6-sim).** Practice 21:20–21:44
(`data\a6_practice`, 1 three-second Fist take, **not analysed**): Chin could not reach targets — the guide showed only numbers, the
physical direction of row/col is unknown, and his natural Fist today sat at col ≈ 3.4–3.7 vs 2.61 on 9 Oct (a ~1-zone day-to-day
placement change on its own) `[STUDENT + screen text]`. Claude's own design error: aiming at numbers is hard and makes the posture
unnatural, which would confound "position" with "posture distortion". `a6_plan_User5.json` (commit `343c8c2`) and the v6.2 `--plan`
guide stay in the repo, **unused**; no A6 take was ever recorded.
**A6-sim design.** Take recorded new-day frames, shift the **raw** 8×8 zone arrays (distance, status, signal) by whole zones — vacated
zones set to an invalid status so `st_dataset._preprocess_session` treats them as empty, zones pushed off the grid are lost — then
preprocess with the training function and predict on the PC. Shifts: **none, row±1, col±1** (primary); **row±2, col±2** (secondary).
Rationale: after preprocessing only hand zones carry values (background removed), so a whole-zone shift of the raw frame closely
mimics moving the hand; board = PC is established (§5.1n–r, > 9,000 frames), so PC results stand for the board.
Datasets: `data\phase56` (sitting), `data\phase56_stand` (A14, standing), `data\a5_love` (Love only). Never `data\phase4` (E5 trained
on it). Models: E5 (primary), the 5 singles (mean ± SD), ST-only baseline. Report per posture × shift: per-frame accuracy, 15-frame
vote accuracy, paired frame flips vs unshifted (correct→wrong / wrong→correct), share of hand zones lost off-grid.
Statistics: frames within a take are correlated → descriptive; no frame-level p-values.
Predictions (Claude): Q1 E5's mean accuracy over the four ±1 shifts is ≥ 5 pp below unshifted on each dataset (60 %).
Q2 **row−1** (toward row 0, the §5.1j failure direction) is the worst ±1 shift (60 %). Q3 Like or Dislike has the largest ±1 drop (60 %).
Q4 Love on `a5_love` stays ≥ 90 % at row±1 (65 %, A5). Q5 ±2 shifts are worse than ±1 for every model (80 %). Q6 E5's drop under
shift is ≤ the singles' mean drop (65 %).
Operational definitions (fixed 22:15, before A6-sim was run on real data; script `pc_tools\a6_sim.py`): Q1–Q3, Q5, Q6 are checked on
`phase56` and `phase56_stand` separately; Q3 = posture with the largest mean E5 recall drop over the four ±1 shifts; Q5 = mean accuracy
over ±2 < mean over ±1 for baseline, each single and E5; Q6 = E5's (unshifted − mean ±1) ≤ the singles' mean of the same.
Decision rule (unchanged from §5.1s): any ±1 shift that lowers E5 per-frame accuracy by ≥ 10 pp on either dataset → 5.5c re-centring is
pre-registered next (selected by LOSO; A6-sim and the A6-sim script re-run on it as report-only before/after).
Limits: a simulated shift ignores perspective change and the weaker illumination near the edge of the field (FoI 43.4° < FoV 45°, §2);
n = 1 subject. Optional later physical spot-check: move the **board** 3 cm with a ruler (hand stays at its natural place), Like and
Dislike only, separately pre-registered.

**A14 — standing new-day set `data\phase56_stand` (the separate condition foreseen in §5.1p).** Chin offered to re-collect standing
"to be sure" (21:46) `[STUDENT]`. Same protocol as `data\phase56` except posture of the body: `collect_own_data.py` (v6.2, no
`--plan`), default 7 postures × 2 takes × 20 s, hold 3 s, 220–320 mm, Chin **standing** as in `data\phase4`, board at arm's length,
no live predictions shown. Evaluated with `evaluate_phase56.py --data-dir data\phase56_stand` (same models and rules). It **adds to**,
never replaces, the sitting result (93.0 %). Test set only (§5.1l).
Predictions: Chin — standing scores higher than sitting `[STUDENT, 19:15]`. Claude — S1 E5 per-frame ≥ 90 % (70 %); S2 |standing −
sitting| ≤ 5 pp (60 %); S3 Like's centroid row is closer to its phase4 value (3.55) than phase56's 4.50 was (60 %); S4 board = PC on
every frame (95 %). Caveat carried over: Chin has watched many live predictions today (§5.1p).

### 5.1u A14 result — standing new-day set `data\phase56_stand`, User 5 (26 Oct 2026, 21:52–22:03) `[MEASURED]`

**Collection:** `collect_own_data.py` v6.2 without `--plan`, defaults (7 × 2 × 20 s), `--minutes-since-poweron 441`, Chin standing:
14/14 takes, 0 discarded, 301–302 frames per take (15.1 Hz), 0 missing, 0 out-of-window, medians 269–276 mm, Y and Q on every frame.
The board check showed 0 % in range only because no hand was held. **Evaluation:** `evaluate_phase56.py --data-dir data\phase56_stand
--out-dir results_phase56_stand\eval_stand`, 4,221 frames. (The report header still prints "data\phase56" — hard-coded title text, data
dir is correct; cosmetic bug to fix.)

| Model (User 5) | standing % (frame / vote) | sitting % (§5.1p) |
|---|---|---|
| ST-only baseline | 71.7 / 71.7 | 68.2 / 68.1 |
| 5 singles mean ± SD | 96.0 ± 4.8 / 96.1 ± 4.8 (89.1–99.9) | 89.0 ± 3.3 / 89.1 ± 3.4 |
| **E5 on the board** (= PC) | **98.7 / 98.6** | 93.0 / 92.7 |

E5 recall: 100 / 100 / 99.8 / 100 / **91.2** / 100 / 100 (Fist … BreakTime). Errors: Love→BreakTime 53 frames, Dislike→BreakTime 1;
all in **Love take 1 (82 %)**; the other 13 takes 100 %. Like and Dislike, the sitting weak spots, are 100 / 99.8 %.
Board vs PC **4,221/4,221**, max |Δp| 2.2e-06. Per-frame work 29,196 ± 21 µs, max 29,297, 0 > 33 ms.
Centroids (row | col, standing vs phase4): Fist 4.21|3.21 vs 4.06|2.61 · FlatHand 3.37|3.24 vs 3.45|2.43 · Dislike 3.13|3.84 vs 3.94|3.48 ·
Like 3.74|2.82 vs 3.55|2.42 · **Love 1.53|3.45 vs 2.55|3.49** · CrossHands 3.31|3.45 vs 3.28|3.47 · BreakTime 2.72|3.62 vs 3.14|3.57.
Love sat **1.0 row further toward row 0** than in training and failed toward BreakTime — the old Love failure mode, in §5.1j's failure
direction (per-take centroids not yet extracted; A6-sim will look).
**Standing vs sitting:** E5 +5.7 pp per-frame. Take level (14 vs 14 takes; sitting 12×100, 39, 63; standing 13×100, 82): Mann-Whitney
U = 106, p = 0.52, rank-biserial r = 0.08 → **no detectable difference at the take level**; the frame-level gap comes from 2 bad takes
sitting vs 1 standing. Confounded with order (sitting 18:41, standing 21:52), practice, fatigue and having watched live predictions.
**Predictions:** S1 (E5 ≥ 90 %) 98.7 ✅ · **S2 (|stand − sit| ≤ 5 pp) 5.7 ❌** — Chin's prediction "standing higher" holds in direction ·
S3 (Like row closer to 3.55 than 4.50 was) 3.74 ✅ · S4 (board = PC) ✅. The evaluator's B-lines (written for the sitting set) also hold
except **B7 literally fails by 4 µs** (mean 29.196 ms vs band 29.2–29.4; max 29.297 ≤ 30.0, none > 33 ms): the ±0.1 ms band was too tight;
not a timing problem.
**Thesis wording:** "a known user on a new day: 93.0 % sitting, 98.7 % standing (same model, on the board); difference not significant at
the take level and confounded with session order". New-user figure stays LOSO 85.3 %.

### 5.1v A6-sim result — simulated whole-zone shift (26 Oct 2026 22:14) `[MEASURED, simulated]`

`pc_tools\a6_sim.py` (committed before running) → `results_a6sim\eval_20261026_221509\` (report.txt, summary.json, table.csv,
a6sim_E5.png). Data: phase56 (sitting, 4,219 frames), phase56_stand (4,221), a5_love (1,501). PC prediction (board = PC established).

**E5 per-frame accuracy % by shift** (none / row+1 / row−1 / col+1 / col−1 / row+2 / row−2 / col+2 / col−2):
- sitting: 93.0 / 86.5 / **84.7** / 87.1 / 94.7 / 85.1 / **35.4** / 77.5 / 86.1
- standing: 98.7 / 98.4 / **58.3** / 99.5 / 94.6 / 99.9 / **31.8** / 80.2 / 70.7
- a5_love: 100 / 100 / 92.9 / 99.9 / 99.2 / 100 / 61.2 / 94.3 / 19.5
Singles mean (sitting / standing, none → row−1): 89.0 → 79.2, 96.0 → 56.1. ST-only: 68.2 → 34.1, 71.7 → 32.3.
Hand zones pushed off the grid at row−1: 2.3 % (sitting), 8.1 % (standing) → the row−1 collapse is **not** clipping.
Standing row−1 by posture (E5 recall): Like 100 → **3.5**, CrossHands 100 → **1.5**, Love 91.2 → 36.9, Dislike 99.8 → 74.3, Fist 91.9,
FlatHand and BreakTime 100. Sitting row−1: Like 81.4 → **0.3**; sitting row+1: Dislike 69.6 → 9.6 (standing row+1 Dislike 100).
**Main finding:** a strong **asymmetry** — shifting toward higher rows (toward ST's hand positions) is tolerated even by 2 zones
(standing row+2 99.9 %), shifting toward row 0 fails fast (row−1 −40.4 pp standing, row−2 ≈ 32–35 % on both). This is the §5.1j failure
direction, and it matches a real event: standing Love take 1 (82 %, → BreakTime) had Love's centroid 1.0 row toward row 0 (§5.1u).
Interpretation: E5 learned where hands sit in its training data (ST rows ≥ ~3.5–4.5) and does not generalise below it — a
training-distribution effect, not a sensor limit.
**Predictions:** Q1 sitting ❌ (drop 4.7 pp < 5) / standing ✅ (11.0) · Q2 ✅ both (row−1 worst) · Q3 sitting ✅ (Like 18.5 pp) / standing ❌
(CrossHands 31.9 pp) · Q4 ✅ (row+1 100, row−1 92.9) · Q5 ✅ both · Q6 ✅ both (E5 drop ≤ singles: 4.7 vs 6.2, 11.0 vs 12.9).
**Decision rule: TRIGGERED** (standing row−1 −40.4 pp) → **5.5c re-centring is to be pre-registered next** (§5.1s/§5.1t).
**Important caveat for 5.5c:** integer re-centring makes a model invariant to whole-zone shifts *by construction*, so A6-sim will look
almost perfect for it and is **not** a fair test of 5.5c; it may only be reported as a sanity check. 5.5c must be judged by
6-subject LOSO (new people) and the natural new-day sets (phase56, phase56_stand) — and ideally a physical shift check (move the board
3 cm with a ruler), because real shifts are not whole zones and change the viewing angle.
**Practical (demo / thesis):** until 5.5c, users should keep their hand at or "above" (toward higher rows) the trained position;
which physical direction row 0 is remains unverified — needed for a demo instruction.
Limits: simulated; n = 1 subject; frames correlated → descriptive.

### 5.1w Phase 5.5c — hand re-centring: PRE-REGISTRATION (written 26 Oct 2026 22:35, before any 5.5c run)

**Trigger:** A6-sim decision rule (§5.1v). Chin asked to continue in this chat and for "the best, surest, most detailed" option
`[STUDENT, 22:20]`. Earlier alternatives already tried: shift augmentation (5.5a, −12.1 pp, §5.1g-R) and GlobalAveragePooling (arm C,
ST-fold mean −3.1 pp vs A, §5.1i-R). Re-centring is the remaining §5.1f option and the one §5.1j / Oberweger et al. (§5.1k) support.
**Method (`pc_tools\recentre.py`, one function used for training, testing, PC re-checks; firmware must match it exactly):** on the
preprocessed 8×8×2 frame, hand zones = distance channel not empty (< 2,000 mm before normalisation); integer counts cnt, Σrow, Σcol;
shift **dr = ⌊(4·cnt − Σrow)/cnt⌋, dc = ⌊(4·cnt − Σcol)/cnt⌋** (= ⌊3.5 − centroid + 0.5⌋, integer floor division, so Python and C
agree bit-for-bit); both channels shifted, vacated zones = empty values, zones pushed off the grid dropped, no-hand frames unchanged.
Afterwards every hand centroid lies in (3.0, 4.0] on both axes. Unweighted centroid, same definition as §5.1j.
**LOSO experiment (`pc_tools\train_phase55c.py`):** arm **B** = the E5 recipe (§5.1i arm B) **re-run in the same job**; arm **R** = B +
`recentre()` on the training frames (after the left-right flip) and on the test frames — the only difference. Seeds 42–46 × 6 folds × 2
arms = 60 models; folds, data (`data\phase4` + `phase4_extra`, ST 4 users), weighting, early stopping exactly as §5.1i (reuses
`train_phase55.load_all / fold_indices / subject_weights`). Our subjects' folds test on their main set only.
**Adoption rule (fixed now):** use R if **mean of 6 (R) − mean of 6 (B) ≥ +1.0 pp** **and** **ST-fold mean (R) ≥ ST-fold mean (B) − 1.0 pp**.
Report per-subject R − B with exact Wilcoxon (n = 6, smallest p 0.031), per-class recall for both arms.
**Predictions (Claude):** P1 mean-of-6 R ≥ B + 1 pp (55 %). P2 User 5: R ≥ B + 3 pp (60 %). P3 User 5 Like recall under R > 50 %
(B was 23.1 %) (55 %). P4 ST-fold mean |R − B| ≤ 2 pp (60 %). P5 the B re-run's mean of 6 is within ±1 pp of 85.30 (70 %).
Risk named in advance: re-centring discards absolute position, which may have carried class information (e.g. two-hand postures).
**If R is adopted (Rule 6 chain, each step its own check):** (1) train 5 final R models on all 6 subjects (seeds 42–46) → ensemble
**E5R**; (2) report on the natural new-day sets `phase56` and `phase56_stand` (report only, never for choosing) and A6-sim (sanity only —
R is shift-invariant to whole zones by construction, §5.1v); (3) X-CUBE-AI conversion; (4) firmware switch `MY_TOF_RECENTRE` with
integer-exact C, board vs PC on every frame of a fresh check; (5) per-frame timing (expected +< 0.2 ms); (6) a physical shift check
(move the board 3 cm with a ruler; separately pre-registered); (7) deploy E5R only if (4) is 100 % and (5) ≤ 33 ms.
**If R is not adopted:** E5 stays deployed; the thesis reports the position asymmetry (§5.1v) as a limitation with a positioning guide.

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
| STSW-IMG050 (ST, added 20 Oct 2026 `[WEB]`) | AI hand-posture recognition package, VL53L7CX/VL53L8CX, NUCLEO-F401RE. No dataset or pre-trained model listed; ST's workflow: collect from multiple users, retrain with the model-zoo script (§5.1k) |

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

**Subject counts for context (relevant to our n = 2, §5.1e-A1):** ST's public dataset 4, Bartoli 2024 4,
Wang 2023 6 (+6 for live re-test), Ahmed 2021 8, Boner 2022 12 — all from this file. Our own collected
data (n = 2) is below every one of them; the pooled ST + ours (6) is within the range.

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
| 5 | Measure timing diagram with a logic analyser | ✅ done — Phase 1, 4–11 Sep 2026 (§1.4, §3.1b–e) |
| — | *"ลองค้นหาข้อมูล data set จาก ToF ว่ามีบ้างมั้ย ครับ"* | 🟡 partially answered; the assistant initially missed ST's dataset |
| — | 28 Sep 2026: ADC + Timer + logic-analyser example (see below) | ✅ **FYI only** — advisor: *"ให้ดูเฉยยย ครับ"* (29 Sep) |

**Advisor's stated position on DMA:** polling for experiments, interrupt for real use,
DMA only if the frequency is high. Our measurement supports deferring DMA — see below.

### ⚠️ TRAP #9 — DMA would not fix the current bottleneck

DMA frees the **CPU**, not the **bus**. Measured CPU idle is already 46% of the frame
period. The constraint is bus occupancy (12.18 ms of a 24.45 ms period), so DMA cannot
shorten the period. Fewer bytes, or SPI, can. State this when the topic comes up.

**Resolved with the advisor (deep talk, direct conversation, 15 Sep 2026):**
static vs dynamic gesture set was asked ~5 times with no answer until this direct
conversation. **Answer: train static first, dynamic second. The advisor will decide
what the trained postures are used for once static is done — not decided yet, and
not the student's or the assistant's job to guess.** This conversation was with the
**main advisor (อ.Sakreya)**, the project owner — see `[[sakreya]]`.

**Superseded by the above, same day:** a separate instruction had come from the
**second advisor (Advisor 2)** for a different, larger scope — up to 36 static
postures matched one-to-one to spoken phrases via a paired microphone board on a
classmate's project, with audio playback hardware (DFPlayer-class module) on this
project's board. That instruction is **no longer in effect** now that the main
advisor has spoken directly. Kept here rather than deleted in case it is revived:
no dataset collection, hardware purchase (DFPlayer/microSD/speaker), or coordination
with the classmate's board should proceed under it unless the main advisor
re-confirms it.

**Practical consequence for scope right now:** proceed with the 7-class
`ST_VL53L8CX_handposture_dataset` (§5.1) for the static phase. This already
satisfies "no more than 10 classes" from the second advisor's earlier instruction,
should that constraint still apply once the advisor names a downstream use.

**Advisor's shopping cart (31 Aug)** — 6 items, 3 are microphone boards for a different
project. Relevant to us: `SATEL-VL53L8` (breakout, 2 boards, VL53L8CA — lets the sensor
be placed away from the board, and is what Bartoli used), `NUCLEO-L476RG` (1 MB flash,
ultra-low-power, Arduino-compatible so our shield fits, and the board used in the
ShortRange paper), `FRDM-MCXN947` (NXP, eIQ Neutron NPU — a possible AI-acceleration
comparison, but a different ecosystem and a port would be substantial work).

### Advisor message 28 Sep 2026 — ADC example (LINE, from อ.Sakreya) — FYI only

**Context:** sent after Chin reported that the model was running on the board (Phase 3). The same
message was also sent to at least one other student `[STUDENT]`.

**Text (verbatim):**
- 13:44 — *"ADC conversion ทุกๆๆๆ 200ms โดยใช้ Timer ตั้งเวลาให้ Trigger ADC"* + a screenshot
- 13:44 — a 48-second screen-recording video
- 13:45 — *"ใช้ Logic analyzer ราคา 200 บาท ส่งสถานะ ADC interrupt ไปออกที่ Digital output เพื่อใช้ Logic analyzer เพื่อทดสอบ ครับ"*

**What the video shows** `[ตรวจแล้ว: 12 frames extracted at 1 per 4 s + audio level check, 28 Sep 2026]`:
- **No speech** — the audio track is silent (−91 dB throughout).
- Project `example-01-adc1-in5-tim4-interrupt-mode` → ADC1 channel 5, triggered by **TIM4**,
  interrupt mode; sibling projects `...-in678-interrupt-mode`, `...-in6/in9-polling-mode`,
  `...-temp-polling-mode(-hal)` — a series of teaching examples.
- Linker script `STM32L476RGTX_FLASH.ld` → the demo runs on a **NUCLEO-L476RG**, not our F411.
- Code (register-level): on each conversion `GPIOA->ODR ^= (1UL << 5)` (toggle PA5), convert to volts,
  `sprintf` + `USART_Write(USART2, ...)`; Hercules terminal on COM6 @ 115200 shows ≈3.29–3.30 V.
- In the logic-analyser view the advisor measures: one level **201.536 ms**, full period
  **403.052 ms**, 2.481 Hz, duty 50 %. Because the pin *toggles* once per conversion, one level =
  one conversion interval ≈ 200 ms — the timer rate is confirmed externally.

**Asked 29 Sep 2026 14:57** (Chin): *"โอเคครับอาจารย์ ผมดูแล้วครับ อันนี้อาจารย์อยากให้ผมลองเอาวิธีนี้ไปใช้กับ Project ด้วยไหมครับ"*
**Answer 29 Sep 2026 16:51:** *"ให้ดูเฉยยย ครับ"* — Chin acknowledged 19:23 (*"โอเคเลยครับอาจารย์"*).

**Status: closed, not a task.** The technique stays available as an optional extra (§1.4, last
subsection). No ADC exists in our system (§1.3) and none is needed for that technique.

---

# 8. WHERE THE PROJECT STANDS

## Contributions — status

| # | Contribution | Novel? | Status |
|---|---|---|---|
| 1 | Measuring sensor→MCU transfer time | none of the 5 papers report it | ✅ done |
| 2 | Sensor/MCU serialisation model, slope 0.981, R² 0.9997 | not reported anywhere | ✅ done |
| 3 | 8-condition timing matrix, one variable per step, both modes | not reported anywhere | ✅ done |
| 3a | Mode-dependent effect of the same optimisation (headroom vs throughput) | not reported anywhere | ✅ done |
| 3b | Working on-board inference + 4-stage latency breakdown | none of the 5 papers report a 4-stage on-device breakdown | ✅ done |
| 4 | Subject-independent + cross-dataset evaluation | ST did not do this — **verified in their source**, `data_loader.py:184-191` shuffles and splits with no user grouping | 🟡 subject-independent part DONE 15 Sep (random 97.24% ± 0.34 vs leave-one-user-out 79.73% ± 8.63 pp, §5.1d); cross-dataset part pending (Phase 4 / 5.5). **With n = 2 own subjects (§5.1e-A1): cross-sensor H8 = descriptive case study; main quantitative result = 6-subject LOSO in Phase 5.5**. **Phase 5.5 done 20 Oct:** 6-subject LOSO, arm B 85.30 % mean of 6 (ST folds 85.61), our subjects 82.8 / 86.6 % (§5.1i-R) |
| 4a | Position sensitivity of the ST architecture characterised live, per posture (§5.1e-R, §5.1f) | not reported in the papers in this file `[ยังไม่ได้ตรวจ beyond them]` | 🟡 observed (n = 1); before/after test planned for Phase 5.5 |
| 5 | First dynamic-gesture dataset for VL53L8CX | nothing comparable exists | ⬜ |

## Plan

```
Phase 0    lock the data format + clock + measurement matrix               DONE 2 Sep
Phase 1    logic analyser verification                                     DONE 11 Sep
Phase 2    train on ST's dataset; subject-independent split                DONE 15 Sep
Phase 3    deploy AI to board, instrument 4 stages                         DONE 21 Sep
Phase 4    closed-loop check: our board's data → ST's model                DONE 20 Oct (§5.1e-F)
             pre-flight 21-trial live test                                 DONE 24–25 Sep (§5.1e-R)
             collection: Chin (informed) + 1 friend (naive), n = 2         DONE 18 Oct (§5.1e-P)
Phase 5    collect our own static dataset (4-6 people)                     MERGED INTO PHASE 4 —
                                                                           only 2 people exist (§5.1e-A1)
Phase 5.5a shift-augmentation ablation on ST only, tested on User 5            PRE-REGISTERED 10 Oct (§5.1g)
Phase 5.5  PRE-REGISTERED 14 Oct (§5.1i): arms A/B/C, 5 seeds, 6-subject LOSO, selection on ST folds;
           LOSO RUN DONE 20 Oct (§5.1i-R): arm B selected; next = final training on 6 subjects + 5.5b;
           train on ST + our 2 (6-subject LOSO), cross-dataset comparison,
           Love edge/centre before/after test, deploy                      1-2 wk
Phase 5.6  different-day test of Chin + friend (not used for training) —
           estimates demo-day accuracy for known users                    planned 10 Oct
Phase 6    collect dynamic dataset                                         3-4 wk
Phase 6.5  train temporal model (TCN/RNN), deploy                          2-3 wk
Phase 7    write up                                                        1 mo
```

Ordering principle: cheapest-and-reversible first, hardest-to-undo last.
Recruiting volunteers is the least reversible step, so it comes after the pipeline is
proven end-to-end. (Applied in Phase 4: Chin runs `collect_own_data.py` first as a dry run
before the friend's session.)

## Immediate next action

**Project origin (confirmed 22 Sep 2026 from a LINE screenshot Chin shared):** on 4 Aug 2026,
Advisor 1 (อ.Sakreya) sent the UWB-Gestures reference repo, both related-work PDFs in §10 below,
and the message "เปลี่ยนจาก UWB เป็น ToF เป็นอีกแนวทางหนึ่งครับ" — the point at which the project's
sensor choice moved from UWB radar to ToF. Useful for the thesis introduction (why ToF over UWB).

**Superseded by QUICK STATUS (17 Oct 2026). Status as of 9 Oct 2026 (late): Phases 0–3 complete; Phase 4 pre-flight done (21/21 with per-posture
positioning, §5.1e-R); `collect_own_data.py` v4 (guided arming, A8) dry-run passed; **Chin's data
collected** — main 14/14 + extra day 1 14/14 (§5.1e-P step 4). Next, in any order that suits the
people involved: evaluation script (step 6, can run on User 5 alone first), friend (User 6), Chin extra
day 2. Back up `data\phase4` and `data\phase4_extra` off the PC before anything else.** The 28 Sep advisor example was FYI only (§7) and does not change the plan.
Progress report 2 finished 21 Sep 2026 (sections 3–6, 8 pages, dated 20/09/2569);
advisor feedback pending; the next progress report continues numbering at section 7.

Phase 0 is complete. Both modes now run at the hardware ceiling with no missed
ticks (`delta` constant, `dup = 0` across all eight conditions).

**Phase 1 is COMPLETE as of 11 Sep 2026.** H1–H7 all answered; TRAP #12 resolved;
TRAP #20 characterised over a full 6-hour cold start (n = 121, 4×4 mode); five
unplanned findings (§3.1c–e); one firmware bug found and fixed. H7 was
pre-registered and **rejected on all five counts**, including the sign of the
change. `fig5_drift.png` has been regenerated from the designed experiment and the
"Preliminary" label is gone. ~~The probe wires have been removed~~ **Corrected 28 Sep 2026
`[STUDENT]`: the probe wires are still attached** (§9 #45). Any future timing work still requires
re-verifying which channel sits on which pin — the board has been moved between work locations
since — and will **not** be directly comparable to the 10–11 Sep results.

Write-ups: `data/phase1_int/RESULTS_phase1.md` (H1–H6, calibration, bus) and
`data/phase1_drift_v2/RESULTS_phase1_drift.md` (cold-start drift, H7 rejection).
Figure regenerated by `pc_tools/make_fig5.py`.

**Scope confirmed 15 Sep 2026 (direct conversation with the main advisor):** static
gesture recognition first, dynamic second, downstream use to be named later by the
advisor. See §7 for the full record and what this supersedes. Practical scope for
Phase 2: the 7-class `ST_VL53L8CX_handposture_dataset` (§5.1), ≤10 classes as
required.

Next work, in order:

1. ~~**Phase 2 — training.**~~ **DONE 15 Sep 2026.** Random split 97.24% ± 0.34 pp;
   subject-independent 79.73% ± 8.63 pp (§5.1d). First live hardware test with an
   unseen hand also done (§5.1c) — all 7 gestures recognised correctly once
   positioned at the corrected ~270 mm test distance.
2. ~~Set `MY_TOF_TIMING_MODE = 0`, capture F/S/G lines~~ **DONE 15 Sep 2026** as
   part of the live-hand test setup (§5.1c) — confirmed `G,` carries sensible,
   physically consistent signal values (near objects give high signal, far objects
   low signal) and distances match expectations.
3. Compare the measured `max_bytes` (from the `T,` line) against the estimates in
   §3.5's "TO BE TESTED" table — **still not done**, needs `TIMING_MODE = 1`
   temporarily (mutually exclusive with item 2's data-mode lines).
4. **DONE 21 Sep 2026.** **Decided 16 Sep 2026: Phase 3 (embedded deployment) chosen next**, per the plan
   document generated that day. In progress: found and fixed a real MCU
   configuration bug that predates this conversation — the whole project had been
   built against STM32F401RE (96 KB RAM) instead of the physical STM32F411RE
   (128 KB RAM) actually on the board, confirmed by a photo of the chip marking.
   Fixed by editing `STM32F401RETX_FLASH.ld` (`RAM LENGTH 96K → 128K`), verified via
   the post-build `.map` file, committed as `ac44da5`. No prior measurement was
   affected (RAM used throughout Phase 0–2 was ~3 KB, far under even the wrong
   ceiling). See correction log #36 for the full account. Device macro itself is
   still F401 (only the RAM ceiling was corrected) — low priority, revisit before
   Phase 6.5 if an F411-only peripheral is ever needed. Next concrete step:
   create a separate CubeMX project (not editing the existing `.ioc` — that
   workflow was tried twice and both times created a new untitled project instead)
   to run X-CUBE-AI and convert `production_model.keras` to C code.
5. Phase 5 (collect the team's own dataset) and further Phase 2 analysis (User4's
   high seed-to-seed variance, §5.1d) remain queued behind Phase 3.
   *(5 Oct 2026: Phase 5 merged into Phase 4, §5.1e-A1.)*
6. **Phase 4 — next (5 Oct 2026):** follow §5.1e-P. Housekeeping before or alongside it:
   diff the two `live_predict.py` copies and delete the root one (§2.4); write the Phase 4
   evaluation script (§5.1e-P step 6).

Open question, **deferred by decision on 11 Sep — see §8.5 D1** for the full
reasoning, the designed experiment, and what to write in the limitations if it is
never done:

7. **Temperature is uninstrumented and now demonstrably matters.** Two runs in the
   same 4×4 mode, both described as cold starts, began ~1100 ppm apart (6038 ppm on
   10 Sep, 7139.8 ppm on 11 Sep). Neither logged ambient temperature or power-off
   duration. Deferred because the effect is 0.059 % of the frame period and blocks
   nothing; resume after Phases 2–3. See correction log #31.

Firmware state after 15 Sep: `USE_4X4 = 0` (8×8, required to match the model's
input shape), `USE_INT = 1`, `FAST_READ = 1`, `TIMING_MODE = 0` (data mode, `F,`/`S,`/`G,`
lines active), `MY_TOF_STREAM_STEP` auto-selected by mode, HSE clock. This differs
from the Phase-1-era state below, which predates today's mode switch.
*(Still the expected state on 24–25 Sep: `live_predict.py` received 64-value lines with no 4×4
warning and valid predictions — consistent with 8×8, raw status, data mode. Switches were not
re-read from source that day.)*

**Firmware state at the end of Phase 1** (for comparison, no longer current):
`USE_4X4` per experiment, `USE_INT=1`,
`FAST_READ=1`, `TIMING_MODE=1`, `DELAY_US=0`, `MY_CAL_ENABLE=0`,
`MY_TOF_STREAM_STEP` auto-selected by mode, 5 DISABLE macros active, HSE clock.

**Tools built in Phase 1** (`pc_tools/`): `analyze_calib.py` (period from edges,
least-squares fit, `--edge` and `--channel` options), `analyze_bus.py` (I2C bus
utilisation, SCL frequency over many pulses, INT→START latency),
`plot_phase1.py` (four figures including the setup block diagram).

**Data:** 25 `.bin` captures + 9 figures + 4 UART logs in `data/phase1_int/`,
3 calibration captures in `data/phase1_calib/`. `.bin` excluded from git by
`data/.gitignore`; backed up to Google Drive as a 5.3 MB zip (verified by
extracting and re-running the analysis).

---

## 8.6 Open items — as of 20 Oct 2026 (night)

| Item | State |
|---|---|
| Final training: arm B on all 6 subjects × 5 seeds, **models saved** | ✅ 20 Oct 2026, `results_phase55_final\`, commits `e1f08ee`/`b0bcc01` |
| 5.5b pre-registration: rule for choosing 1 of 5 final models, or a seed ensemble | ✅ written §5.1l (E5 if it fits the board, else seed 42) |
| Phase 5.6 new-day test set `data\phase56` (never for training) | ✅ User 5 (§5.1p, E5 93.0 %); ⬜ User 6 |
| X-CUBE-AI conversion + on-board timing of the final model; compare a `padding="valid"` variant (D2) | ✅ E5 deployed 26 Oct (§5.1n: 29.3 ms per frame, 203/203 board vs PC); D2 `padding="valid"` / `-O2` comparison ⬜ |
| A5 Love edge/centre live test with the new model | ⬜ |
| 5.5c (round 2) pre-registration: re-centring / metric-size normalisation, `padding="valid"`, recomputed constants — each needs firmware parity | ⬜ candidates only |
| Split-fixed ST-only model on User 6, several seeds (clean baseline for §5.1i-R's User 6 comparison) | ⬜ |
| Read ST's `.keras` Conv2D config directly (closes §9 #56) | ⬜ |
| Is row 0 physically "up" on our mount? | ⬜ `[ยังไม่ได้ตรวจ]` |
| Disclose the quarantined friend takes (A9) in the thesis | ⬜ writing |
| Copy `backup_phase4_20261020` to Drive/USB | ⬜ not confirmed; `phase4_for_claude.zip` (Documents) may be deleted |
| Progress report 3 (sections 7–12) | ✅ finished by 25 Oct 2026 `[STUDENT]` (Word; Thai line-breaking fixed; 2-take sets called ชุดหลัก/ชุดเพิ่มเติม, §9 #60) |
| Flash 111.89 KiB (analyze report) vs 112.61 KiB (CubeMX panel) for E5 | ⬜ unexplained, 0.72 KiB; settle from the build size in step 2.4 |
| Stray `53L8A1_SimpleRanging\Inc\my_tof.c` (not compiled) — compare with `STM32CubeIDE\Application\User\my_tof.c`, then delete | ⬜ |
| `collect_own_data.py` with `Y,` lines present (`MY_TOF_PRED_LINE = 1`) — check before `data\phase56`, or build with 0 | ⬜ `[ยังไม่ได้ตรวจ]` |
| Single-frame worst-case per-frame work | ✅ 26 Oct: max 29,295 µs over 4,219 frames (§5.1p) |
| **Reminder (Chin's explicit request, 20 Oct 2026):** when the project ends — or when thesis/defence wrap-up starts — remind Chin, even if he forgets, to build a **reusable skill** capturing every good working method of this project, from §9, the project memory's learnings and preferences | ⬜ at project end |


# 8.5 DEFERRED WORK — questions consciously left open, with the plan to resume

Not abandoned. Each entry records what is unanswered, why it was deferred, the
experiment already designed to answer it, where the raw data sits, and what goes
into the thesis limitations if it is never done. Written on 11 Sep 2026 so that a
future session can resume without reconstructing the reasoning.

## D1 — Is temperature the cause of the sensor's frame-rate drift?

**Status:** open. Deferred 11 Sep 2026 by the student's decision, with the reason
recorded below. Does **not** block any later phase.

**The unanswered question.** §3.1d identifies the sensor's RC oscillator as the
drift source and names temperature as the leading suspect, but **no temperature has
ever been measured in this project**. The strongest evidence that it matters is the
unexplained gap in §9 #31: two 4×4 cold starts, both described the same way, began
~1100 ppm apart (6038 ppm on 10 Sep, 7139.8 ppm on 11 Sep). Neither run logged
ambient temperature or power-off duration, so the cause cannot be recovered from
the data already collected.

**Why deferring is defensible.** The effect is real but operationally irrelevant:
the entire 6-hour drift is 592.7 ppm = **0.059 % of the frame period**, and even
the much larger per-frame jitter (200 µs at 4×4, §3.1e) accumulates to only ~1.55 ms
over a 1-second gesture window — 0.15 %. Nothing in Phases 2–6 depends on resolving
it. It is a completeness question for the write-up, not a functional defect.

**Why simply logging temperature would NOT answer it.** In a single cold-start run,
ambient temperature rises together with elapsed time, so the two are confounded and
any correlation found is uninterpretable. **Temperature must be deliberately varied**
for the causal claim to hold.

**Designed experiment (option B), to run after Phases 2–3:**

1. **Instrument the board to time itself.** H2 established that DWT is trustworthy
   (verified three independent ways). The ±1013 ppm figure quoted in §3.1b is a
   limitation of the `R` line — it counts whole milliseconds over a ~1 s window —
   **not of DWT itself**. Timestamping the first and last frame of a 10-second
   window with DWT gives a resolution of order 1 ppb, thousands of times finer than
   the ~600 ppm effect being measured. `[ยังไม่ได้ตรวจ]` — `my_tof.c` has not been
   opened to confirm how hard this change is; verify before committing to the plan.
2. **Run the same cold-start protocol at two or three deliberately different ambient
   temperatures** (for example air-conditioned vs not), logging ambient temperature
   at every round, and chip surface temperature too if an IR thermometer is available.
3. **Test whether ppm tracks temperature across conditions**, not merely across time
   within one run. Report the coefficient in ppm/°C with its confidence interval.

**Why this design is preferred.** It needs **no logic-analyser probes**, so the
"probes removed, pin positions would need re-verification" constraint does not
apply and the run can be repeated as often as wanted. It also converts a weakness
("we found drift but do not know why") into a contribution ("we measured the
sensor's temperature coefficient"). *(5 Oct 2026: the probes were in fact never removed,
§9 #45 — the design's advantage of needing no probes still stands.)*

**Raw data already on disk, still usable:**
- `data/phase1_drift/` — 121 captures, 8×8 mode, 6 h, 3 Sep protocol (the run that
  was mislabelled; see §9 #29). Valid as an 8×8 cold-start record.
- `data/phase1_drift_v2/` — 121 captures, 4×4 mode, 6 h, the H7 run.
- Both have `manifest.csv` with real wall-clock timestamps per round.

**If D1 is never done, the thesis must state:** that the sensor's frame rate drifts
by several hundred ppm over hours; that the cause was not established; that
temperature is the leading suspect on the basis of the RC-oscillator mechanism and
the unexplained 1100 ppm gap between two nominally identical cold starts; that no
temperature was instrumented; and that the magnitude is operationally negligible
for gesture recognition at 0.059 % of the frame period. Offer the option-B design
above as future work.

## D2 — Why is our measured inference 3.197 ms when ST publishes 1.46 ms?

Added 21 Sep 2026. Not urgent: inference is 19.3% of the 16.58 ms per-frame work and
the system is not inference-bound. Recorded so the gap is not explained by guessing.

| | ST's figure | Ours |
|---|---|---|
| Model | shipped st_cnn2d, 8 classes | our 7-class production_model (same architecture, last Dense 8→7) |
| MCU | NUCLEO-F401RE @ 84 MHz | NUCLEO-F411RE @ 84 MHz, HSE |
| Tool | ST model-zoo benchmark | X-CUBE-AI 10.2.1, DWT cycle counter |
| Inference | 1.46 ms (published) | 3.197 ms (measured) |

A second ST source (Hugging Face model card, CNN2D_ST_HandPosture, FLOAT32 8×8×2, STM32F401
@ 84 MHz, X-CUBE-AI 10.2.0) lists **1.53–1.54 ms** `[WEB]` — so ST's own sources give 1.46 ms
(README, §5.1) and 1.53–1.54 ms, both about half of our 3.197 ms.

Candidates, **none tested**: compiler optimisation level; runtime library / kernel
choice; what the two measurements include; **our model has 4,511 parameters vs ST's published 2,752 (§9 #56)**.
**Leading candidate (20 Oct 2026, `[คำนวณ]`, untested):** padding "same" vs "valid" gives 13,536 vs 7,744 MACs = 1.75×, against a
measured time ratio of 3.197 / 1.54 = 2.08× (§5.1k). Test: convert a `padding="valid"` model with the same tool chain and time it. Resume when inference time starts to matter
(a larger or temporal model, Phase 6.5).

## D3 — Independent (logic-analyser) check of the Phase 3 stage timings — optional

Added 5 Oct 2026. The four on-board stage times (13,254 / 126 / 3,197 / 2 µs) were measured with
DWT only. A GPIO-marker capture (§1.4, last subsection) would give an external, visual timing
diagram. **Not requested by the advisor** (§7, 28–29 Sep) and not needed for correctness — DWT is
already verified three ways (§3.1b). Do it only if a figure for the report/defence is wanted;
estimated half a day to a day `[ประเมิน]`. Prerequisites: a confirmed free GPIO (§1.3), probe
channel→pin mapping re-checked (§1.4).

## D4 — `None` / open-set class

Added 5 Oct 2026. The model has no reject class and no confidence threshold (§5.1f), so it always
names one of 7 postures. Whether to collect `None` from our two subjects, or to add a confidence
threshold in the decision stage, is **undecided**. Relevant for any live demo where a committee
member may hold a non-posture hand shape.

---

# 10. RELATED WORK — reviewed 22 Sep 2026, for thesis/paper writing

All in Project Knowledge, read this turn `[PAPER]`. Paraphrased, not quoted (copyright).

## 10.1 Directly on our sensor family (ST VL53L8)

**Bartoli, Saporito, Scandelli, Giudici, De Vecchi, Zappa (Politecnico di Milano), "Time-of-Flight
Hand-Posture Recognition using Compact Normalized Histogram", IEEE conf. 2024.**
- Sensor: VL53L8CH on a VL53L8-SATEL board + NUCLEO-F401RE, I2C.
- Compares two data types from the same sensor: the 8×8 Distance Map (DM, what our project uses)
  vs. Compact Normalized Histogram (CNH, a richer per-pixel histogram of photon arrival times).
  CNH reached 96% accuracy vs. DM's 88%, and used less memory (9.1 kB vs 18.8 kB) — CNH carries
  3-D information DM does not.
- 3 postures + "Null" (Rock/Paper/Scissors/none), 4 subjects, 825 frames/class collected at 20–30 cm,
  mirror-augmented to 6600 frames, **random** (not subject-independent) train/test splits repeated
  over 100 seeds via sklearn — a methodological point worth contrasting against: our subject-independent
  split (§5.1d) is the more rigorous choice for a generalization claim, and is worth stating as such.
  70/20/10 train/val/test.
- Cites ST's own shipped 7-class model figure independently: **1.5 ms** inference on STM32F401 — a
  third source (with README §5.1 and the Hugging Face card, §8.5 D2) agreeing ST's own number is
  ~1.5 ms, distinct from our measured 3.197 ms.
- Their own CNH model benchmarked at **5.34 ms** on NUCLEO-F4R01RE @ 84 MHz (376,416 cycles) and
  **0.5585 ms** on an H743ZI2 @ 480 MHz (25,944 cycles, with a hardware accelerator) — this is the
  traced source of the "5.34 ms" figure in Correction Log #8.
- VL53L8CH-specific settings they used for CNH acquisition: Ranging Mode Autonomous, Integration
  Time 20 ms, Start Bin 0, 7 histogram bins, sub-sample 2 — these are CNH parameters, not directly
  the DM acquisition settings ST used for our 7-class dataset (still not published, §5.1e item 6).
- **Extension checked and ruled out 22 Sep 2026:** CNH is **not** available to us. `[UM3109]`'s
  full feature table of contents (Resolution, Ranging frequency, Ranging mode, Integration time,
  Power modes, Sharpener, Target order, Xtalk margin, Detection thresholds, Interrupt autostop, ...)
  has **no histogram/CNH entry**. CNH is specific to VL53L8CH/VL53L7CH (their own datasheet, DS14310,
  is different from ours, DS14161, per `[UM3120]` §1.6's ordering table) and needs that variant's own
  driver — not the standard ULD (UM3109) we use for VL53L8CX/CA. Replicating Bartoli's DM-vs-CNH
  comparison would need a VL53L8CH part and its driver, which we do not have. Do not propose this as
  a contribution without first acquiring that hardware.

## 10.2 Dynamic gesture recognition with low-pixel ToF (motivates future dynamic-gesture phases)

**Wang, Feng, Shi, Wang (Zhejiang University), "Research on Dynamic Gesture Recognition with
Low-pixel ToF-Sensors", IEEE conf. 2023.**
- Sensor: an 8×8 ToF module (P8-SMD-B15/P8864), **not** an ST part — resolution matches ours, vendor
  does not. 30–60 cm range, STM32F411CEU6 (same MCU family as ours).
- Pipeline: non-global mean filter for outlier zones, frame-difference sliding window to segment a
  gesture's start/end, Lagrangian interpolation to a fixed 10-frame length, then a small parallel-CNN
  ("TPC-Net": 1×1/1×1/3×3 conv, channels reduced 10→8→4, BN after each, two FC layers).
- 6 dynamic gestures (grasp, release, left/right wave, left-right wave, finger rotation), 6 volunteers,
  1440 samples: 98.47% in-sample, **91.04% cross-user**, and a **live re-test with 6 new volunteers**
  (10 trials × 6 gestures each) at **89% overall**, response time < 30 ms.
- **This per-user × per-gesture results table (their Table V) is the closest existing template for
  how to report our own Phase 4/5 results** — report every subject × every class, not just an average.
- Their own stated limitation: cross-user accuracy still has room to improve; future work = more
  subjects. Same open question we have for Phase 4.

**Boner, Vogt, Magno, "Tiny TCN model for gesture recognition with multi-point low power ToF-sensors",
IEEE AICAS 2022 ("Tiny TCN").**
- Sensor: **VL53L5CX**, 4×4 @ 60 Hz, continuous mode, 20 cm — this is the sensor the ST community
  reply (§1.1) warned does not transfer well to VL53L8-trained models; a **different** family from
  ours, confirmed by resolution (4×4 max width) and part number.
  - "P8" in the Zhejiang paper is a third, unrelated vendor part — three different 8×8/4×4 ToF
    products appear across these three papers; none is VL53L8CA, and only Bartoli 2024 (§10.1) uses
    our exact sensor family.
- 6 gestures (Finger Slide, Slow Swipe, Push, Pull, Palm Tilt + idle), 12 users, TCN+CNN hybrid,
  <100 kB memory, ~95% per-sequence accuracy. Pareto sweep of filters/kernel/dense size reported
  (Table I) — a good template if we ever need to justify a model-size vs. accuracy trade-off.

## 10.3 Dataset methodology reference (not our sensor, cited for pipeline design)

**Ahmed, Wang, Park, Cho, "UWB-gestures, a public dataset of dynamic hand gestures acquired using
impulse radar sensors", Scientific Data (Nature) 2021.** `[Advisor-recommended, 4 Aug 2026]`
- UWB impulse radar (not ToF), 12 dynamic gestures, 8 subjects, 9,600 samples, 3 radars at 3
  locations. This is the paper and dataset behind the `DenissStepanjuk/UWB-Gestures-classification-
  with-Neural-Networks` reference repo (§7) — **confirmed 22 Sep 2026** by opening the repo: it
  implements CNN/RNN/Transformer classifiers in PyTorch on exactly this dataset, no ST or ToF
  hardware involved. Useful only for the *shape* of a public-dataset paper (how they describe
  acquisition protocol, subject diversity, multi-location capture) — not for any numeric comparison.
- **Advisor context (confirmed 22 Sep 2026 from a LINE screenshot):** อ.Sakreya sent this repo and
  both papers in this section together on 4 Aug 2026, then wrote "เปลี่ยนจาก UWB เป็น ToF เป็นอีก
  แนวทางหนึ่งครับ" — the message that moved the project from a UWB-radar direction to the current
  ToF-sensor direction. Record this in the thesis introduction as the reason for choosing ToF.

## 10.4a Independent cross-check of our FoI spec, and VL53L8CH used for radar-comparison sensing

**Zabierowski, Hałaczkiewicz, Łagosz, Falęcki (Warsaw University of Technology), "Short-Range Human
Movement Sensing: Evaluating Time-of-Flight Sensors as a Low-Cost Radar Alternative", 2026 (SignAI
Hackathon, Gdansk, Mar 2026).** `[PAPER]` — found in Project Knowledge but missed in the first pass
of this review; read in full 22 Sep 2026 after Chin asked whether only 4 papers existed.
- Sensor: **VL53L8CH** (not CX or CA — a third VL53L8 variant now seen in our own literature,
  alongside CX in ST's README and CH again here; §1.1's ordering table already distinguishes all
  three), on a **NUCLEO-L476RG** (STM32L4, not our F4 family), 8×8 continuous mode, raw depth +
  reflectance streamed over USB serial, analysed offline in MATLAB.
- **Independent confirmation of our own spec:** states FoI = **43.4° × 43.4°** at 75% of max signal
  — matches our `[DS]`-sourced value in §1.1 exactly, from a source outside our own datasheet reading.
  Strengthens confidence in that number.
- Detection range degrades sharply under bright ambient light: **4 m → as low as 70 cm** at 5 kLux
  (bright indirect sunlight) against a 17%-reflectance grey target — worth citing if we ever discuss
  why data collection should avoid bright rooms.
- Compared against a reference 122 GHz FMCW radar (XY-DemoRad, ~3 cm range resolution): for
  short-range human-movement sensing, the ToF sensor gave results consistent with the radar at a
  fraction of the cost and compute — general validation that VL53L8-family sensors are a credible,
  low-cost substitute for radar in this class of application, supporting the project's original
  UWB→ToF pivot (§8's project-origin note).
- Not a hand-posture classification paper (its task is range-velocity mapping of whole-body
  movement, not our 7-class posture task) — cite for sensor validation and the FoI cross-check, not
  as a directly comparable accuracy benchmark.

## 10.4 Most recent related work (2026, same reference dataset as §10.3)

**Ahmed, Mahmoud, Elbahnasawy, Driessen, Youssef, "Real-Time Radar Hand Motion Recognition Using a
Hybrid CNN–SVM Framework With FPGA Acceleration", IEEE Trans. Radar Systems, Vol. 4, 2026.**
`[Advisor-sent, 4 Aug 2026]` `[PAPER]` read in full 22 Sep 2026.
- Same 12-class UWB-Gestures-style dataset (classes labelled G1–G12, matching §10.3's dataset).
  Not ToF — UWB radar, image-based (binarized range–time maps) and a continuous-valued range–time
  map (CV-RTM) pipeline compared.
- Hybrid CNN–SVM (three conv blocks: 16/32/64 filters, ~688k parameters) beats a CNN–LSTM–SVM
  baseline (93.3% vs 92.2%) with 75.5% fewer parameters; curriculum learning (progressively noisier
  training data) gives the most stable accuracy across SNR −20 to 10 dB.
- Deployed on a Xilinx Kria KR260 FPGA via Vitis-AI (INT8 quantization, accuracy drop < 2%):
  **91.67% on-device accuracy**, **0.946 ms** inference latency per sample.
- **Notable structural parallel to our own Phase 3 finding:** radar acquisition itself takes ~4.5 s
  per gesture (90 frames at 20 fps) — acquisition, not inference, dominates end-to-end latency, the
  same pattern we measured (sensor read = 80% of our per-frame budget, model = 19%, §6.2). Worth
  citing in the discussion section as evidence this is a general pattern in embedded gesture
  recognition, not an artifact of our own pipeline.
- FPGA deployment (Vitis-AI/DPU) is a different embedded target from our STM32 MCU deployment — not
  directly transferable, but a citable alternative if a future phase explores hardware acceleration.

---

# 9. CORRECTION LOG — mistakes the assistant actually made

Kept so the same errors are not repeated. Each was caught by the student.
**Scope (20 Oct 2026):** only mistakes in this project's work go here. Side tasks in the same chat (e.g. Office / Claude-for-Word set-up)
are not logged in this file — it lives in the repo.

| # | Claim made | Reality | Root cause |
|---|---|---|---|
| 1 | "43.4° is wrong, it's 45°" (implying the advisor erred) | Both correct; FoI ≠ FoV | Did not open the datasheet |
| 2 | "The 291 mm reading is noise, filter it" | Real object; 100% of statuses valid | Assumed instead of measuring |
| 3 | "Cut the unused fields" | signal_per_spad is required by ST's model | Did not check what ML would need |
| 4 | "INT is slower than polling" (from n=1) | INT is slightly faster (n=14) | Concluded from a single data point |
| 5 | "No dataset matches our sensor" | ST publishes an exact-match dataset | **Never actually searched** |
| 6 | "I have searched exhaustively" | Two queries, both reusing terms from an already-found result | Searched to confirm, not to discover |
| 7 | "No dynamic dataset at all" | One exists (VL53L5CX, IEEE DataPort) | Did not check the list item by item |
| 8 | inference = 5.34 ms | Traced 22 Sep 2026 to its actual source: Bartoli et al. 2024 ("Time-of-Flight Hand-Posture Recognition using Compact Normalized Histogram", VL53L8CH, §10) Table V, their own CNH-based model on NUCLEO-F4R01RE @ 84 MHz, 376,416 cycles — **not** ST's shipped model and **not** ours. ST's own shipped-model figure is 1.46 ms (§5.1); our own measured on-board figure is 3.197 ms (§9 #40) | Quoted a different paper's model, without tracing which paper |
| 9 | Deployment listed as one sub-bullet | It is the step that completes the whole contribution | Under-weighted in planning |
| 10 | Phases 5 and 6 ended at "collect data" | Collected data must then be trained and deployed | Plan written carelessly |
| 11 | Proposed `FAST_READ` without stating it requires INT | In polling mode it duplicates 43% of frames | Did not think through when the removed check mattered |
| 12 | Payload estimates: predicted 148 B (got 196), 536 B (got 580) | Consistently ~45 B low | Scaled by zone count, ignored per-block headers |
| 13 | Wrote "we exceeded the datasheet spec" | "up to 60 Hz" is a ceiling; +0.7% is an unattributed offset | Treated a favourable number as a result instead of questioning it |
| 14 | Claimed INT is what takes 4×4 to 60 Hz | INT changes nothing there (E→F: 40.57→40.49); the payload reduction does (F→G: +49%) | Generalised the 8×8 result to 4×4 without measuring |
| 15 | Assumed `huart2` was a variable and used `HAL_UART_Transmit(&huart2, …)` | It is a macro from `stm32f4xx_nucleo.h`; the project uses `printf` via `_write` | Guessed the project's structure instead of opening the file |
| 16 | Wrote `/* USER CODE BEGIN */` inside a `/* … */` comment block | C does not nest comments; the file broke with 188 errors | Basic language rule, not checked |
| 17 | Used HAL TIM without checking the module was enabled | The project has the TIM module disabled; had to write registers directly | Same root cause as #15 |
| 18 | Printed with `%f` | The project builds with `nano.specs`, which drops float printf; fields came out empty | Did not check the build flags |
| 19 | Estimated drift by linear extrapolation and got 424 ppm | The A-B-A experiment showed 975 ppm | Extrapolated a process already known to be non-linear |
| 20 | Reported H5 as "INT to first SCL rising edge", 9.17 µs | The meaningful instant is the I2C START, 6.61 µs — one clock period earlier, 39% overstated | Chose an easy-to-code definition instead of the physically correct one |
| 21 | Concluded "no drift, t = 1.50, not significant" | With a 10× better estimator the same data gives t = 4.69 — the drift was always there | Treated "not detected" as "not present" when the instrument was too coarse |
| 22 | Used first-and-last edge only, discarding 601 of 603 edges | Least-squares over all edges is ~10× more precise; the run-to-run SD of 46 ppm was the *method*, not the sensor | Did not ask whether the estimator was wasting data |
| 23 | Titled the drift figure "after power-on" and joined the points with lines | The board had been running for an unrecorded time, and there is no data between points | Figure claimed more than the data supported |
| 24 | Stated the sensor has one internal tick shared by both modes | The internal period differs by 698 ppm between modes (§3.1b) | A 0.07% discrepancy was invisible to the coarse instrument, and the model was accepted rather than tested |
| 25 | Predicted INT would lift 4×4 to 60 Hz | INT changed the rate by 0.08 Hz; the payload cut did it | Assumed the 8×8 finding transferred; it is a deadline, not a proportion |
| 26 | Predicted rows E–G would drop 1.36% with the new clock | Period is quantised to the sensor tick, so it barely moved | Applied a scale factor to a quantised quantity |
| 27 | Saw 4,177 ≠ 11,443 and explained it as *"the note confused frames with zones"* | The two numbers are from **two different datasets**; ST trained on an unpublished internal one. Nothing to do with units | **Invented a plausible-sounding explanation instead of tracing the number to its source.** The zone figures happened to fit, which made the wrong answer feel right |
| 28 | Reported "STSW-IMG035 is a new finding that may threaten contribution #5" | It was already in §5.2, researched in more depth than the fresh search | Read this file with a truncated view and did not notice the gap |
| 29 | Ran a 6-hour cold-start capture believing the board was in 4×4 mode | The board ran a stale 8×8 binary: `my_tof.h` was edited at 11:53 but never rebuilt, and the `.elf` was still from the previous day (09/10 16:53). Detected only at the analyse step, when ppm came back as −748,025 (≈15.12 Hz = 8×8, not 60 Hz = 4×4) | Treated editing the source as equivalent to flashing the device; no pre-run check that the binary matched the intended configuration |
| 30 | Built H7 predicting a monotonic exponential rise of +700 to +950 ppm | The real curve is three-phase and the net change is **−154.6 ppm** — opposite sign. H7 rejected on all five counts | Extrapolated the curve's shape from the 92-minute window of 10 Sep, which happened to capture only the decay phase and missed the peak at t = 18 min |
| 31 | Treated "cold start" as a reproducible condition without instrumenting it | Two 4×4 cold starts began 1100 ppm apart (6038 ppm on 10 Sep vs 7139.8 ppm on 11 Sep). Neither run logged ambient temperature or power-off duration, so the cause cannot be determined | Assumed an uncontrolled variable was controlled because it had a name |
| 32 | Predicted the H6 mode difference would land at **1,000–1,400 ppm** after removing drift | **700.2 ± 11.3 ppm** — 300 ppm below the predicted floor, and corroborated at 698 ppm by the independent period-ratio route. H6's direction held; its magnitude did not | Same root cause as #19 and #30: extrapolated drift linearly across the 28-minute gap between the two mode measurements, when the drift was already known to be non-linear. Logged separately because #19 covers the drift estimate itself, not the mode-difference prediction built on top of it |
| 33 | Wrote a data loader assuming `glob_data` held the class name as text (matching how the field was documented) | The real npz files store a numeric code in `glob_data` (e.g. `27.0`) with no documented meaning; the class name is only recoverable from the folder path (`dataset_dir/<ClassName>/log__.../npz/`). First real run loaded 162/162 files but **0 usable frames** | Wrote the loader from documentation alone without ever opening a real `.npz` file first; `inspect_npz.py` (written after the failure, not before) would have caught this immediately |
| 34 | Told the student the FlatHand class was collected at a median distance of 155 mm, based on one sample frame inspected 3 Sep | Full-dataset measurement (15 Sep, all 1,738 FlatHand frames) gives **278.5 mm** — 123 mm off, and enough to reverse the ordering versus Fist (262.5 mm) that the 155 mm figure implied | Generalised a single-frame observation to a per-class fact without checking it against the full dataset, which was already downloaded and available |
| 35 | Started a live hand-tracking test against a `production_model.keras` trained on 8×8 data while the board was still flashed with the 4×4 firmware left over from the Phase 1 drift experiment | Caught before any prediction was trusted, by counting comma-separated values in a live `F,` line (16, not 64) | Did not re-verify the board's firmware configuration matched the new task's requirements before starting; same root cause as #29 (assumed a prior setting still held) |
| 36 | Told the student to fix the MCU mismatch by clicking the board-name breadcrumb inside the existing `.ioc` in CubeMX | That workflow creates a brand-new untitled project rather than editing the existing one — confirmed twice by the same "Untitled" project appearing both times it was tried | Assumed a CubeMX workflow without having verified it works the way described in this CubeMX version |
| 37 | Called `ai_network_create_and_init()` passing the weights buffer's own pointer as the weights argument | X-CUBE-AI's generated runtime already knows the weights' location from the embedded `s_network_weights_array_u64[]` table; passing an address manually corrupted the pointer table. Every inference output was NaN | Assumed the same manual-buffer-management pattern used for activations (which genuinely does need a caller-supplied buffer) also applied to weights, without checking the generated `network_data.h` API, which expects `NULL` when weights are compiled in |
| 38 | Wrote `g_class_names[]` in `my_tof.c` ordered by ST's numeric posture label (matching `st_dataset.py`'s sort order) | `train_st_cnn2d.py` — the script that actually produced `production_model.keras` — defines `CLASSES` in a different, non-numeric order. The two scripts serve different purposes and were never meant to share an ordering. On-device predictions were computing the correct class index throughout, but printing the wrong label for it | Copied a class-ordering convention from a data-analysis script instead of the training script that owns the model's output layer, and did not cross-check until a live test (`live_predict.py`, which does use `train_st_cnn2d.py`'s order) gave a different answer than the board for the same gesture |
| 39 | Wrote in a draft progress-report sentence that "the model uses Flash 32.5% / RAM 14.8%" | Those are the whole firmware's `.elf` footprint (`arm-none-eabi-size`: model + sensor driver + HAL + startup), not the model alone. Caught on whole-document review, 21 Sep 2026 | Shortened a sentence and dropped the noun that said what was measured |
| 40 | Described the 1.46 ms in §3.1 Headroom as an estimate of our model before deployment | It is ST's published figure for the shipped 8-class st_cnn2d on NUCLEO-F401RE (§5.1). Our measured on-device inference is 3.197 ms (7-class production_model, X-CUBE-AI 10.2.1, F411RE @ 84 MHz HSE, DWT). The ~2.2× gap is NOT yet explained (§8.5 D2) | Characterised a number without re-opening its source |
| 41 | Told the student to run `python live_predict.py --port COM3 --model results_phase2\production_model.keras` from his home folder | The script is in the repo (`pc_tools\`), and the deployed model is `results_phase2_final\production_model.keras`; root `results_phase2\` holds no model at all (§5.1d-F, §2.4). Found by `dir /s /b` and `convert_to_tflite.py`, 24 Sep 2026 | Copied the path from the script's docstring example instead of locating the files first — same family as #15 (guessed the project's structure) |
| 42 | Said "ห้าม orientation เปลี่ยน" in a way that read as "never move the board / never change workplace" | Only the hand-vs-sensor geometry needs to be held constant **within one test session**; moving the setup between desks is irrelevant. Chin, who moves the setup constantly, caught it | Imprecise statement of a control variable |
| 43 | Proposed reading each live trial after "3 consecutive identical lines" | The vote buffer is a sliding 15-frame window, so consecutive printed lines overlap 14/15 — 3 lines is barely more than one. Chin's rule (same label for ≈1 s ≈ 15 lines, agreement ≥70–80 %) actually flushes the previous posture; adopted (§5.1e-R) | Did not reason through what an overlapping window means for consecutive outputs |
| 44 | Discussed "natural positioning when training" as if Phase 4 involved training | Phase 4 is **test-only** by its own pre-registration (§5.1e controls); training on our data is Phase 5.5. Corrected when Chin asked what Phase 4 is | Followed the student's word "เทรน" without re-checking §5.1e |
| 45 | This file (§8, written 11 Sep) stated "The probe wires have been removed" | Still attached as of 28 Sep 2026 `[STUDENT]` | Recorded an assumed end-of-phase state as a fact without confirmation |
| 46 | Dated the 21-trial test "23 Sep" in text suggested for this file | Upload timestamps (24 Sep 22:46/22:56) and the TensorFlow log (25 Sep 00:10) place it on 24–25 Sep 2026 | Wrote a date from assumption instead of from a timestamp |
| 47 | Wrote `collect_own_data.py` assuming `reset_input_buffer()` empties everything, and tested it only on simulated serial data | On the board (8 Oct dry run) 1–2 frames from before Enter survived the reset and became the first frames of the next take ("missing" 291 / 3683 frames, FlatHand starting at Fist's 217/215 mm). Fixed by a 0.5 s drain plus dropping leading frames before a frame-number gap; v4 records from the same stream as arming, so the issue cannot recur. The assistant also predicted the stale frames would look like "no hand" — wrong, they came from the previous posture | A fake serial port only reproduces the behaviour you already imagined — dry-run on hardware before trusting a data-collection script |
| 48 | First script version treated any answer other than `y` as "skip this take" | Chin typed `ัyy` with the Thai keyboard active and a Fist take was silently lost. Now: y/r/s with Thai-key equivalents (ั/พ/ห), unknown answers re-asked | Designed a destructive default for a typo-prone prompt |
| 49 | Gave `findstr /n "HOLD_S = 3.0" <file>` to check the file version | Without `/c:`, findstr treats each space-separated word as an OR search, so it printed ~200 lines; correct form `findstr /n /c:"HOLD_S = 3.0" <file>`. The output still confirmed the version (line 104), 9 Oct 2026 | Gave a Windows command from memory without checking its quoting semantics |
| 50 | First `evaluate_phase4.py` loaded `st_dataset.py` with `importlib` without registering it in `sys.modules`; `@dataclass` + `from __future__ import annotations` then crashed (`'NoneType' object has no attribute '__dict__'`) on Chin's PC, 9 Oct 2026 | Fixed by `sys.modules[name] = mod` before `exec_module`. The local test had used a trimmed copy of `st_dataset.py` without the dataclass, so it could not reproduce the crash; the test now includes it | Tested against a trimmed copy instead of the real file — same family as #47 (simulation only reproduces what you put in) |
| 51 | `train_st_cnn2d.py` (Phase 2, written by the assistant) used `model.fit(validation_split=0.1)` on arrays ordered by class folder | Keras takes the last 10 % before shuffling (confirmed from Keras 3.15.1 source on Chin's PC, 10 Oct); with alphabetical class order that is ≈ all of Love, so Love was starved in the deployed model and in the Phase 2 LOSO runs. Exposed 10 Oct 2026 by the pre-registered sanity check S1 of Phase 5.5a (§5.1g-R) | Used a library default without checking what it does to ordered data — always shuffle (or split explicitly, stratified) before any "last n %" split |
| 52 | Gave two Keras-internals commands from memory on 10 Oct (`keras.Model.fit.__doc__` → None in Keras 3; `data_adapter_utils.train_validation_split` → moved to `array_slicing.py`) | Found by searching the installed package for `def train_validation_split` instead of guessing its location | Same family as #49 — a command written from memory for a version not checked; for library internals, search the installed files first |
| 53 | Designed `collect_own_data.py` v4 to save any take that raised no `[WARNING]` without asking the operator | The script can check distance and frame loss but not posture correctness. The friend's first extra set (15 Oct 00:01–00:10) contained one wrong posture; Chin noticed afterwards but not which take, so all 14 takes were quarantined (A9). Fixed in v5: ask after every take | Automated everything the script could measure and assumed the one thing it cannot measure — label correctness — was covered; a human confirmation belongs exactly where the human is the only sensor |
| 54 | Said (and the v4 docstring says) that Thai-keyboard letters ั/พ/ห work for y/r/s, tested only on a fake `input()` | On Chin's Windows PC (15 Oct, first prompt of the session) a Thai-keyboard `ั` came back as "Unknown answer". The re-ask worked and nothing was lost; the cause was not diagnosed (console encoding is a guess). Rule: switch to the English keyboard before starting | Same family as #47 and #50 — a simulation reproduces only what you imagined; the assumption was carried over from the #48 fix |
| 55 | Told Chin (16 Oct) that the friend's extras must be split over two days and guessed that 28 takes in one sitting would tire him | A7's revision is a reasoned preference and §5.1e-P step 5 says "if possible"; Chin himself did 28 takes in one evening on 9 Oct (board on 19:47, last take 23:11). The fatigue claim had no measurement. Chin pushed back with the right fact; decision recorded in A10 | Turned a recorded preference into a rule and stated a risk with no evidence (Rules 1 and 2) |
| 56 | Told Chin (20 Oct) "the model has 2,752 parameters" as if it were ours | 2,752 is ST's published 8-class model (§5.1). Our `build_model` (`padding="same"`) has **4,511** parameters for 7 classes `[คำนวณ จากโค้ด]`; 2,752 matches `padding="valid"` (152 + 2,336 + 264). So "exact same architecture as ST" (§5.1, `train_st_cnn2d.py` docstring) is unverified; ST's padding was never recorded. Noticed while explaining "parameters" to Chin from the code. Whether it explains 3.197 vs 1.46 ms (D2): untested | Quoted a number from §5.1 for the wrong model (same family as #40) and the architecture note carried no padding / parameter-count check |
| 57 | Gave `cd /d C:\...` (20 Oct) as "the" command after saying "open cmd" | Chin opened **PowerShell**, where `/d` is rejected; the following `git add/commit/push` ran outside the repo and did nothing (no harm). Fix: `cd "C:\..."` works in both shells | Wrote a shell-specific command without checking which shell the student would use — give commands valid in both cmd and PowerShell |
| 58 | Estimated 8–30 h for the 90-model run before any model had finished | Measured: started 17:12 (TensorFlow log timestamp), report pasted 20:38 → at most ≈ 3 h 26 min on Chin's CPU; the script's own running estimate was of that order | Gave a duration without a measurement — label guesses as `[ประมาณ]` and replace them with the first measured rate |
| 59 | Told Chin (25 Oct) that his Like would "still fail" with E5, quoting the 23.1 % Like recall | 23.1 % is the LOSO fold in which the model had **never seen User 5** (§5.1i-R); E5 is trained on User 5's own 12,664 frames, so that number says nothing about E5 on Chin. Chin's live test on 26 Oct (anecdotal) was all correct | Quoted a figure without the condition it was measured under — same family as #40 and #56 (Rule 5: a number belongs to a model **and** a test condition) |
| 60 | Draft progress report 3 called the 2-take sets "ชุดทดสอบ (ใช้ทดสอบเท่านั้น)" | Both `data\phase4` sets are in the Phase 5.5 final training data (§5.1e-A4, §5.1l); only `data\phase56` is test-only. Renamed ชุดหลัก/ชุดเพิ่มเติม and sections 7.3, 9.1, 11.1 rewritten before submission | Kept the registration-time role of a data set after its role changed — check each data set's **current** use before naming it |
| 61 | Wrote (26 Oct) "the main loop is in `app_tof.c` `[ตรวจแล้ว: build log]`" | The build log shows only that `app_tof.c` is **compiled**; `app_tof.c` is ST's disabled demo and the loop is in `main.c` lines 137–151 | Tagged a conclusion `[ตรวจแล้ว]` when the evidence supported only a weaker statement — the tag must cover the claim, not just the file that was looked at |
| 62 | Told Chin (26 Oct) "the script `compare_board_pc.py` has been sent" and walked him through saving it, before the file was actually sent | He could not find the file card; sent afterwards | Described an action as done before doing it — send the file first, then write the instructions that refer to it |

**Real finding behind #36 (not an assistant mistake — a genuine project fact):**
The `53L8A1_SimpleRanging` project's `.ioc` has always been configured for **STM32F401RETx / NUCLEO-F401RE**, while the physical board (confirmed by a photo of the chip marking) is **STM32F411RET6**. Confirmed via the linker script header: *"Linker script for NUCLEO-F401RE Board embedding STM32F401RETx Device... 96KBytes RAM"*. This has been true since the project was first created and was never the assistant's error to log — it predates this conversation.

**Practical impact, checked before fixing:** RAM actually used throughout Phase 0–2 was ~3 KB, far below even the wrong 96 KB ceiling, so **no measurement in this file was affected**. Fixed 16 Sep 2026 by editing `STM32F401RETX_FLASH.ld` directly (`RAM LENGTH 96K → 128K`), verified against the post-build `.map` file (`Length 0x00020000` = 131,072 B = 128 KB exactly) rather than trusting the source edit alone. Committed as `ac44da5`.
**Not yet fixed:** the underlying MCU device macro is still F401 throughout the project (only the RAM ceiling was corrected). No functional difference expected since only I2C1/USART2/DMA/GPIO are used (present identically on both parts), but should be corrected properly before Phase 6.5 if a peripheral unique to F411 (e.g. SPI4/5) is ever needed. A sibling file `STM32F401RETX_RAM.ld` (used only for a RAM-execution debug config, never invoked in any build log seen) still says 96 K — deferred, not urgent, since it has never been used.

**Pattern:** answering from memory while presenting it as verified.
**Countermeasure:** the tagging rule at the top of this file.

**Fourth pattern, added 15 Sep — trusting a documented data format over the actual
file.** Errors #33 and #34 both came from treating what PROJECT_FACTS *said* about a
data format as equivalent to what the file *actually contains*. Both were only caught
because a smoke test or a fresh full measurement was run against the real data instead
of trusting the write-up. **Countermeasure:** for anything about a data file's exact
structure or a per-class statistic, prefer re-deriving it from the file directly over
citing an earlier note, especially when that note was based on a small sample.


**Second pattern, added 3 Sep — inventing explanations for discrepancies.**
When two numbers disagree, the only acceptable moves are: open the primary source, or
say `[ยังไม่ได้ตรวจ]` and ask. A hypothesis that "makes sense" is not an answer, and is
more dangerous than saying nothing, because it sounds finished. Error #27 would have
gone into the thesis if the student had not said *"go and find out"*.

**Third pattern, added 11 Sep — believing an intended configuration instead of
verifying the running one.** Errors #29 and #31 are the same shape: a condition was
*named* ("4×4 mode", "cold start") and then treated as *established*. Six hours of
capture were spent on the wrong mode, and a 1100 ppm discrepancy cannot now be
explained because nothing was logged. **Countermeasure:** before any long run,
confirm the built artefact is newer than the source that defines the condition, take
a short capture and check the value lands where expected, and log every environmental
variable that the hypothesis depends on — even the ones assumed to be constant.

**Fifth pattern, added 5 Oct — paths, dates and physical states taken from assumption.**
Errors #41, #45 and #46: a file path copied from a docstring, a date written without a
timestamp, and "probes removed" recorded without anyone confirming it. **Countermeasure:**
locate files with `dir /s /b` before giving a command; date events from file or log
timestamps; mark anything about the physical bench that the assistant cannot see as
`[STUDENT]` and ask Chin when it matters.

**Sixth pattern, added 5 Oct — a test can quietly change which question it answers.**
The 21-trial check was written to test "one orientation for all postures" but, because the
operator repositioned per posture, it answered "a working position exists per posture"
(§5.1e-R). Neither the operator nor the assistant noticed until trial 7. **Countermeasure:**
before a protocol starts, write down what the operator may and may not adjust during it, and
what result would change the conclusion; after it, state which question was actually answered.

---

*Last updated: 2026-09-11 · Phase 0 complete · **Phase 1 COMPLETE**: H1–H7 all
answered, TRAP #12 resolved (the sensor's RC oscillator is the fast one, not the
board), TRAP #20 characterised over a full 6-hour cold start (n = 121, 4×4), five
unplanned findings recorded in §3.1c–e, one firmware bug found and fixed (skip
counter at 8×8), correction log grown from 14 to 32 entries (and four duplicate
numbers in the table corrected on 11 Sep). H7 was pre-registered and rejected on
all five counts: the drift is not a monotonic exponential but a three-phase curve
— rise to a peak of 7476.8 ppm at 18 min, decay to a trough of 6884.1 ppm at
147 min, then a slow rise to a statistically stationary plateau of 6946.8 ± 19.3 ppm
over 240–360 min. A single-exponential fit gives R² = 0.879 but its residuals fail
a runs test at z = −9.03, so the model is rejected despite the apparently good fit.
Write-ups and raw data in `data/phase1_int/RESULTS_phase1.md` and
`data/phase1_drift_v2/RESULTS_phase1_drift.md`. Probes removed; Phase 1 closed. One question consciously deferred rather than
dropped — whether temperature causes the drift — with the resume plan in §8.5 D1.*
*(Note 5 Oct 2026: "Probes removed" in the line above is wrong — §9 #45. Kept as written because
this is the historical log.)*

*Last updated: 2026-09-15 · **Scope confirmed by the main advisor in direct
conversation: static gesture recognition first, dynamic second, downstream use to
be named later** (§7, §8). This supersedes the second advisor's earlier separate
instruction (up to 36 postures paired with spoken phrases via a classmate's board,
with on-board audio playback hardware) — that instruction is not deleted, only
inactive, in case it is revived. Practical effect: Phase 2 proceeds on the 7-class
`ST_VL53L8CX_handposture_dataset` already downloaded and verified (§5.1), which
satisfies the "≤10 classes" constraint. No dataset collection, hardware purchase,
or coordination with the classmate's board should happen under the superseded
instruction unless the main advisor re-confirms it.*

*Phase 2 (same day): trained the ST architecture (7 classes) with 3 seeds per
configuration. Random split (reproducing ST's own method) 97.24% ± 0.34 pp;
subject-independent (leave-one-user-out) 79.73% ± 8.63 pp — a 17.51-point gap, 2.36×
larger than Wang et al. 2023's. A reproducible confusion pattern separates
Fist/FlatHand/Dislike/Like ("hard", thumb-position-limited) from
Love/CrossHands/BreakTime ("easy") across all 4 folds (§5.1d). First live
hardware test followed, with the student's own hand (an unseen "user 5"): caught
a firmware mode mismatch (board still in 4×4 from Phase 1) before it corrupted the
test; found and corrected a data-loader bug (`glob_data` is a numeric code, not
class text — class name comes from the folder path instead); corrected an earlier
single-frame distance estimate (FlatHand 155 mm → true full-dataset median
278.5 mm); and established that all 7 classes were collected at similar distances
(240–300 mm), with 270 mm as a single practical test distance inside every class's
p10–p90 band. Once positioned at ~270 mm, every gesture attempted was recognised
correctly (§5.1c). Tools added: `live_predict.py`, `measure_typical_distance.py`,
and `train_st_cnn2d.py` extended with `--split production` and `--equalize-train`.
Decided to proceed to Phase 3 (embedded deployment) next, per a generated plan
document. 16 Sep 2026: found and fixed a pre-existing MCU configuration bug — the
project had been built for STM32F401RE (96 KB RAM) instead of the physical
STM32F411RE (128 KB RAM), confirmed by a photo of the chip. Fixed the linker
script, verified via the `.map` file, committed as `ac44da5` (correction log #36).
No prior measurement was affected. Next: create a separate CubeMX project for
X-CUBE-AI to convert `production_model.keras`, then integrate the generated C code
into the existing firmware.*

*Phase 3 (21 Sep 2026): embedded deployment complete — the trained model now runs
inference on the MCU itself, no PC in the loop. Converted `production_model.keras`
to `.tflite` (Keras 3.x's format is incompatible with X-CUBE-AI 10.2.1's parser),
generated C code via X-CUBE-AI, and validated 100% agreement with the desktop model
on 50 held-out samples before touching hardware. Integrated into the firmware with
majority-vote decoding over a 15-frame window. Found and fixed two bugs before the
system worked correctly (correction log #37–38): a weights-pointer bug that made
every inference output NaN, and a class-label ordering bug that displayed the wrong
gesture name despite the underlying prediction being correct — caught by
cross-checking against `live_predict.py` running the same model on the same live
sensor data on the PC. Confirmed correct with real-hand testing on the board
(CrossHands: 14/15 vote agreement once the hand was held still). Measured 4-stage
on-device latency via DWT cycle counting (same method as Phase 1): sensor read
13,254 µs (80.0%), preprocessing 126 µs (0.8%), inference 3,197 µs (19.3%),
decision/argmax 2 µs (0.01%); total 16,580 µs per frame, about 25% of the 66.17 ms
frame budget at 15 Hz — not the system's bottleneck. Firmware footprint: Flash
166.2 KiB / 512 KiB (32.5%), RAM 18.9 KiB / 128 KiB (14.8%) via
`arm-none-eabi-size`, leaving headroom for a temporal model in Phase 6.5 (Boner et
al. 2022's comparable TCN design targets <100 KB for this MCU class). Committed and
pushed to GitHub (`c88c0d2`). Next: Phase 4 (closed-loop check — our board's own
sensor data through ST's model).*

*Last updated: 2026-09-21 (evening) · Progress report 2 finished (sections 3–6, 8 pages,
dated 20/09/2569; advisor feedback pending). Correction log grown from 38 to 40 entries
(#39: whole-firmware footprint attributed to the model in a draft; #40: ST's published
1.46 ms treated as our own estimate). §3.1 Headroom marked superseded by the measured
3.197 ms; new open question §8.5 D2 (why 3.197 ms vs 1.46 ms). §5.1 sensor row corrected
(ours is VL53L8CA, not "exact match"); §1.1 chip-name check against UM3120 §1.6. §7
logic-analyser row and §8 plan/contributions brought up to date. §5.1e added: Phase 4
pre-registration DRAFT — the prediction and the number of subjects must be filled in and
dated before any data is collected.*

*Last updated: 2026-09-21 (night) · Phase 4 pre-registration filled in (§5.1e): n = 2 subjects
(pilot), prediction written by Claude at Chin's request, pre-flight checks 2 and 3 verified from
`my_tof.c` / the three Python scripts, check 6 (sensor orientation parity with ST) opened as
UNVERIFIED. New verified facts: §2.1 addendum (status is raw when `FAST_READ = 1`), §5.1b ST label
codes, §8.5 D2 second ST source. Web search of 21 Sep recorded as partial coverage in §5.1e.
`collect_own_data.py` written and tested on simulated data only; not yet run on the board.*

*Last updated: 2026-09-22 (night) · §5.1e item 6 (sensor orientation) updated after Chin found ST's
own `README_DATASETS.md`: it confirms the dataset was recorded with a specific hand-raise orientation
(a diagram, not a rigid board-rotation spec) and the recording tool (STSW-IMG035_EVK, "8x8" preset,
`log_.../npz/*.npz` layout matching `collect_own_data.py`). Acquisition-level settings (ranging
frequency, integration time, sharpener) were not published anywhere found — recorded as still open,
not guessed. Orientation is now to be checked empirically (the n=3-per-posture repeat test proposed in
§5.1e), not by visual comparison of photos.*

*Last updated: 2026-09-22 (later) · Corrected an overstated risk claim from earlier the same day: an
ST community warning about the VL53L5CX (a different, lower-resolution sensor family) not
transferring well was mistakenly generalised into "VL53L8CA transfer is a real risk" in §5.1e's
prediction and open items. Both are now corrected to reflect what ST actually said: VL53L8CA is
described by ST support as a superset covering VL53L8CX and VL53L8CH, not a distinct or lesser part
(§1.1). Phase 4's transfer question remains open and worth testing empirically — but is not backed by
any published ST warning. Added a publication-quality standard for Phase 5 (§5.1e): minimum n = 8–10
subjects, Mann–Whitney U with effect size against ST's 4 leave-one-user-out values, mandatory error
analysis, full reproducibility log — and an explicit note that the n = 2 pilot is a pre-flight check,
not citable evidence. This conversation has grown very long; the student was advised to continue
Phase 4 execution in a new chat, starting from this file.*

*Last updated: 2026-09-22 (latest) · Added §10 Related Work from a full review of the four papers in
Project Knowledge (Bartoli 2024 on our exact sensor family VL53L8CH — DM vs CNH, a third independent
source for ST's ~1.5 ms figure, and the traced source of the "5.34 ms" in Correction Log #8; Wang 2023
and Boner 2022 on dynamic low-pixel ToF gesture recognition, useful for future dynamic-gesture phases
and for how to report per-subject results; the UWB-Gestures paper confirmed as the source behind the
reference GitHub repo). Recorded project history: อ.Sakreya sent the UWB repo + both related papers on
4 Aug 2026 with the message that moved the project from UWB radar to ToF — the origin of the sensor
choice, worth citing in the thesis introduction. Confirmed the UWB reference repo (opened 22 Sep 2026)
contains only PyTorch notebooks on the UWB-Gestures dataset, nothing ToF-related. i-CREATe/gSIC 2026
(19th edition) call-for-papers not yet found by web search as of 22 Sep 2026 — the 18th edition was
24–26 Nov 2025 in Bangkok; recommend emailing secretariat.icreate@nstda.or.th directly to ask whether
the 2026 edition is open, rather than assuming a deadline has passed.*

*Last updated: 2026-09-22 (final pass) · Read the fourth and last advisor-sent paper (Ahmed et al.
2026, radar CNN-SVM + FPGA, §10.4) — same UWB-Gestures-derived dataset as §10.3, offers a structural
parallel to our Phase 3 finding (acquisition time dominates over inference time) worth citing in the
discussion section. Checked `[UM3109]`'s full feature list and **ruled out** the CNH extension
proposed earlier in §10.1: Compact Normalized Histogram is specific to VL53L8CH/VL53L7CH (a different
datasheet, DS14310) and is not available through the driver we use for VL53L8CX/CA — corrected so
this is not proposed as a contribution without first acquiring that hardware. All four advisor-sent
and self-sourced papers currently in Project Knowledge have now been read in full for this file.*

*Last updated: 2026-09-22 (final) · Chin caught a real gap: a sixth paper, `ShortRangeL476RG.pdf`
(Zabierowski et al. 2026, Warsaw University of Technology), existed in Project Knowledge but was
missed by the first review pass — now read in full (§10.4a). It independently confirms our FoI spec
(43.4° × 43.4°, matching §1.1 exactly, from a source outside our own datasheet reading) and uses a
third VL53L8 variant (VL53L8CH) for radar-vs-ToF human-movement sensing, supporting the project's
original UWB→ToF pivot. Full file count re-verified: 11 PDFs in Project Knowledge, all now read;
the two textbooks (Mastering STM32, Bare-Metal Embedded C) are not research papers and are out of
scope for this related-work review. Six research papers have now been read in full for this file:
Bartoli 2024, Wang 2023, Boner 2022, the UWB-Gestures paper, Ahmed 2026, and Zabierowski 2026.*

*Last updated: 2026-10-05 · Full rewrite pass covering 24 Sep – 5 Oct 2026. **Phase 4 pre-flight
check 6 done** (§5.1e-R): 21/21 correct on 24–25 Sep with `results_phase2_final` model and a
seed-42 randomised order — but with per-posture hand positioning disclosed by Chin, so it shows a
reproducible working position exists per posture, not that one position works for all; Love needs
the heart's edge over the sensor, FlatHand is height-sensitive. Mechanism documented from code
(§5.1f): Flatten without global pooling, no re-centring, argmax with no threshold, sliding vote
window not reset. **Subject pool fixed at n = 2** (§5.1e-A1): Phase 5 merged into Phase 4, the
n = 8–10 publication standard marked unattainable, H8 becomes a descriptive case study, the main
quantitative claim moves to Phase 5.5's 6-subject LOSO (with the CA/CX confound caveat).
Collection protocol amendments A2–A6 (natural positioning, informed vs naive subject, one collection
used twice, Love position, labels from intent). Execution plan and figures list in §5.1e-P. New
verified facts: deployed model path (§5.1d-F), PC environment and duplicate `live_predict.py`
(§2.4), no ADC in the system and shield pin note (§1.3). **Probes are still attached** — §8's
11 Sep statement corrected. Advisor's 28 Sep ADC/logic-analyser example recorded with video
details and closed as FYI (§7); GPIO-marker technique kept as optional D3. New deferred items D3,
D4. Correction log grown from 40 to 46 entries (#41–#46), plus fifth and sixth patterns.*

*Last updated: 2026-10-09 · Phase 4 collection tooling finished. `collect_own_data.py` went through
v2 (stale-frame fix, §9 #47), v3 (y/r/s redo flow, §9 #48) and v4 (guided arming: 220–320 mm window,
3 s hold, discard after 8 consecutive out-of-window frames, no frame-level filtering, English screen
text, settings box and summary table for report screenshots — §5.1e-A8). Dry run 3 on the board:
7/7 postures, 150 frames per 10 s take (15.0 Hz), 0 missing, discard verified twice. Amendment A7
adds 4 extra takes per posture per subject in `data\phase4_extra` for Phase 5.5 training only; the
registered H8 test set (2 takes) is unchanged. NUCLEO external-power options from UM1724 recorded in
§1.3 (not bench-tested). Correction log now 49 entries.*

*Last updated: 2026-10-09 23:15 · **First own data collected.** Chin (User 5): main set 14/14 in
`data\phase4` (21:11–21:36, 1 discarded Dislike take, 301–302 frames per take) and extra day 1 14/14
in `data\phase4_extra` (finished 23:11, 0 discarded). A7 revised: extras split 2 + 2 over two days.
Report screenshots user5_1/2/5 captured. Remaining Phase 4: friend (User 6), Chin extra day 2,
evaluation script.*

*Last updated: 2026-10-09 23:50 · **First H8 number (User 5 only, n = 1): 66.3 % per-frame** with the
deployed model; errors concentrated in Like (→ Dislike) and Love (→ BreakTime), consistent with the
position-sensitivity finding; preprocessing verified by two independent implementations (§5.1e-E).
`evaluate_phase4.py` written and fixed once (§9 #50). Next: hand-footprint comparison ours vs ST,
friend's collection.*

*Last updated: 2026-10-10 00:05 · Position diagnosis (§5.1e-E): systematic ~1–2-row vertical offset
between Chin's natural hand position and ST's; shift test recovers Like/Love but breaks Dislike;
flips rule out orientation. Working hypothesis: position distribution shift + position-sensitive model.
Phase 5.5 idea to pre-register: spatial-shift augmentation.*

*Last updated: 2026-10-10 00:45 · Phase 5.5a done: shift augmentation negative (−12.1 pp paired over
3 seeds, §5.1g-R). Its sanity check S1 exposed a Phase 2 bug: `validation_split=0.1` on class-ordered
arrays held out ≈ all Love frames (§9 #51) — deployed model and Phase 2 LOSO numbers need re-making
before H8 can be reported. Love's 0 % no longer attributed to position.*

*2026-10-10 00:45 addendum: Keras' last-10 % validation split confirmed from the installed source (§5.1g-R, §9 #51); §9 #52 added.*

*Last updated: 2026-10-10 01:15 · Phase 2 re-run with the split fix (§5.1h-R): bug confirmed (validation
was 100 % Love); ST LOSO now 85.28 % ± 12.90 (was 79.73); fixed production model on User 5 = 68.8 % —
Love recovered (98.8 %), Like still fails (7.1 %), FlatHand fell to 6.8 %. Per-class results for unseen
subjects are strongly seed-dependent → multi-seed evaluation / ensemble needed. Friend (User 6) collection next.*

*2026-10-10 04:35: session ended; friend's collection postponed to another day (see QUICK STATUS).* 

*Last updated: 2026-10-17 evening · Collection record for 14–17 Oct 2026. Friend's main set 14/14 (14 Oct);
his first 14 extra takes quarantined after one unidentified wrong posture (§5.1e-A9, §9 #53); `collect_own_data.py`
v5 asks the operator to confirm the posture every take; `quarantine_takes.py` added. Chin's extra day 2 done
15 Oct (14/14, one 11-frame loss event, cause unknown, n = 1). Friend's 28 extra takes planned as one sitting
(A10), postponed on 17 Oct. Phase 5.5 pre-registration pushed 15 Oct 00:31 `[STUDENT]` before any friend extra
data. Correction log now 55 entries (#53–#55).*

*Last updated: 2026-10-20 · Phase 4 collection complete (both subjects 42 takes; local backups made, Drive/USB copy not
confirmed). First two-subject H8 evaluation (§5.1e-F): per-frame 66.3 % (User 5) and 75.9 % (User 6) with the deployed,
bug-affected model; P1 and P3 hold, P2 and H9 do not; descriptive, n = 2. New open item: our model has 4,511 parameters vs ST's
published 2,752 (padding "same" vs "valid"), so "same architecture" is unverified (§5.1, §9 #56). Next: Phase 5.5 real run;
progress report 3 planning. Correction log now 56 entries.*

*Last updated: 2026-10-20 (night) · **Phase 5.5 LOSO run done** (§5.1i-R, commit `5c724d9`): 90 models; the pre-registered rule
selects arm B (ST folds 85.61 vs A 84.17, C 81.05); B − A not significant over 6 subjects (Wilcoxon W = 5, p = 0.31). Our subjects
with arm B: User 5 82.8 ± 5.2 %, User 6 86.6 ± 6.0 %. Q1/Q2/Q4/Q5 hold (Q5 by 0.11 pp), Q3 fails — User 5's Like stays low (A 3.2 %,
B 23.1 %). Error analysis (§5.1j): hand-position offset toward row 0 is the best-supported cause. Literature update (§5.1k):
STSW-IMG050, dataset-specific normalisation constants, Hugging Face evidence that ST's model is the `padding="valid"` network.
New §8.6 open list (incl. the skill reminder Chin asked for). Correction log now 58 entries (#57–#58). Next: new chat → final training
of arm B with saved models → 5.5b → X-CUBE-AI.*

*Last updated: 2026-10-20 22:05 · Final arm-B models trained and pushed (`results_phase55_final\`, `e1f08ee`, `b0bcc01`).
§5.1l pre-registers the deployed model (5-model ensemble E5 if it fits the board, else seed 42) and the Phase 5.6 new-day test set
`data\phase56` (never for training), with predictions B1–B4 and the Love edge/centre live check. Next: build E5 and convert.*

*Last updated: 2026-10-21 · E5 ensemble built and verified (`3fb36f0`); X-CUBE-AI conversion project located; next = step 2.3 (§5.1l).*


*Last updated: 2026-10-26 · **Step 2.3 done** (§5.1m): E5 converted with X-CUBE-AI 10.2.1 — 22,556 params, 73,762 MACC, Flash 111.89 KiB, RAM 11.21 KiB; desktop validation on 200 real board-check frames 200/200, rmse 1.4e-08; code generated. Timing prediction for step 2.4 on record (≈ 29–30 ms total). Live PC demo with E5 by both subjects (anecdotal) and the Phase 5.6 contamination caveat recorded. Progress report 3 finished. Correction log now 60 entries (#59–#60). Next: step 2.4 on the board.*

*Last updated: 2026-10-26 (evening) · **Step 2.4 done — E5 deployed** (§5.1n): firmware 247.4 KiB Flash / 26.6 KiB RAM; per-frame work 29.3 ± 0.04 ms (inference 15.9 ms ≈ 5 × Phase 3, 216 ns/MACC); board vs PC 203/203 with the new `MY_TOF_PRED_LINE` firmware switch and `pc_tools\compare_board_pc.py` (preprocessing via `st_dataset`). Head-in-FoV episode and `-O0` build recorded. Correction log now 62 entries (#61–#62). Next: commit, A5 Love edge/centre, Phase 5.6.*
*2026-10-26 18:00 addendum: step 2.4 commit `84bfea1` pushed.*
*2026-10-26 ≈18:35 addendum: §5.1o (Phase 5.6 amendment A11, board-side logging) pre-registered before any phase56 data.*

*Last updated: 2026-10-26 (night) · **Phase 5.6 User 5 done** (§5.1o amendment, §5.1p): E5 93.0 % per-frame on the new-day set, measured on the board itself (= PC, 4,219/4,219); ST-only 68.2 %, singles 89.0 ± 3.3 %; per-frame work max 29.3 ms; B1–B7 hold; 5.5c not triggered; hand-position shift caveat recorded. Next: commit, User 6 Phase 5.6, A5.*
*2026-10-26 19:20 addendum: Phase 5.6 User 5 commit `9e5e7a4` pushed.*
*2026-10-26 20:20 addendum: **A5 done** (§5.1q A12, §5.1r): Love centre 5/5, edge 5/5 on the board, board = PC 1,501/1,501; collector v6.1 + `a5_eval.py`. Next: commit, then A6 (all-posture position test, §5.1s).*
*2026-10-26 20:23 addendum: A5 commit `441c67a` (28 files: data\a5_love, results_a5, a5_eval.py, collector v6.1, live_demo v3, FACTS). The repo copy of this file predates this line — save the Project copy before the next commit.*
*2026-10-26 20:33 addendum: **A6 pre-registered** (§5.1s) — tools `collect_own_data.py` v6.2 (`--plan` position guide), `a6_plan.py`, `a6_eval.py`, tested on simulated data. Plan file + tools + this file to be committed before any A6 data.*
*2026-10-26 21:18 addendum: A6 pre-registration commit `343c8c2` pushed (plan `data\a6_shift\a6_plan_User5.json`; natural centroids identical to the phase4 values in §5.1p). No A6 data existed at that time.*
*2026-10-26 21:50 addendum: **A13** — live A6 guide abandoned before any data (practice only), replaced by simulated shift A6-sim; **A14** — standing new-day set `data\phase56_stand` pre-registered (§5.1t).*
*2026-10-26 22:05 addendum: **A14 done** (§5.1u): standing E5 98.7 % on the board (= PC 4,221/4,221) vs sitting 93.0 %; take-level U = 106, p = 0.52. Love take 1 82 % (→ BreakTime), Love sat 1 row toward row 0. Next: A6-sim.*
*2026-10-26 22:16 addendum: **A6-sim done** (§5.1v): strong asymmetry — row−1 standing 98.7 → 58.3 %, row−2 ≈ 32–35 %, row+1/+2 tolerated; decision rule TRIGGERED → pre-register 5.5c re-centring next (new chat). A6-sim is not a fair test of re-centring.*
*2026-10-26 22:35 addendum: **5.5c pre-registered** (§5.1w): re-centring arm R vs re-run arm B, 60-model LOSO; tools `recentre.py`, `train_phase55c.py` tested on synthetic data.*
