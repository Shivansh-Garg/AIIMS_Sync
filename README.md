# ECG-PPG Synchronization Workflow

This project synchronizes Polar PPG device streams to a Holter ECG timeline and then verifies whether the synchronized signals preserve beat-to-beat timing. The main working notebook is `AIIMs_PPG_ECG_sync_fixed_current_cerebrum.ipynb`.

## Goal

The ECG and PPG devices record using different clocks. Even if both devices begin close together, their timestamps can have:

- an offset, meaning one clock is shifted relative to the other;
- drift, meaning one clock runs slightly faster or slower;
- local noise, missed peaks, or motion artifacts.

The notebook estimates a linear clock map from PPG time to ECG time, resamples PPG and other Polar sensor streams onto the ECG timebase, and verifies the result using peak timing and interval agreement.

## Core Time Mapping

The synchronization model assumes the relation between PPG time and ECG time is approximately linear:

```text
t_ecg = a * t_ppg + b
```

where:

- `t_ppg` is time in the PPG device clock;
- `t_ecg` is time in the ECG clock;
- `a` is the clock scale factor;
- `b` is the clock offset;
- `(a - 1) * 1e6` gives drift in parts per million, ppm.

The notebook fits this model using manually chosen anchor pairs:

```text
(ppg_idx_1, ecg_idx_1)
(ppg_idx_2, ecg_idx_2)
```

The anchor timestamps are converted to relative seconds before fitting:

```text
t_ppg_rel = (ppg_anchor_ns - t_ref_ns) / 1e9
t_ecg_rel = (ecg_anchor_ns - t_ref_ns) / 1e9
```

Then `np.polyfit(t_ppg_rel, t_ecg_rel, deg=1)` estimates:

```text
t_ecg_rel = a * t_ppg_rel + b_rel
```

Relative time is used because raw nanosecond timestamps are very large. Fitting on smaller relative values improves numerical stability and makes the offset easier to interpret.

## Resampling Onto the ECG Timeline

After fitting the clock map, the notebook asks: for each ECG timestamp, what PPG timestamp corresponds to it?

Starting from:

```text
t_ecg_rel = a * t_ppg_rel + b_rel
```

the inverse map is:

```text
t_ppg_rel = (t_ecg_rel - b_rel) / a
```

The notebook uses this inverse time coordinate with `np.interp`:

```text
ppg_resampled = interp(t_ecg_as_ppg_rel, t_ppg_rel, ppg_clean)
```

The output is a PPG signal sampled at the ECG timestamps. That means ECG and PPG arrays can be compared by sample index after synchronization.

## Notebook Sections

### 1. Paths and Data Loading

This section defines the ECG CSV path and the PPG device CSV paths. It also builds device file bundles so related Polar files, such as metadata and motion sensors, can be found consistently.

Logical reason: synchronization needs both the physiological signal and the device timestamp column. Metadata is used to confirm expected sampling behavior, and the same clock map can later be applied to other sensor files from the same device.

### 2. Anchor Indices

This section defines two ECG anchor indices and two PPG anchor indices for each PPG device.

Logical reason: two anchor pairs are enough to estimate both offset and linear drift. One point can only estimate a shift; two points estimate a line.

### 3. Signal Cleaning and ECG R-Peak Detection

The ECG signal is cleaned with NeuroKit, then ECG R-peaks are detected. PPG signals are also cleaned using NeuroKit's PPG cleaning method.

Logical reason: peak-based timing checks are only meaningful if peak detection is performed on cleaned signals. ECG R-peaks are used as the reference because ECG electrical activation is sharper and easier to localize than PPG pulse peaks.

### 4. Stable Clock-Map Fit

This section fits the linear PPG-to-ECG clock map:

```text
t_ecg_rel = a * t_ppg_rel + b_rel
```

It then resamples the PPG signal into ECG time using the inverse map.

Logical reason: this corrects both absolute offset and long-term drift. Without the scale term `a`, a signal may look aligned near one anchor but gradually shift away elsewhere.

### 5. ECG-Guided PPG Peak Detection

For each ECG R-peak, the notebook searches for a PPG peak in a physiologically reasonable delay window:

```text
R_i + 0.10 s <= PPG peak <= R_i + 0.460 s
```

Within that window, it finds locally prominent PPG peaks and selects the earliest sufficiently prominent candidate.

Logical reason: the PPG pulse should arrive after the ECG R-peak because blood pulse arrival is delayed by pulse transit time. The delay gate prevents unrelated PPG peaks from being paired with an ECG beat.

### 6. Device 1 Before/After Drift Correction

This section computes:

- PPG before correction, using direct interpolation in absolute timestamp space;
- PPG after correction, using the fitted clock map;
- PTT before and after correction:

```text
PTT_i = (PPG_peak_i - ECG_R_peak_i) / fs_ecg
```

Residual PTT outliers are filtered with a median absolute deviation rule.

Logical reason: if synchronization improves, the PTT series should become more stable over time. PTT is not expected to be constant, but a strong artificial trend often indicates clock drift.

### 7. Device 2 Processing

Device 2 follows the same mapping, peak detection, and PTT calculation workflow as Device 1.

Logical reason: keeping both devices on the same method makes the comparison fair and prevents device-specific verification logic from hiding sync problems.

### 7a. Interval-Correlation 60-Second Verification For PPG Channels

This is the main sync verification section. The previous ECG R-R versus PPG peak-peak scatter plots were removed. The notebook now uses 60-second time-series checks for the PPG columns present in the data.

For each device, the notebook selects the first available numeric PPG columns from the PPG dataframe, excluding the `time` column. The displayed series names are the real dataframe column names.

For each PPG column, the before-sync calculation uses the original ECG timestamps and original PPG timestamps directly. The after-sync calculation maps the original PPG timestamps into the ECG clock using the fitted clock map.

Each ECG R-peak is paired using the dynamic R-R-window rule:

```text
For the interval [R_i, R_{i+1}), choose the highest-amplitude PPG point.
```

The notebook then builds beat-to-beat interval pairs from consecutive matched beats:

```text
ECG_RR_i = ECG_R_peak_time_{i+1} - ECG_R_peak_time_i
PPG_PP_i = PPG_peak_time_{i+1} - PPG_peak_time_i
```

For every valid 60-second window, it computes:

```text
R2 = corr(ECG_RR, PPG_PP)^2
```

This R2 is calculated separately before sync and after sync for each selected PPG column.

Logical reason: this is a meaningful synchronization-quality check because it tests whether beat-to-beat timing structure is preserved between ECG and PPG. It is better than raw ECG-vs-PPG waveform correlation, because ECG and PPG have different waveform shapes. It should still be interpreted with PTT plots and visual inspection, because local artifacts or missed peaks can lower R2 even when the global clock map is reasonable.

### 7a.3. 60-Second R2 And PTT Plots

The notebook now generates time-series plots instead of scatter plots:

- 60-second R2 over time, where R2 is calculated between ECG R-R intervals and PPG peak-peak intervals;
- before-sync and after-sync R2 curves for each selected PPG column;
- PTT mean over time for each selected PPG column;
- PTT standard deviation over time for each selected PPG column;
- a separate diagnostic cell that compares R2 before and after a chosen time threshold and reports where that threshold lies relative to the sync anchors.

PTT is calculated as:

```text
PTT_i = PPG_peak_time_i - ECG_R_peak_time_i
```

Each ECG R-peak is paired with the highest-amplitude PPG point between that R-peak and the next R-peak. PTT summaries are computed per 60-second window before and after sync.

Logical reason: interval R2 shows whether ECG and PPG preserve the same beat-to-beat rhythm structure, while PTT mean and standard deviation show whether ECG-to-PPG delay is stable over time.

The ECG/PPG overlay plots are kept in separate cells from the R2/PTT graphs. There is one dedicated overlay cell for Device 1 and another for Device 2. These plots mark:

- ECG R-peaks;
- matched PPG peaks;
- dashed ECG-to-PPG correspondence lines showing which PPG peak was paired with each ECG R-peak.

A separate manual visualisation cell lets the user provide arguments for device, PPG column, start time, and window length, making it easier to inspect a specific part of the recording.

### 7a.4. Count-Matched Offset Verification

This section performs an additional synchronization consistency check using beat-count matching between synchronized PPG windows and ECG peak sequences.

For each selected PPG column, the analysis operates on the synchronized ECG and PPG peak times obtained from the interval-correlation workflow.

The synchronized PPG peak sequence is divided into non-overlapping 60-second windows:

```text
window_length = 60 seconds

ECG_RR_i = ECG_R_peak_time_{i+1} - ECG_R_peak_time_i
PPG_PP_i = PPG_peak_time_{i+1} - PPG_peak_time_i

R = corr(ECG_RR, PPG_PP)

Offset =
mean(ECG_peak_time_i - PPG_peak_time_i)

### 8. Windowed ECG/PPG Visual Inspection

This section plots ECG and PPG before and after correction for a user-selected time interval. It overlays detected ECG and PPG peak markers.

Logical reason: numerical metrics can say whether sync is good, but visual inspection helps confirm whether the selected peaks are physiologically reasonable and whether any section has artifacts or missed detections.

### 9. Quick Diagnostics

The notebook summarizes PTT distributions and checks feasible ranges.

Logical reason: even if the clock map is mathematically correct, bad peak selection can still produce unreasonable PTT values. Distribution checks quickly reveal those cases.

### Synced Sensor Export

The same clock map is also applied to other Polar sensor files. For any source sensor timestamp `t_source`, the notebook maps ECG timestamps back into the source clock:

```text
t_source_query = (t_ecg_rel - b_rel) / a
```

Then each sensor column is interpolated onto ECG timestamps and written to synced CSV outputs.

Logical reason: all streams from the same Polar device share the same device clock. Once the PPG-to-ECG clock map is known for that device, accelerometer, gyroscope, magnetometer, and similar files can be aligned to ECG using the same map.

### PPG With Accelerometer Axes

The notebook also plots PPG together with all available accelerometer axes for both devices. The accelerometer streams are resampled with the same fitted clock map:

```text
t_acc_query = (t_ecg_rel - b_rel) / a
```

The improved accelerometer plot shows before-sync and after-sync views side by side. In each view, PPG is shown in the top panel, accelerometer X/Y/Z are shown in stacked panels underneath, and the final row shows the usual combined acceleration magnitude:

```text
ACC_mag = sqrt(ACC_x^2 + ACC_y^2 + ACC_z^2)
```

Logical reason: poor PPG sections are often related to motion or loose sensor contact. Viewing PPG beside accelerometer X/Y/Z and the combined magnitude makes it easier to see whether bad PPG morphology occurs at the same time as movement along any accelerometer axis or overall acceleration bursts.

## Verification Metrics

The notebook uses several complementary checks:

- `a` and drift ppm: tells how different the device clock rate is from ECG.
- `b_rel`: gives the offset between clocks in the fitted relative-time system.
- PTT mean and standard deviation: checks ECG-to-PPG timing stability.
- PTT trend ppm: fits PTT over time; large slope suggests residual drift.
- 60-second interval R2: checks whether ECG R-R intervals and PPG peak-peak intervals agree locally.
- Median and mean absolute interval error: summarize the ECG R-R versus PPG peak-peak mismatch inside each 60-second window.
- Per-column PPG comparison: repeats the same checks for the selected PPG columns found in the dataframe.
- Count-matched best correlation: highest interval correlation obtained between a PPG window and candidate ECG sequences with the same beat count.
- Count-matched offset: average temporal difference between synchronized ECG and PPG peaks for the best-matching ECG sequence.
- Number of ECG candidates evaluated: indicates the search-space size used during count-matched verification.

## How To Interpret The Main 60-Second Graphs

A strong result should show:

- higher after-sync interval R2 than before-sync interval R2;
- stable PTT mean over time;
- lower PTT standard deviation after sync;
- improvement from before sync to after sync.

If after-sync R2 remains low or PTT spread remains high in isolated windows, those sections are likely affected by motion artifacts, weak PPG morphology, or local peak-detection failures. If poor values persist across the recording, the anchor choices or clock-map fit should be revisited.

