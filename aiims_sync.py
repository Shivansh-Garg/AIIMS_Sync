#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import warnings
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


FS_ECG = 200
SYNC_SENSORS = ("accelerometer", "gyroscope", "magnetometer")
TIME_COLUMN_CANDIDATES = (
    "time",
    "timestamp",
    "phoneTimestamp",
    "sensorTimestampNs",
    "sensorTimestamp",
    "systemTimeNs",
    "systemTime",
)


@dataclass
class DeviceBundle:
    label: str
    device_name: str
    ppg_path: Path
    files: Dict[str, Path]
    ppg_df: pd.DataFrame
    sensor_dfs: Dict[str, pd.DataFrame]


@dataclass
class RecordConfig:
    base_path: Path
    output_base_path: Path | None
    sync_deltas: list
    ppg1_sync_boundaries: list[list[int]]
    ppg2_sync_boundaries: list[list[int]]
    ecg_sync_boundaries: list[list[int]]

    @property
    def ppg1_anchor_indices(self) -> list[int]:
        return extract_anchor_indices(self.ppg1_sync_boundaries)

    @property
    def ppg2_anchor_indices(self) -> list[int]:
        return extract_anchor_indices(self.ppg2_sync_boundaries)

    @property
    def ecg_anchor_indices(self) -> list[int]:
        return extract_anchor_indices(self.ecg_sync_boundaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sync AIIMS ECG/PPG/IMU streams using boundary indices exported from the sync notebook."
        )
    )
    parser.add_argument(
        "input_table",
        type=Path,
        help="TSV/CSV/text file containing base_path, sync_deltas, and *_sync_boundaries columns.",
    )
    parser.add_argument(
        "--delimiter",
        default="\t",
        help="Input delimiter. Default: tab.",
    )
    parser.add_argument(
        "--ecg-fs",
        type=float,
        default=FS_ECG,
        help="ECG sampling rate used by the original notebook. Default: 200 Hz.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional directory under which synced outputs are written. If set, each record is written to "
            "<output_root>/<patient>/<record>/sync while preserving the original patient/record folder layout."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and anchors without writing synced files.",
    )
    parser.add_argument(
        "--graphs-dirname",
        default="sync_graphs",
        help="Folder name used under the sync parent to store generated plots. Default: sync_graphs.",
    )
    return parser.parse_args()


def parse_literal(value):
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return value
        return ast.literal_eval(value)
    return value


def extract_anchor_indices(boundaries: Sequence[Sequence[int]]) -> list[int]:
    if len(boundaries) < 2:
        raise ValueError(
            "Expected at least two boundary entries because the 0th element of boundary 0 and boundary 1 "
            "define the anchor indices."
        )
    return [int(boundaries[0][0]), int(boundaries[1][0])]


def load_config_table(path: Path, delimiter: str) -> list[RecordConfig]:
    df = pd.read_csv(path, sep=delimiter)
    required_columns = {
        "base_path",
        "sync_deltas",
        "ppg1_sync_boundaries",
        "ppg2_sync_boundaries",
        "ecg_sync_boundaries",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    records: list[RecordConfig] = []
    for _, row in df.iterrows():
        records.append(
            RecordConfig(
                base_path=Path(str(row["base_path"])).expanduser(),
                output_base_path=(
                    Path(str(row["output_base_path"])).expanduser()
                    if "output_base_path" in row and pd.notna(row["output_base_path"]) and str(row["output_base_path"]).strip()
                    else None
                ),
                sync_deltas=parse_literal(row["sync_deltas"]),
                ppg1_sync_boundaries=parse_literal(row["ppg1_sync_boundaries"]),
                ppg2_sync_boundaries=parse_literal(row["ppg2_sync_boundaries"]),
                ecg_sync_boundaries=parse_literal(row["ecg_sync_boundaries"]),
            )
        )
    return records


def build_device_file_bundle(ppg_path: Path) -> Dict[str, Path]:
    stem = ppg_path.stem
    if not stem.endswith("ppg"):
        raise ValueError(f"Unexpected PPG filename format: {ppg_path.name}")

    prefix = stem[:-3]
    bundle = {"ppg": ppg_path}
    for sensor_name in ("metaData", *SYNC_SENSORS):
        bundle[sensor_name] = ppg_path.with_name(f"{prefix}{sensor_name}.csv")
    return bundle


def discover_recording_files(base_path: Path) -> tuple[Path, Path, Path]:
    holter_dir = base_path / "Holter"
    polar_dir = base_path / "Polar"

    if not holter_dir.exists():
        raise FileNotFoundError(f"Holter directory not found under {base_path}")
    if not polar_dir.exists():
        raise FileNotFoundError(f"Polar directory not found under {base_path}")

    ecg_candidates = sorted(p for p in holter_dir.glob("*.csv") if p.is_file())
    if not ecg_candidates:
        raise FileNotFoundError(f"No ECG CSV found in {holter_dir}")

    ppg_candidates = sorted(
        p for p in polar_dir.glob("*/*ppg.csv") if p.is_file()
    )
    if len(ppg_candidates) < 2:
        raise FileNotFoundError(
            f"Expected at least two PPG CSVs under {polar_dir}, found {len(ppg_candidates)}"
        )

    return ecg_candidates[0], ppg_candidates[0], ppg_candidates[1]


def load_device_streams(device_files: Dict[str, Path], device_label: str) -> DeviceBundle:
    ppg_df = pd.read_csv(device_files["ppg"])
    sensor_dfs: Dict[str, pd.DataFrame] = {}

    for sensor_name in SYNC_SENSORS:
        sensor_path = device_files[sensor_name]
        if not sensor_path.exists():
            continue
        sensor_df = pd.read_csv(sensor_path)
        sensor_df.attrs["source_path"] = sensor_path
        sensor_dfs[sensor_name] = sensor_df

    return DeviceBundle(
        label=device_label,
        device_name=device_files["ppg"].parent.name,
        ppg_path=device_files["ppg"],
        files=device_files,
        ppg_df=ppg_df,
        sensor_dfs=sensor_dfs,
    )


def fit_linear_clock_map_from_anchors(
    ppg_time_ns: np.ndarray,
    ecg_time_ns: np.ndarray,
    ppg_anchor_idx: Sequence[int],
    ecg_anchor_idx: Sequence[int],
) -> dict:
    ppg_anchor_ns = ppg_time_ns[np.asarray(ppg_anchor_idx, dtype=int)]
    ecg_anchor_ns = ecg_time_ns[np.asarray(ecg_anchor_idx, dtype=int)]

    t_ref_ns = int(min(ppg_anchor_ns.min(), ecg_anchor_ns.min()))
    t_ppg_rel = (ppg_anchor_ns - t_ref_ns) / 1e9
    t_ecg_rel = (ecg_anchor_ns - t_ref_ns) / 1e9

    a, b_rel = np.polyfit(t_ppg_rel, t_ecg_rel, deg=1)
    return {
        "a": float(a),
        "b_rel": float(b_rel),
        "t_ref_ns": t_ref_ns,
        "ppg_anchor_ns": ppg_anchor_ns,
        "ecg_anchor_ns": ecg_anchor_ns,
    }


def infer_time_column(source_df: pd.DataFrame) -> str:
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in source_df.columns and pd.api.types.is_numeric_dtype(source_df[candidate]):
            return candidate

    numeric_columns = [
        column for column in source_df.columns if pd.api.types.is_numeric_dtype(source_df[column])
    ]
    if not numeric_columns:
        source_name = source_df.attrs.get("source_path", "<in-memory dataframe>")
        raise ValueError(f"Could not find a numeric time column in {source_name}")

    return numeric_columns[0]


def map_timebase_to_source_rel(target_time_ns: np.ndarray, map_info: dict) -> np.ndarray:
    target_time_rel = (np.asarray(target_time_ns, dtype=np.int64) - map_info["t_ref_ns"]) / 1e9
    return (target_time_rel - map_info["b_rel"]) / map_info["a"]


def resample_signal_to_ecg(
    source_df: pd.DataFrame,
    ecg_time_ns: np.ndarray,
    map_info: dict,
    columns: Sequence[str],
) -> pd.DataFrame:
    time_col = infer_time_column(source_df)
    source_time_rel = (source_df[time_col].to_numpy(np.int64) - map_info["t_ref_ns"]) / 1e9
    source_query_rel = map_timebase_to_source_rel(ecg_time_ns, map_info)

    output = pd.DataFrame({"time": np.asarray(ecg_time_ns, dtype=np.int64)})
    for column in columns:
        output[column] = np.interp(
            source_query_rel,
            source_time_rel,
            source_df[column].to_numpy(dtype=np.float32, copy=False),
        ).astype(np.float32)
    return output


def pick_numeric_value_columns(df: pd.DataFrame) -> list[str]:
    time_col = infer_time_column(df)
    return [
        col
        for col in df.columns
        if col != time_col and pd.api.types.is_numeric_dtype(df[col])
    ]


def resolve_sync_parent(record: RecordConfig, output_root: Path | None) -> Path:
    if record.output_base_path is not None:
        return record.output_base_path
    if output_root is None:
        return record.base_path

    tail_parts = record.base_path.parts[-2:] if len(record.base_path.parts) >= 2 else record.base_path.parts
    return output_root.joinpath(*tail_parts)


def write_synced_csv(output_path: Path, synced_df: pd.DataFrame) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*invalid value encountered in cast.*",
            category=RuntimeWarning,
        )
        cleaned_df = synced_df.replace([np.inf, -np.inf], np.nan)
        cleaned_df.to_csv(output_path, index=False)


def identify_xyz_columns(df: pd.DataFrame) -> tuple[str, str, str] | None:
    time_col = infer_time_column(df)
    numeric_columns = [
        col for col in df.columns
        if col != time_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_columns) < 3:
        return None

    lowered = {col.lower(): col for col in numeric_columns}
    axis_options = [
        ("x", "y", "z"),
        ("accx", "accy", "accz"),
        ("x_axis", "y_axis", "z_axis"),
        ("axis_x", "axis_y", "axis_z"),
    ]
    for x_key, y_key, z_key in axis_options:
        if x_key in lowered and y_key in lowered and z_key in lowered:
            return lowered[x_key], lowered[y_key], lowered[z_key]

    return tuple(numeric_columns[:3])


def save_accelerometer_before_after_plot(
    device: DeviceBundle,
    raw_accel_df: pd.DataFrame,
    synced_accel_df: pd.DataFrame,
    graph_dir: Path,
    prefix: str,
) -> Path | None:
    if importlib.util.find_spec("matplotlib.pyplot") is None:
        print("Skipping accelerometer graph generation because matplotlib is not installed.")
        return None
    plt = importlib.import_module("matplotlib.pyplot")

    raw_axes = identify_xyz_columns(raw_accel_df)
    synced_axes = identify_xyz_columns(synced_accel_df)
    if raw_axes is None or synced_axes is None:
        print(
            f"Skipping accelerometer graph for {device.device_name}: could not infer x/y/z axis columns."
        )
        return None

    raw_time_col = infer_time_column(raw_accel_df)
    synced_time_col = infer_time_column(synced_accel_df)

    raw_t = (raw_accel_df[raw_time_col].to_numpy(np.int64) - int(raw_accel_df[raw_time_col].iloc[0])) / 1e9
    synced_t = (
        synced_accel_df[synced_time_col].to_numpy(np.int64) - int(synced_accel_df[synced_time_col].iloc[0])
    ) / 1e9

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    axis_labels = ["X", "Y", "Z"]
    for idx, axis_label in enumerate(axis_labels):
        raw_col = raw_axes[idx]
        synced_col = synced_axes[idx]

        axes[idx, 0].plot(raw_t, raw_accel_df[raw_col].to_numpy(dtype=float), lw=0.9, color="tab:blue")
        axes[idx, 0].set_title(f"Before sync: {axis_label} ({raw_col})")
        axes[idx, 0].set_ylabel("Acceleration")
        axes[idx, 0].grid(alpha=0.2)

        axes[idx, 1].plot(synced_t, synced_accel_df[synced_col].to_numpy(dtype=float), lw=0.9, color="tab:red")
        axes[idx, 1].set_title(f"After sync: {axis_label} ({synced_col})")
        axes[idx, 1].grid(alpha=0.2)

    axes[2, 0].set_xlabel("Time since start (sec)")
    axes[2, 1].set_xlabel("Time since start (sec)")
    fig.suptitle(f"{device.label} ({device.device_name}) accelerometer: before vs after sync", y=1.02)
    fig.tight_layout()

    graph_dir.mkdir(parents=True, exist_ok=True)
    output_plot_path = graph_dir / f"{prefix}_accelerometer_before_after.png"
    fig.savefig(output_plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_plot_path


IST = timezone(timedelta(hours=5, minutes=30))


def format_time_ns_as_ist(time_ns: int) -> str:
    return datetime.fromtimestamp(time_ns / 1e9, tz=IST).strftime("%d-%m-%Y %H:%M:%S.%f IST")


def compute_sync_summary_row(
    device: DeviceBundle,
    device_prefix: str,
    map_info: dict,
    ppg_anchor_indices: Sequence[int],
    ecg_anchor_indices: Sequence[int],
    ecg_df: pd.DataFrame,
    ecg_time_ns: np.ndarray,
) -> dict:
    ppg_time_ns = device.ppg_df["time"].to_numpy(np.int64)
    mapped_anchor_rel = (ppg_time_ns[np.asarray(ppg_anchor_indices, dtype=int)] - map_info["t_ref_ns"]) / 1e9
    mapped_anchor_rel = map_info["a"] * mapped_anchor_rel + map_info["b_rel"]
    ecg_anchor_rel = (ecg_time_ns[np.asarray(ecg_anchor_indices, dtype=int)] - map_info["t_ref_ns"]) / 1e9
    anchor_error_ms = (mapped_anchor_rel - ecg_anchor_rel) * 1e3

    mapped_ppg_time_ns = np.rint((map_info["a"] * ((ppg_time_ns - map_info["t_ref_ns"]) / 1e9) + map_info["b_rel"]) * 1e9 + map_info["t_ref_ns"]).astype(np.int64)
    overlap_start_ns = max(int(mapped_ppg_time_ns[0]), int(ecg_time_ns[0]))
    overlap_end_ns = min(int(mapped_ppg_time_ns[-1]), int(ecg_time_ns[-1]))
    overlap_duration_sec = max(0.0, (overlap_end_ns - overlap_start_ns) / 1e9)

    ecg_duration_sec = (int(ecg_time_ns[-1]) - int(ecg_time_ns[0])) / 1e9 if len(ecg_time_ns) > 1 else 0.0
    ppg_duration_sec = (int(ppg_time_ns[-1]) - int(ppg_time_ns[0])) / 1e9 if len(ppg_time_ns) > 1 else 0.0
    mapped_duration_sec = (int(mapped_ppg_time_ns[-1]) - int(mapped_ppg_time_ns[0])) / 1e9 if len(mapped_ppg_time_ns) > 1 else 0.0

    ppg_anchor_times_ns = ppg_time_ns[np.asarray(ppg_anchor_indices, dtype=int)]
    ecg_anchor_times_ns = ecg_time_ns[np.asarray(ecg_anchor_indices, dtype=int)]

    return {
        "device": device_prefix,
        "device_name": device.device_name,
        "device_path": str(device.ppg_path.parent),
        "ppg_source_file": device.ppg_path.name,
        "ppg_samples": len(device.ppg_df),
        "ecg_samples": len(ecg_df),
        "ppg_duration_sec": ppg_duration_sec,
        "ecg_duration_sec": ecg_duration_sec,
        "mapped_ppg_duration_sec": mapped_duration_sec,
        "overlap_duration_sec": overlap_duration_sec,
        "clock_scale_a": map_info["a"],
        "clock_offset_b_rel_sec": map_info["b_rel"],
        "clock_drift_ppm": (map_info["a"] - 1.0) * 1e6,
        "anchor_1_error_ms": float(anchor_error_ms[0]),
        "anchor_2_error_ms": float(anchor_error_ms[1]),
        "max_abs_anchor_error_ms": float(np.max(np.abs(anchor_error_ms))),
        "ppg_anchor_1": int(ppg_anchor_indices[0]),
        "ppg_anchor_1_time_ist": format_time_ns_as_ist(int(ppg_anchor_times_ns[0])),
        "ppg_anchor_2": int(ppg_anchor_indices[1]),
        "ppg_anchor_2_time_ist": format_time_ns_as_ist(int(ppg_anchor_times_ns[1])),
        "ecg_anchor_1": int(ecg_anchor_indices[0]),
        "ecg_anchor_1_time_ist": format_time_ns_as_ist(int(ecg_anchor_times_ns[0])),
        "ecg_anchor_2": int(ecg_anchor_indices[1]),
        "ecg_anchor_2_time_ist": format_time_ns_as_ist(int(ecg_anchor_times_ns[1])),
        "sync_success": bool(overlap_duration_sec > 0 and np.all(np.isfinite(anchor_error_ms))),
    }


def sync_record(
    record: RecordConfig,
    ecg_fs: float,
    dry_run: bool = False,
    output_root: Path | None = None,
    graphs_dirname: str = "sync_graphs",
) -> list[Path]:
    base_path = record.base_path
    ecg_path, ppg1_path, ppg2_path = discover_recording_files(base_path)

    ecg_df = pd.read_csv(ecg_path)
    ecg_time_ns = ecg_df["time"].to_numpy(np.int64)
    sync_parent = resolve_sync_parent(record, output_root)
    summary_output_dir = sync_parent / "sync"
    graphs_output_dir = sync_parent / graphs_dirname

    if dry_run:
        print(f"[DRY RUN] {base_path}")
        print(f"  Output summary dir: {summary_output_dir}")
        print(f"  Graphs output dir: {graphs_output_dir}")
        print(f"  Device output dirs: {sync_parent} / <device_name>")
        print(f"  ECG:  {ecg_path}")
        print(f"  PPG1: {ppg1_path}")
        print(f"  PPG2: {ppg2_path}")
        print(f"  Anchors PPG1/ECG: {record.ppg1_anchor_indices} -> {record.ecg_anchor_indices}")
        print(f"  Anchors PPG2/ECG: {record.ppg2_anchor_indices} -> {record.ecg_anchor_indices}")
        return []

    device_1 = load_device_streams(build_device_file_bundle(ppg1_path), "Device 1")
    device_2 = load_device_streams(build_device_file_bundle(ppg2_path), "Device 2")

    map_device_1 = fit_linear_clock_map_from_anchors(
        device_1.ppg_df["time"].to_numpy(np.int64),
        ecg_time_ns,
        record.ppg1_anchor_indices,
        record.ecg_anchor_indices,
    )
    map_device_2 = fit_linear_clock_map_from_anchors(
        device_2.ppg_df["time"].to_numpy(np.int64),
        ecg_time_ns,
        record.ppg2_anchor_indices,
        record.ecg_anchor_indices,
    )

    summary_output_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []

    for device, map_info, prefix in [
        (device_1, map_device_1, "ppg1"),
        (device_2, map_device_2, "ppg2"),
    ]:
        device_output_dir = sync_parent / device.device_name
        ppg_columns = pick_numeric_value_columns(device.ppg_df)
        if not ppg_columns:
            raise ValueError(f"No numeric PPG columns found in {device.ppg_path}")
        synced_ppg = resample_signal_to_ecg(device.ppg_df, ecg_time_ns, map_info, ppg_columns)
        ppg_output = device_output_dir / f"{prefix}_synced_ppg.csv"
        write_synced_csv(ppg_output, synced_ppg)
        written_files.append(ppg_output)

        for sensor_name, sensor_df in device.sensor_dfs.items():
            sensor_columns = pick_numeric_value_columns(sensor_df)
            if not sensor_columns:
                continue
            renamed_columns = [f"{sensor_name}_{col}_resampled" for col in sensor_columns]
            synced_sensor = resample_signal_to_ecg(sensor_df, ecg_time_ns, map_info, sensor_columns)
            synced_sensor.columns = ["time", *renamed_columns]
            sensor_output = device_output_dir / f"{prefix}_synced_{sensor_name}.csv"
            write_synced_csv(sensor_output, synced_sensor)
            written_files.append(sensor_output)

            if sensor_name == "accelerometer":
                plot_path = save_accelerometer_before_after_plot(
                    device=device,
                    raw_accel_df=sensor_df,
                    synced_accel_df=synced_sensor,
                    graph_dir=graphs_output_dir,
                    prefix=prefix,
                )
                if plot_path is not None:
                    written_files.append(plot_path)

    drift_summary = pd.DataFrame(
        [
            compute_sync_summary_row(
                device=device_1,
                device_prefix="ppg1",
                map_info=map_device_1,
                ppg_anchor_indices=record.ppg1_anchor_indices,
                ecg_anchor_indices=record.ecg_anchor_indices,
                ecg_df=ecg_df,
                ecg_time_ns=ecg_time_ns,
            ),
            compute_sync_summary_row(
                device=device_2,
                device_prefix="ppg2",
                map_info=map_device_2,
                ppg_anchor_indices=record.ppg2_anchor_indices,
                ecg_anchor_indices=record.ecg_anchor_indices,
                ecg_df=ecg_df,
                ecg_time_ns=ecg_time_ns,
            ),
        ]
    )
    summary_output = summary_output_dir / "sync_summary.csv"
    write_synced_csv(summary_output, drift_summary)
    written_files.append(summary_output)

    print(f"Synced {base_path} -> summary: {summary_output_dir}")
    for written_file in written_files:
        print(f"  wrote {written_file}")

    return written_files


def main() -> None:
    args = parse_args()
    records = load_config_table(args.input_table, args.delimiter)
    all_written: list[Path] = []
    for record in records:
        all_written.extend(
            sync_record(
                record,
                ecg_fs=args.ecg_fs,
                dry_run=args.dry_run,
                output_root=args.output_root,
                graphs_dirname=args.graphs_dirname,
            )
        )

    if args.dry_run:
        print(f"Validated {len(records)} record(s).")
    else:
        print(f"Created {len(all_written)} synced file(s) across {len(records)} record(s).")


if __name__ == "__main__":
    main()
