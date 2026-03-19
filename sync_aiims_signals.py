#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


FS_ECG = 200
SYNC_SENSORS = ("metaData", "accelerometer", "gyroscope", "magnetometer")
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
    ppg_path: Path
    files: Dict[str, Path]
    ppg_df: pd.DataFrame
    meta_df: pd.DataFrame | None
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
    for sensor_name in SYNC_SENSORS:
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
    meta_df: pd.DataFrame | None = None

    for sensor_name in SYNC_SENSORS:
        sensor_path = device_files[sensor_name]
        if not sensor_path.exists():
            continue
        sensor_df = pd.read_csv(sensor_path)
        sensor_df.attrs["source_path"] = sensor_path
        if sensor_name == "metaData":
            meta_df = sensor_df
        else:
            sensor_dfs[sensor_name] = sensor_df

    return DeviceBundle(
        label=device_label,
        ppg_path=device_files["ppg"],
        files=device_files,
        ppg_df=ppg_df,
        meta_df=meta_df,
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
    synced_df.to_csv(output_path, index=False)


def sync_record(
    record: RecordConfig,
    ecg_fs: float,
    dry_run: bool = False,
    output_root: Path | None = None,
) -> list[Path]:
    base_path = record.base_path
    ecg_path, ppg1_path, ppg2_path = discover_recording_files(base_path)

    ecg_df = pd.read_csv(ecg_path)
    ecg_time_ns = ecg_df["time"].to_numpy(np.int64)
    sync_parent = resolve_sync_parent(record, output_root)
    sync_dir = sync_parent / "sync"

    if dry_run:
        print(f"[DRY RUN] {base_path}")
        print(f"  Output: {sync_dir}")
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

    sync_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []

    for device, map_info, prefix in [
        (device_1, map_device_1, "ppg1"),
        (device_2, map_device_2, "ppg2"),
    ]:
        ppg_columns = pick_numeric_value_columns(device.ppg_df)
        if not ppg_columns:
            raise ValueError(f"No numeric PPG columns found in {device.ppg_path}")
        synced_ppg = resample_signal_to_ecg(device.ppg_df, ecg_time_ns, map_info, ppg_columns)
        ppg_output = sync_dir / f"{prefix}_synced_ppg.csv"
        write_synced_csv(ppg_output, synced_ppg)
        written_files.append(ppg_output)

        if device.meta_df is not None:
            meta_columns = pick_numeric_value_columns(device.meta_df)
            if meta_columns:
                synced_meta = resample_signal_to_ecg(device.meta_df, ecg_time_ns, map_info, meta_columns)
                meta_output = sync_dir / f"{prefix}_synced_metadata.csv"
                write_synced_csv(meta_output, synced_meta)
                written_files.append(meta_output)

        for sensor_name, sensor_df in device.sensor_dfs.items():
            sensor_columns = pick_numeric_value_columns(sensor_df)
            if not sensor_columns:
                continue
            renamed_columns = [f"{sensor_name}_{col}_resampled" for col in sensor_columns]
            synced_sensor = resample_signal_to_ecg(sensor_df, ecg_time_ns, map_info, sensor_columns)
            synced_sensor.columns = ["time", *renamed_columns]
            sensor_output = sync_dir / f"{prefix}_synced_{sensor_name}.csv"
            write_synced_csv(sensor_output, synced_sensor)
            written_files.append(sensor_output)

    drift_summary = pd.DataFrame(
        [
            {
                "device": "ppg1",
                "clock_scale_a": map_device_1["a"],
                "clock_offset_b_rel_sec": map_device_1["b_rel"],
                "clock_drift_ppm": (map_device_1["a"] - 1.0) * 1e6,
                "ppg_anchor_1": record.ppg1_anchor_indices[0],
                "ppg_anchor_2": record.ppg1_anchor_indices[1],
                "ecg_anchor_1": record.ecg_anchor_indices[0],
                "ecg_anchor_2": record.ecg_anchor_indices[1],
            },
            {
                "device": "ppg2",
                "clock_scale_a": map_device_2["a"],
                "clock_offset_b_rel_sec": map_device_2["b_rel"],
                "clock_drift_ppm": (map_device_2["a"] - 1.0) * 1e6,
                "ppg_anchor_1": record.ppg2_anchor_indices[0],
                "ppg_anchor_2": record.ppg2_anchor_indices[1],
                "ecg_anchor_1": record.ecg_anchor_indices[0],
                "ecg_anchor_2": record.ecg_anchor_indices[1],
            },
        ]
    )
    summary_output = sync_dir / "sync_summary.csv"
    write_synced_csv(summary_output, drift_summary)
    written_files.append(summary_output)

    print(f"Synced {base_path} -> {sync_dir}")
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
            )
        )

    if args.dry_run:
        print(f"Validated {len(records)} record(s).")
    else:
        print(f"Created {len(all_written)} synced file(s) across {len(records)} record(s).")


if __name__ == "__main__":
    main()
