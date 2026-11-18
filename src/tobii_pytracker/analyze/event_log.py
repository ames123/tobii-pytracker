import ast
from pathlib import Path
import pandas as pd
import numpy as np


def _extract_aoi_from_bboxes(x: float, y: float, raw_bboxes) -> str | None:
    """Dopasowuje punkt (x, y) do gotowych ramek z kolumny objects_bboxes (format tobii-pytracker)."""
    if pd.isna(x) or pd.isna(y) or pd.isna(raw_bboxes):
        return None
    if isinstance(raw_bboxes, str):
        try:
            raw_bboxes = ast.literal_eval(raw_bboxes)
        except Exception:
            return None

    boxes = (
        raw_bboxes.get("image_bboxes")
        or raw_bboxes.get("words")
        or raw_bboxes.get("bboxes")
        or []
        if isinstance(raw_bboxes, dict)
        else raw_bboxes
    )

    for idx, item in enumerate(boxes):
        if not isinstance(item, dict):
            continue
        b = item.get("bbox", item)
        if {"cx", "cy", "w", "h"}.issubset(b.keys()):
            if (b["cx"] - b["w"] / 2) <= x <= (b["cx"] + b["w"] / 2) and (
                b["cy"] - b["h"] / 2
            ) <= y <= (b["cy"] + b["h"] / 2):
                label = item.get("class") or item.get("word") or item.get("label") or f"AOI_{idx}"
                return f"{label}_{idx}" if label in ("grid", "superpixel", "saliency") else str(label)
    return None


def create_event_log(
    source: pd.DataFrame | Path | str,
    task_type: str | None = None,
    aoi_col: str | None = None,
    analyzer: object | None = None,
    participant_col: str = "set_name",
    stimulus_col: str = "slide_index",
    time_col: str = "logged_time",
    x_col: str = "avg_gaze_x",
    y_col: str = "avg_gaze_y",
    export_pm4py_names: bool = False,
    save_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Generuje Event Log (PM4Py) z danych okulograficznych (DataFrame z DataLoader lub ścieżka do CSV).
    Zwraca 12 wymaganych kolumn ze specyfikacji.
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source, sep=None, engine="python")
    else:
        df = source.copy()

    if time_col not in df.columns and "system_time" in df.columns:
        time_col = "system_time"

    # 1. Uruchomienie analizatora z tobii-pytracker (np. ClusterAnalyzer)
    if analyzer is not None:
        df = analyzer.analyze(df)

    # 2. Wyznaczenie obszaru AOI ('activity')
    if aoi_col and aoi_col in df.columns:
        df["activity"] = df[aoi_col]
    elif "cluster" in df.columns:
        df["activity"] = np.where(df["cluster"] == -1, None, "Cluster_" + df["cluster"].astype(str))
    elif "objects_bboxes" in df.columns:
        df["activity"] = [
            _extract_aoi_from_bboxes(x, y, bboxes)
            for x, y, bboxes in zip(df[x_col], df[y_col], df["objects_bboxes"])
        ]
    else:
        raise ValueError(
            "Brak przypisanych obszarów AOI. Podaj 'aoi_col', przekaż 'analyzer' (np. ClusterAnalyzer) "
            "lub dostarcz dane z kolumną 'cluster' / 'objects_bboxes'."
        )

    # 3. Odfiltrowanie szumu i punktów spoza AOI oraz sortowanie po czasie
    df = df[df["activity"].notna() & (df["activity"] != "-1") & (df["activity"] != "None")].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values([participant_col, stimulus_col, time_col]).reset_index(drop=True)

    # 4. Konwersja czasu na Datetime
    if pd.api.types.is_numeric_dtype(df[time_col]):
        max_t = df[time_col].max()
        unit = "us" if max_t > 1e14 else ("ms" if max_t > 1e11 else "s")

        if unit == "s" and max_t < 1e9:
            session_start = pd.to_datetime(
                df[participant_col].astype(str), format="%Y%m%d_%H%M%S", errors="coerce"
            )
            df["timestamp"] = session_start + pd.to_timedelta(df[time_col], unit="s")
            missing = df["timestamp"].isna()
            if missing.any():
                df.loc[missing, "timestamp"] = pd.to_datetime(df.loc[missing, time_col], unit="s")
        else:
            df["timestamp"] = pd.to_datetime(df[time_col], unit=unit)
    else:
        df["timestamp"] = pd.to_datetime(df[time_col])

    # 5. Łączenie kolejnych odczytów z tego samego obszaru AOI w jedną wizytę
    is_new_visit = (
        (df[participant_col] != df[participant_col].shift())
        | (df[stimulus_col] != df[stimulus_col].shift())
        | (df["activity"] != df["activity"].shift())
    )
    df["visit_id"] = is_new_visit.cumsum()

    # 6. Agregacja punktów w ramach jednej wizyty
    file_col = next(
        (c for c in ("screenshot_file", "screenshot_path", "input_data") if c in df.columns),
        None,
    )
    agg_dict = dict(
        activity=("activity", "first"),
        start_timestamp=("timestamp", "first"),
        end_timestamp=("timestamp", "last"),
        mean_x=(x_col, "mean"),
        mean_y=(y_col, "mean"),
        sample_count=(x_col, "count"),
    )
    if file_col:
        agg_dict["stimulus_file"] = (file_col, "first")
    if "task_type" in df.columns:
        agg_dict["task_type"] = ("task_type", "first")

    log = df.groupby([participant_col, stimulus_col, "visit_id"], as_index=False).agg(**agg_dict)

    # 7. Budowa 12 kolumn ze specyfikacji
    log["duration_ms"] = (
        (log["end_timestamp"] - log["start_timestamp"]).dt.total_seconds() * 1000
    ).round().astype(int)

    log["start_timestamp"] = log["start_timestamp"].dt.round("10ms").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-4]
    log["end_timestamp"] = log["end_timestamp"].dt.round("10ms").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-4]

    log["sequence_index"] = log.groupby([participant_col, stimulus_col]).cumcount() + 1
    log["participant_id"] = log[participant_col].astype(str)

    if "stimulus_file" in log.columns:
        log["stimulus_id"] = log["stimulus_file"].astype(str).apply(lambda s: Path(s).name)
    else:
        log["stimulus_id"] = log[stimulus_col].astype(str)

    if task_type is not None:
        log["task_type"] = task_type
    elif "task_type" in log.columns:
        pass
    elif "classification" in df.columns or "user_classification" in df.columns:
        log["task_type"] = "classification"
    else:
        log["task_type"] = pd.NA

    clean_stim = log["stimulus_id"].apply(lambda s: Path(s).stem)
    log["case_id"] = np.where(
        log["task_type"].notna(),
        log["participant_id"] + "_" + clean_stim + "_" + log["task_type"].astype(str),
        log["participant_id"] + "_" + clean_stim,
    )

    log["mean_x"] = log["mean_x"].round(2)
    log["mean_y"] = log["mean_y"].round(2)

    cols = [
        "case_id", "sequence_index", "activity", "start_timestamp", "end_timestamp",
        "duration_ms", "mean_x", "mean_y", "sample_count",
        "participant_id", "stimulus_id", "task_type",
    ]
    log = log[cols]

    if export_pm4py_names:
        log = log.rename(
            columns={
                "case_id": "case:concept:name",
                "activity": "concept:name",
                "start_timestamp": "time:timestamp",
                "participant_id": "org:resource",
            }
        )

    # 8. Zapis do CSV (tworzy nowy plik lub nadpisuje istniejący)
    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(out_path, index=False)

    return log


#do testowania czy działa
if __name__ == "__main__":
    from tobii_pytracker.analyze.data_loader import DataLoader
    from tobii_pytracker.configs.custom_config import CustomConfig
    from tobii_pytracker.analyze.models import ClusterAnalyzer

    config_path = Path("./examples/images/configs/config.yaml")
    data_root = Path("./examples/images")

    config = CustomConfig(str(config_path))
    loader = DataLoader(config, root=data_root)
    gaze_data = loader.get_all_data(flatten=True)
    if isinstance(gaze_data, dict):
        gaze_data = pd.concat(gaze_data.values(), ignore_index=True)

    cluster_model = ClusterAnalyzer(
        output_folder=Path("./analysis_outputs/clusters"),
        eps=30.0,
        min_samples=3,
    )

    out_csv = Path("./analysis_outputs/event_log.csv")
    event_log = create_event_log(
        gaze_data,
        analyzer=cluster_model,
        save_path=out_csv,
    )

    print(f"Zapisano plik: {out_csv.resolve()} (istnieje: {out_csv.exists()}, wierszy: {len(event_log)})")
    print(event_log.head(10))