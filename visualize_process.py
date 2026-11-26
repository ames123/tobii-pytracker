from pathlib import Path
import pandas as pd
import pm4py
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

from tobii_pytracker.analyze.data_loader import DataLoader
from tobii_pytracker.analyze.event_log import create_event_log
from tobii_pytracker.analyze.models import ClusterAnalyzer
from tobii_pytracker.configs.custom_config import CustomConfig


def main():
    config_path = Path("./examples/images/configs/config.yaml")
    data_root = Path("./examples/images")
    out_dir = Path("./analysis_outputs")
    clusters_dir = out_dir / "clusters"

    out_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir.mkdir(parents=True, exist_ok=True)

    print("Wczytywanie danych z Tobii-Pytracker...")
    config = CustomConfig(str(config_path))
    loader = DataLoader(config, root=data_root)
    gaze_data = loader.get_all_data(flatten=True)

    if isinstance(gaze_data, dict):
        gaze_data = pd.concat(gaze_data.values(), ignore_index=True)

    print("Tworzenie Event Loga...")
    cluster_model = ClusterAnalyzer(
        output_folder=clusters_dir,
        eps=30.0,
        min_samples=3,
    )

    out_csv = out_dir / "event_log.csv"
    event_log = create_event_log(
        gaze_data,
        analyzer=cluster_model,
        save_path=out_csv,
    )

    if event_log.empty:
        raise RuntimeError(
            "Event log jest pusty – zmniejsz min_samples lub zwiększ eps w ClusterAnalyzer."
        )

    stimuli = event_log["stimulus_id"].unique()
    print(f"Znalezione obrazy w logu: {stimuli}")

    for stim_id in stimuli:
        print(f"Generowanie wykresów dla obrazu: {stim_id}...")
        df_stim = event_log[event_log["stimulus_id"] == stim_id].copy()

        df_stim["start_timestamp"] = pd.to_datetime(df_stim["start_timestamp"])
        df_stim["end_timestamp"] = pd.to_datetime(df_stim["end_timestamp"])

        log = pm4py.format_dataframe(
            df_stim,
            case_id="case_id",
            activity_key="activity",
            timestamp_key="end_timestamp",
            start_timestamp_key="start_timestamp",
        )

        clean_name = Path(str(stim_id)).stem

        # Częstotliwość przejść (Directly-Follows Graph)
        dfg, start_act, end_act = pm4py.discover_dfg(log)
        pm4py.save_vis_dfg(
            dfg,
            start_act,
            end_act,
            str(out_dir / f"{clean_name}_1_czestotliwosc.png"),
        )

        # Czas trwania wizyt i przejść (Performance DFG)
        perf_dfg, p_start, p_end = pm4py.discover_performance_dfg(log)
        pm4py.save_vis_performance_dfg(
            perf_dfg,
            p_start,
            p_end,
            str(out_dir / f"{clean_name}_2_czas.png"),
        )

        # Sieć heurystyczna (odcina szum pojedynczych spojrzeń)
        heu_net = pm4py.discover_heuristics_net(log, dependency_threshold=0.5)
        pm4py.save_vis_heuristics_net(
            heu_net,
            str(out_dir / f"{clean_name}_3_heurystyka.png"),
        )

    print(f"\nGotowe! Wszystkie pliki (CSV oraz wykresy PNG) zapisano w folderze: {out_dir.resolve()}")


if __name__ == "__main__":
    main()