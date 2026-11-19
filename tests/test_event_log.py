from __future__ import annotations

import tempfile
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

from support import bootstrap_test_environment

ROOT = bootstrap_test_environment()

from tobii_pytracker.analyze.event_log import create_event_log
from tobii_pytracker.analyze.models import ClusterAnalyzer


def run_test(name: str, test_fn) -> bool:
    print(f"TEST: {name} ... ", end="")
    try:
        test_fn()
        print("\033[92m[PASS]\033[0m")
        return True
    except AssertionError as e:
        print(f"\033[91m[FAIL] -> {e}\033[0m")
        return False
    except Exception as e:
        print(f"\033[91m[ERROR] -> {type(e).__name__}: {e}\033[0m")
        traceback.print_exc()
        return False


def test_real_cluster_analyzer_integration():
    """Testowanie integracji z rzeczywistym ClusterAnalyzerem DBSCAN z scikit-learn przy użyciu tobii-pytracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cluster_out = Path(tmpdir) / "clusters"
        analyzer = ClusterAnalyzer(
            output_folder=cluster_out,
            eps=30.0,
            min_samples=3,
        )

        #Dwa skupiska punktów (100, 100) i (800, 500) oraz 1 punkt szumu (400, 400)
        df = pd.DataFrame({
            "set_name": ["20261119_120000"] * 7,
            "slide_index": [0] * 7,
            "screenshot_file": ["memes/waldo_01.png"] * 7,
            "system_time": [1.0, 1.1, 1.2, 2.0, 3.0, 3.2, 3.4],
            "avg_gaze_x": [100.0, 105.0, 102.0, 400.0, 800.0, 805.0, 810.0],
            "avg_gaze_y": [100.0, 102.0, 98.0, 400.0, 500.0, 502.0, 498.0],
            "classification": ["found"] * 7,
        })

        log = create_event_log(df, analyzer=analyzer)

        expected_cols = [
            "case_id", "sequence_index", "activity", "start_timestamp", "end_timestamp",
            "duration_ms", "mean_x", "mean_y", "sample_count",
            "participant_id", "stimulus_id", "task_type",
        ]
        assert list(log.columns) == expected_cols, f"Złe kolumny: {list(log.columns)}"
        assert len(log) == 2, f"Oczekiwano 2 wizyt (szum powinien odpaść), jest {len(log)}"
        assert list(log["activity"]) == ["Cluster_0", "Cluster_1"], f"Złe klastry: {list(log['activity'])}"
        assert list(log["sample_count"]) == [3, 3], f"Zła liczba próbek: {list(log['sample_count'])}"
        assert log.loc[0, "duration_ms"] == 200, f"Oczekiwano 200ms, jest {log.loc[0, 'duration_ms']}"
        assert log.loc[1, "duration_ms"] == 400, f"Oczekiwano 400ms, jest {log.loc[1, 'duration_ms']}"
        assert log.loc[0, "case_id"] == "20261119_120000_waldo_01_classification"


def test_noise_filtering_and_empty_result():
    """Sprawdza odfiltrowanie szumu (-1 / None) oraz zwrócenie pustego DataFrame, gdy brak AOI."""
    df_mixed = pd.DataFrame({
        "set_name": ["P1"] * 4,
        "slide_index": [1] * 4,
        "system_time": [10.0, 10.1, 10.2, 10.3],
        "avg_gaze_x": [100.0, np.nan, np.nan, 120.0],
        "avg_gaze_y": [100.0, 100.0, 100.0, 100.0],
        "cluster": [0, -1, -1, 0],
    })
    log = create_event_log(df_mixed)
    assert len(log) == 1, f"Po wycięciu szumu oczekiwano 1 połączonej wizyty, jest {len(log)}"
    assert log.loc[0, "sample_count"] == 2

    df_only_noise = pd.DataFrame({
        "set_name": ["P1", "P1"],
        "slide_index": [1, 1],
        "system_time": [10.0, 10.1],
        "avg_gaze_x": [100.0, 100.0],
        "avg_gaze_y": [100.0, 100.0],
        "cluster": [-1, -1],
    })
    empty_log = create_event_log(df_only_noise)
    assert empty_log.empty, "Dla samych punktów szumu powinien zwrócić pusty DataFrame"


def test_objects_bboxes_extraction():
    """Sprawdza współpracę z parse_objects_bboxes i point_inside_bbox z tobii_pytracker."""
    #słownik wyśrodkowany: cx, cy, w, h
    bboxes_json = (
        '[{"class": "caption_top", "bbox": {"cx": 100, "cy": 50, "w": 200, "h": 100}}, '
        '{"class": "grid", "bbox": {"cx": 400, "cy": 400, "w": 200, "h": 200}}]'
    )
    df = pd.DataFrame({
        "set_name": ["P1", "P1", "P1"],
        "slide_index": [0, 0, 0],
        "system_time": [1.0, 2.0, 3.0],
        "avg_gaze_x": [50.0, 400.0, 999.0],
        "avg_gaze_y": [50.0, 400.0, 999.0],
        "objects_bboxes": [bboxes_json, bboxes_json, bboxes_json],
    })

    log = create_event_log(df)
    assert len(log) == 2, f"Oczekiwano 2 wizyt (trzecia poza bbox), jest {len(log)}"
    assert list(log["activity"]) == ["caption_top", "grid_1"], f"Złe nazwy AOI: {list(log['activity'])}"

def test_custom_aoi_col_and_pm4py_names():
    """Sprawdza własną kolumnę aoi_col, brak task_type oraz mapowanie nazw na standard PM4Py."""
    df = pd.DataFrame({
        "set_name": ["UserA", "UserA"],
        "slide_index": [5, 5],
        "system_time": [1700000000000, 1700000000500],  # milisekundy (ms)
        "avg_gaze_x": [10.0, 20.0],
        "avg_gaze_y": [30.0, 40.0],
        "custom_region": ["Header", "Footer"],
    })

    log = create_event_log(df, aoi_col="custom_region", export_pm4py_names=True)
    for col in ("case:concept:name", "concept:name", "time:timestamp", "org:resource"):
        assert col in log.columns, f"Brak kolumny PM4Py '{col}'"
    assert log.loc[0, "case:concept:name"] == "UserA_5"


def test_csv_read_and_write_roundtrip():
    """Sprawdza wczytanie CSV z dysku, mikrosekundy (us) oraz zapis event_log.csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_csv = tmp / "raw_gaze.csv"
        output_csv = tmp / "nested_dir" / "event_log.csv"

        raw_df = pd.DataFrame({
            "set_name": ["20261119_153000", "20261119_153000"],
            "slide_index": [0, 0],
            "system_time": [1732030200000000, 1732030200250000],  # mikrosekundy (us)
            "avg_gaze_x": [150.1234, 155.5678],
            "avg_gaze_y": [250.9876, 252.1111],
            "cluster": [2, 2],
        })
        raw_df.to_csv(input_csv, index=False)

        create_event_log(input_csv, task_type="visual_search", save_path=output_csv)

        assert output_csv.exists(), "Plik wyjściowy CSV nie został utworzony"
        reloaded = pd.read_csv(output_csv)
        assert len(reloaded) == 1
        assert reloaded.loc[0, "duration_ms"] == 250
        assert reloaded.loc[0, "mean_x"] == 152.85
        assert reloaded.loc[0, "task_type"] == "visual_search"


def test_missing_aoi_raises_value_error():
    """Sprawdza rzucenie ValueError przy braku kolumn AOI."""
    df = pd.DataFrame({
        "set_name": ["P1"],
        "slide_index": [0],
        "system_time": [1.0],
        "avg_gaze_x": [100.0],
        "avg_gaze_y": [100.0],
    })
    try:
        create_event_log(df)
        raise AssertionError("Oczekiwano wyjątku ValueError przy braku kolumn AOI")
    except ValueError:
        pass


def test_unsorted_input_and_sequence_reset():
    """Sprawdza sortowanie po czasie oraz resetowanie sequence_index między bodźcami."""
    df = pd.DataFrame({
        "set_name": ["P2", "P1", "P1", "P1"],
        "slide_index": [0, 1, 0, 0],
        "system_time": [5.0, 2.0, 1.5, 1.0],
        "avg_gaze_x": [100, 100, 600, 100],
        "avg_gaze_y": [100, 100, 100, 100],
        "cluster": [0, 0, 1, 0],
    })

    log = create_event_log(df)
    assert len(log) == 4
    assert list(log["sequence_index"]) == [1, 2, 1, 1]


if __name__ == "__main__":
    print("=" * 68)
    print(f"ROOT REPOZYTORIUM: {ROOT}")
    print("URUCHAMIANIE TESTÓW create_event_log()")
    print("=" * 68)
    
tests = [
        ("Integracja z prawdziwym ClusterAnalyzer (DBSCAN)", test_real_cluster_analyzer_integration),
        ("Filtrowanie szumu (-1) i pusty wynik", test_noise_filtering_and_empty_result),
        ("Dopasowanie punktów do ramek objects_bboxes", test_objects_bboxes_extraction),
        ("Własna kolumna aoi_col i eksport nazw PM4Py", test_custom_aoi_col_and_pm4py_names),
        ("Odczyt/zapis CSV z dysku i mikrosekundy (us)", test_csv_read_and_write_roundtrip),
        ("Obsługa błędu przy braku kolumn AOI (ValueError)", test_missing_aoi_raises_value_error),
        ("Sortowanie nieuporządkowanych danych i reset sequence_index", test_unsorted_input_and_sequence_reset),
    ]

passed = sum(run_test(name, fn) for name, fn in tests)
print("=" * 68)
print(f"PODSUMOWANIE: {passed}/{len(tests)} testów zakończonych sukcesem.")
print("=" * 68)