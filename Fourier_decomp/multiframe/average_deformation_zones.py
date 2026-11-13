import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io


FRAME_DIR = Path(
    r"F:\Luca_Data\20261110_250mM_Ch4_70um\250mM_coacervates_after0h_1uls"
)
RESULTS_ROOT = Path(
    r"C:\Users\RHEODLS\Documents\Luca\script_rheoflu\results"
    r"\coacervate_deformation\250mM_coacervates_after0h_1uls"
)
OUTPUT_DIR = RESULTS_ROOT / "spatial_zone_averages"
ZONE_COUNT = 10
CSV_CANDIDATES = ("fourier_result.csv", "fourier_results.csv")
REQUIRED_COLUMNS = {
    "frame",
    "center_x_px",
    "a2",
    "a3",
    "strain",
}
METRICS = ("a2", "a3", "strain")


def get_frame_width(frame_dir: Path) -> int:
    tif_files = sorted(
        path for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not tif_files:
        raise FileNotFoundError(f"Nessun frame TIFF trovato in {frame_dir}")

    frame = io.imread(str(tif_files[0]))
    if frame.ndim < 2:
        raise ValueError(f"Il frame {tif_files[0]} non ha dimensioni immagine valide.")
    return int(frame.shape[1])


def find_results_csv(droplet_dir: Path) -> Path | None:
    for filename in CSV_CANDIDATES:
        candidate = droplet_dir / filename
        if candidate.is_file():
            return candidate
    return None


def read_droplet_rows(csv_path: Path) -> list[dict[str, float | int]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV vuoto o senza intestazione: {csv_path}")

        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Colonne mancanti in {csv_path}: {missing}")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                frame_value = float(row["frame"])
                if not np.isfinite(frame_value) or not frame_value.is_integer():
                    raise ValueError("frame deve essere un numero intero finito")

                values = {
                    "frame": int(frame_value),
                    "center_x_px": float(row["center_x_px"]),
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Valore non valido in {csv_path}, riga {line_number}: {error}"
                ) from error

            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(
                    f"Valore non finito in {csv_path}, riga {line_number}."
                )
            rows.append(values)

    if not rows:
        raise ValueError(f"Nessun dato presente in {csv_path}")
    return rows


def aggregate_zone_data(
    frame_dir: Path = FRAME_DIR,
    results_root: Path = RESULTS_ROOT,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    frame_width = get_frame_width(frame_dir)
    zone_edges = np.linspace(0, frame_width, ZONE_COUNT + 1)
    # First average frames within each droplet and zone, then average droplets equally.
    droplet_zone_data = defaultdict(
        lambda: {"n_frames": 0, **{metric: 0.0 for metric in METRICS}}
    )

    csv_count = 0
    for droplet_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        if droplet_dir.resolve() == output_dir.resolve():
            continue
        csv_path = find_results_csv(droplet_dir)
        if csv_path is None:
            print(f"CSV Fourier non trovato, salto: {droplet_dir}")
            continue

        rows = read_droplet_rows(csv_path)
        csv_count += 1

        for row in rows:
            x_position = row["center_x_px"]
            if not 0 <= x_position <= frame_width:
                raise ValueError(
                    f"center_x_px={x_position} fuori dal frame [0, {frame_width}] "
                    f"in {csv_path}, frame {row['frame']}."
                )

            zone_index = min(
                int(np.searchsorted(zone_edges, x_position, side="right") - 1),
                ZONE_COUNT - 1,
            )
            bucket = droplet_zone_data[(droplet_dir.name, zone_index)]
            bucket["n_frames"] += 1
            for metric in METRICS:
                bucket[metric] += row[metric]

    if csv_count == 0:
        raise FileNotFoundError(
            f"Nessun CSV fourier_result.csv o fourier_results.csv trovato "
            f"nelle sottocartelle di {results_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_droplet_rows = []
    zone_aggregates = {
        zone_index: {"n_droplets": 0, "n_frames": 0, **{metric: 0.0 for metric in METRICS}}
        for zone_index in range(ZONE_COUNT)
    }

    for (droplet_name, zone_index), values in sorted(droplet_zone_data.items()):
        row = {
            "droplet": droplet_name,
            "zone": zone_index + 1,
            "x_min_px": zone_edges[zone_index],
            "x_max_px": zone_edges[zone_index + 1],
            "n_frames": values["n_frames"],
        }
        row.update(
            {
                f"{metric}_mean": values[metric] / values["n_frames"]
                for metric in METRICS
            }
        )
        per_droplet_rows.append(row)

        zone_bucket = zone_aggregates[zone_index]
        zone_bucket["n_droplets"] += 1
        zone_bucket["n_frames"] += values["n_frames"]
        for metric in METRICS:
            zone_bucket[metric] += row[f"{metric}_mean"]

    aggregated_rows = []
    for zone_index, values in zone_aggregates.items():
        row = {
            "zone": zone_index + 1,
            "x_min_px": zone_edges[zone_index],
            "x_max_px": zone_edges[zone_index + 1],
            "n_droplets": values["n_droplets"],
            "n_frames_total": values["n_frames"],
        }
        row.update(
            {
                f"{metric}_mean": (
                    values[metric] / values["n_droplets"]
                    if values["n_droplets"]
                    else None
                )
                for metric in METRICS
            }
        )
        aggregated_rows.append(row)

    summary_csv = output_dir / "zone_averages_across_droplets.csv"
    summary_columns = [
        "zone",
        "x_min_px",
        "x_max_px",
        "n_droplets",
        "n_frames_total",
        "a2_mean",
        "a3_mean",
        "strain_mean",
    ]
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(aggregated_rows)

    per_droplet_csv = output_dir / "zone_averages_by_droplet.csv"
    per_droplet_columns = [
        "droplet",
        "zone",
        "x_min_px",
        "x_max_px",
        "n_frames",
        "a2_mean",
        "a3_mean",
        "strain_mean",
    ]
    with per_droplet_csv.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=per_droplet_columns)
        writer.writeheader()
        writer.writerows(per_droplet_rows)

    for metric in METRICS:
        plotted_rows = [
            row for row in aggregated_rows
            if row["n_droplets"] > 0
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(
            [row["zone"] for row in plotted_rows],
            [row[f"{metric}_mean"] for row in plotted_rows],
            marker="o",
            markersize=5,
            linewidth=1.5,
        )
        ax.set_xticks(range(1, ZONE_COUNT + 1))
        ax.set_xlim(0.5, ZONE_COUNT + 0.5)
        ax.set_xlabel("Numero della fascia")
        ax.set_ylabel(f"{metric} medio tra le gocce")
        ax.set_title(f"{metric} medio per fascia")
        ax.grid(True, linestyle="--", alpha=0.35)
        fig.tight_layout()
        plot_path = output_dir / f"{metric}_mean_by_zone.png"
        fig.savefig(plot_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

    print(
        f"Larghezza frame: {frame_width} px; fasce: {ZONE_COUNT} "
        f"da {frame_width / ZONE_COUNT:.1f} px."
    )
    print(f"Medie per fascia tra gocce salvate in: {summary_csv.resolve()}")
    print(f"Medie per singola goccia e fascia salvate in: {per_droplet_csv.resolve()}")
    print(f"Grafici a2, a3 e strain in funzione della fascia salvati in: {output_dir.resolve()}")
    return summary_csv


if __name__ == "__main__":
    aggregate_zone_data()
