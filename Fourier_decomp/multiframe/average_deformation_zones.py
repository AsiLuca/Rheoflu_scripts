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
SQUARED_AMPLITUDE_PLOTS_DIR = OUTPUT_DIR / "squared_amplitude_histograms"
SIGNED_AMPLITUDE_PLOTS_DIR = OUTPUT_DIR / "signed_amplitude_histograms"
SHARED_SCALE_PLOTS_DIR = OUTPUT_DIR / "histograms_shared_y_scale"
SHARED_SIGNED_PLOTS_DIR = SHARED_SCALE_PLOTS_DIR / "signed"
SHARED_SQUARED_PLOTS_DIR = SHARED_SCALE_PLOTS_DIR / "squared"
ZONE_COUNT = 10
IQR_MULTIPLIER = 1.5
IQR_MINIMUM_SAMPLES = 4
CSV_CANDIDATES = ("fourier_result.csv", "fourier_results.csv")
REQUIRED_COLUMNS = {
    "frame",
    "center_x_px",
    "a2",
    "a3",
    "a4",
    "strain",
}
METRICS = ("a2", "a3", "strain")
AMPLITUDES = ("a2", "a3", "a4")
FILTERED_FIELDS = ("a2", "a3", "a4", "strain")
SQUARED_AMPLITUDES = tuple(f"{amplitude}_squared" for amplitude in AMPLITUDES)
SIGNED_AMPLITUDES = (
    "a4_negative",
    "a3_negative",
    "a2_negative",
    "a2_positive",
    "a3_positive",
    "a4_positive",
)


def filter_iqr_outliers(values: list[float]) -> tuple[np.ndarray, int]:
    samples = np.asarray(values, dtype=float)
    if samples.size < IQR_MINIMUM_SAMPLES:
        return samples, 0

    q1, q3 = np.percentile(samples, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - IQR_MULTIPLIER * iqr
    upper_bound = q3 + IQR_MULTIPLIER * iqr
    keep = (samples >= lower_bound) & (samples <= upper_bound)
    return samples[keep], int(np.count_nonzero(~keep))


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


def save_numbered_zone_image(
    frame_dir: Path,
    zone_edges: np.ndarray,
    output_dir: Path,
) -> Path:
    tif_files = sorted(
        path for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not tif_files:
        raise FileNotFoundError(f"Nessun frame TIFF trovato in {frame_dir}")

    frame = io.imread(str(tif_files[0]))
    if frame.ndim < 2:
        raise ValueError(f"Il frame {tif_files[0]} non ha dimensioni immagine valide.")

    frame_height, frame_width = frame.shape[:2]
    fig, ax = plt.subplots(figsize=(12, max(2.5, 12 * frame_height / frame_width)))
    ax.imshow(frame, cmap="gray" if frame.ndim == 2 else None)
    for zone_index, left_edge in enumerate(zone_edges[:-1]):
        ax.axvline(left_edge, color="cyan", linewidth=1)
        center_x = (left_edge + zone_edges[zone_index + 1]) / 2
        ax.text(
            center_x,
            frame_height * 0.07,
            str(zone_index + 1),
            color="white",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 3},
        )
    ax.axvline(zone_edges[-1], color="cyan", linewidth=1)
    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)
    ax.axis("off")
    fig.tight_layout(pad=0)

    image_path = output_dir / "frame_zones_numbered.png"
    fig.savefig(image_path, dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return image_path


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
                    **{amplitude: float(row[amplitude]) for amplitude in AMPLITUDES},
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
        lambda: {
            "n_frames": 0,
            **{field: [] for field in FILTERED_FIELDS},
        }
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
            for field in FILTERED_FIELDS:
                bucket[field].append(row[field])

    if csv_count == 0:
        raise FileNotFoundError(
            f"Nessun CSV fourier_result.csv o fourier_results.csv trovato "
            f"nelle sottocartelle di {results_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    squared_amplitude_plots_dir = output_dir / SQUARED_AMPLITUDE_PLOTS_DIR.name
    squared_amplitude_plots_dir.mkdir(parents=True, exist_ok=True)
    signed_amplitude_plots_dir = output_dir / SIGNED_AMPLITUDE_PLOTS_DIR.name
    signed_amplitude_plots_dir.mkdir(parents=True, exist_ok=True)
    shared_scale_plots_dir = output_dir / SHARED_SCALE_PLOTS_DIR.name
    shared_signed_plots_dir = shared_scale_plots_dir / SHARED_SIGNED_PLOTS_DIR.name
    shared_squared_plots_dir = shared_scale_plots_dir / SHARED_SQUARED_PLOTS_DIR.name
    shared_signed_plots_dir.mkdir(parents=True, exist_ok=True)
    shared_squared_plots_dir.mkdir(parents=True, exist_ok=True)
    numbered_zone_image = save_numbered_zone_image(
        frame_dir,
        zone_edges,
        output_dir,
    )
    per_droplet_rows = []
    total_outliers_removed = {field: 0 for field in FILTERED_FIELDS}
    zone_aggregates = {
        zone_index: {
            "n_droplets": 0,
            "n_frames": 0,
            **{metric: 0.0 for metric in METRICS},
            **{amplitude: 0.0 for amplitude in SQUARED_AMPLITUDES},
            **{f"{field}_n_frames_used": 0 for field in FILTERED_FIELDS},
            **{f"{field}_outliers_removed": 0 for field in FILTERED_FIELDS},
            **{name: 0.0 for name in SIGNED_AMPLITUDES},
            **{f"{name}_n_droplets": 0 for name in SIGNED_AMPLITUDES},
        }
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
        filtered_values = {}
        for field in FILTERED_FIELDS:
            samples, outliers_removed = filter_iqr_outliers(values[field])
            if samples.size == 0:
                raise ValueError(
                    f"Il filtro IQR ha rimosso tutti i valori di {field} "
                    f"per la goccia {droplet_name}, fascia {zone_index + 1}."
                )
            filtered_values[field] = samples
            row[f"{field}_n_frames_used"] = int(samples.size)
            row[f"{field}_outliers_removed"] = outliers_removed
            total_outliers_removed[field] += outliers_removed

        row.update(
            {
                f"{metric}_mean": float(np.mean(filtered_values[metric]))
                for metric in METRICS
            }
        )
        for amplitude in AMPLITUDES:
            row[f"{amplitude}_squared_mean"] = float(
                np.mean(filtered_values[amplitude] ** 2)
            )
        for signed_amplitude in SIGNED_AMPLITUDES:
            amplitude, sign = signed_amplitude.rsplit("_", 1)
            samples = filtered_values[amplitude]
            signed_samples = samples[samples > 0] if sign == "positive" else samples[samples < 0]
            frame_count = int(signed_samples.size)
            row[f"{signed_amplitude}_mean"] = (
                float(np.mean(signed_samples))
                if frame_count
                else None
            )
            row[f"{signed_amplitude}_n_frames"] = frame_count
        per_droplet_rows.append(row)

        zone_bucket = zone_aggregates[zone_index]
        zone_bucket["n_droplets"] += 1
        zone_bucket["n_frames"] += values["n_frames"]
        for metric in (*METRICS, *SQUARED_AMPLITUDES):
            zone_bucket[metric] += row[f"{metric}_mean"]
        for field in FILTERED_FIELDS:
            zone_bucket[f"{field}_n_frames_used"] += row[f"{field}_n_frames_used"]
            zone_bucket[f"{field}_outliers_removed"] += row[f"{field}_outliers_removed"]
        for signed_amplitude in SIGNED_AMPLITUDES:
            droplet_mean = row[f"{signed_amplitude}_mean"]
            if droplet_mean is not None:
                zone_bucket[signed_amplitude] += droplet_mean
                zone_bucket[f"{signed_amplitude}_n_droplets"] += 1

    aggregated_rows = []
    for zone_index, values in zone_aggregates.items():
        row = {
            "zone": zone_index + 1,
            "x_min_px": zone_edges[zone_index],
            "x_max_px": zone_edges[zone_index + 1],
            "n_droplets": values["n_droplets"],
            "n_frames_total": values["n_frames"],
            **{
                f"{field}_{suffix}": values[f"{field}_{suffix}"]
                for field in FILTERED_FIELDS
                for suffix in ("n_frames_used", "outliers_removed")
            },
        }
        row.update(
            {
                f"{metric}_mean": (
                    values[metric] / values["n_droplets"]
                    if values["n_droplets"]
                    else None
                )
                for metric in (*METRICS, *SQUARED_AMPLITUDES)
            }
        )
        for signed_amplitude in SIGNED_AMPLITUDES:
            n_droplets = values[f"{signed_amplitude}_n_droplets"]
            row[f"{signed_amplitude}_mean"] = (
                values[signed_amplitude] / n_droplets if n_droplets else None
            )
            row[f"{signed_amplitude}_n_droplets"] = n_droplets
        aggregated_rows.append(row)

    histogram_heights = [
        row[f"{amplitude}_squared_mean"]
        for row in aggregated_rows
        for amplitude in AMPLITUDES
        if row[f"{amplitude}_squared_mean"] is not None
    ] + [
        abs(row[f"{amplitude}_mean"])
        for row in aggregated_rows
        for amplitude in SIGNED_AMPLITUDES
        if row[f"{amplitude}_mean"] is not None
    ]
    maximum_histogram_height = max(histogram_heights, default=0.0)
    shared_y_max = maximum_histogram_height * 1.05 if maximum_histogram_height else 1.0

    summary_csv = output_dir / "zone_averages_across_droplets.csv"
    summary_columns = [
        "zone",
        "x_min_px",
        "x_max_px",
        "n_droplets",
        "n_frames_total",
        "a2_mean",
        "a3_mean",
        "a2_squared_mean",
        "a3_squared_mean",
        "a4_squared_mean",
        "strain_mean",
        *[
            f"{field}_{suffix}"
            for field in FILTERED_FIELDS
            for suffix in ("n_frames_used", "outliers_removed")
        ],
        *[f"{name}_mean" for name in SIGNED_AMPLITUDES],
        *[f"{name}_n_droplets" for name in SIGNED_AMPLITUDES],
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
        "a2_squared_mean",
        "a3_squared_mean",
        "a4_squared_mean",
        "strain_mean",
        *[
            f"{field}_{suffix}"
            for field in FILTERED_FIELDS
            for suffix in ("n_frames_used", "outliers_removed")
        ],
        *[f"{name}_mean" for name in SIGNED_AMPLITUDES],
        *[f"{name}_n_frames" for name in SIGNED_AMPLITUDES],
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

    for row in aggregated_rows:
        zone = row["zone"]
        fig, ax = plt.subplots(figsize=(7, 5))
        if row["n_droplets"] > 0:
            values = [row[f"{amplitude}_squared_mean"] for amplitude in AMPLITUDES]
            ax.bar(
                [rf"${amplitude[0]}_{{{amplitude[1:]}}}^2$" for amplitude in AMPLITUDES],
                values,
                width=0.6,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Nessun dato in questa fascia",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel("Ampiezza")
        ax.set_ylabel("Media dell'ampiezza al quadrato")
        ax.set_title(f"Ampiezze quadratiche medie - fascia {zone}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        squared_plot_path = (
            squared_amplitude_plots_dir / f"zone_{zone:02d}_squared_amplitudes.png"
        )
        fig.savefig(squared_plot_path, dpi=250, bbox_inches="tight")
        ax.set_ylim(0, shared_y_max)
        shared_squared_plot_path = (
            shared_squared_plots_dir / f"zone_{zone:02d}_squared_amplitudes.png"
        )
        fig.savefig(shared_squared_plot_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

    for row in aggregated_rows:
        zone = row["zone"]
        fig, ax = plt.subplots(figsize=(9, 5))
        signed_values = [row[f"{name}_mean"] for name in SIGNED_AMPLITUDES]
        bar_labels = [
            rf"${name[:2]}$ {'+' if name.endswith('positive') else '-'}"
            for name in SIGNED_AMPLITUDES
        ]
        plot_indices = [
            index for index, value in enumerate(signed_values) if value is not None
        ]
        if plot_indices:
            ax.bar(
                [bar_labels[index] for index in plot_indices],
                [abs(signed_values[index]) for index in plot_indices],
                width=0.65,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Nessun dato con segno in questa fascia",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlabel("Coefficiente Fourier e segno")
        ax.set_ylabel("Modulo del valore medio del coefficiente")
        ax.set_title(f"Moduli delle medie dei coefficienti - fascia {zone}")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        signed_plot_path = (
            signed_amplitude_plots_dir / f"zone_{zone:02d}_signed_amplitudes.png"
        )
        fig.savefig(signed_plot_path, dpi=250, bbox_inches="tight")
        ax.set_ylim(0, shared_y_max)
        shared_signed_plot_path = (
            shared_signed_plots_dir / f"zone_{zone:02d}_signed_amplitudes.png"
        )
        fig.savefig(shared_signed_plot_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

    legacy_plot_path = output_dir / "squared_amplitudes_by_zone.png"
    if legacy_plot_path.exists():
        legacy_plot_path.unlink()
    for old_plot in output_dir.glob("zone_*_squared_amplitudes.png"):
        old_plot.unlink()
    for old_plot in output_dir.glob("zone_*_signed_amplitudes.png"):
        old_plot.unlink()
    for old_plot in shared_scale_plots_dir.glob("zone_*.png"):
        old_plot.unlink()

    print(
        f"Larghezza frame: {frame_width} px; fasce: {ZONE_COUNT} "
        f"da {frame_width / ZONE_COUNT:.1f} px."
    )
    print(f"Medie per fascia tra gocce salvate in: {summary_csv.resolve()}")
    print(f"Medie per singola goccia e fascia salvate in: {per_droplet_csv.resolve()}")
    print(
        f"Filtro outlier: IQR {IQR_MULTIPLIER}x, applicato separatamente per "
        f"goccia e fascia; campioni con meno di {IQR_MINIMUM_SAMPLES} valori non filtrati."
    )
    print(f"Outlier rimossi per variabile: {total_outliers_removed}")
    print(f"Immagine del frame suddiviso in fasce salvata in: {numbered_zone_image.resolve()}")
    print(f"Grafici a2, a3 e strain in funzione della fascia salvati in: {output_dir.resolve()}")
    print(
        f"Grafici delle ampiezze al quadrato per ciascuna fascia salvati in: "
        f"{squared_amplitude_plots_dir.resolve()}"
    )
    print(
        f"Grafici dei coefficienti positivi e negativi per ciascuna fascia salvati in: "
        f"{signed_amplitude_plots_dir.resolve()}"
    )
    print(
        f"Copie di tutti gli istogrammi con scala y comune (0-{shared_y_max:.6g}) "
        f"salvate nelle sottocartelle {shared_squared_plots_dir.resolve()} e "
        f"{shared_signed_plots_dir.resolve()}"
    )
    return summary_csv


if __name__ == "__main__":
    aggregate_zone_data()
