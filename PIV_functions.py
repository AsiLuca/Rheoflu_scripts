
def save_plot_artifacts(fig, filename, metadata):
    """Salva un grafico PNG e un JSON con i relativi metadati in OUTPUT_DIR."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path = OUTPUT_DIR / filename
    metadata_path = image_path.with_suffix(".json")
    dpi = 300
    fig.savefig(image_path, dpi=dpi, bbox_inches="tight")

    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_file": image_path.name,
        "figure_size_inches": [float(size) for size in fig.get_size_inches()],
        "dpi": dpi,
        "analysis_parameters": {
            "input_dir": str(INPUT_DIR),
            "first_frame": FIRST_FRAME,
            "final_frame": FINAL_FRAME,
            "frame_step": FRAME_STEP,
            "window_size": WINDOW_SIZE,
            "overlap": OVERLAP,
            "search_area_size": SEARCH_AREA_SIZE,
            "dt_seconds": DT_SECONDS,
            "s2n_threshold": S2N_THRESHOLD,
            "pixel_size_um": pixel_size_um,
        },
        **metadata,
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, indent=2, ensure_ascii=False)

    print(f"Grafico salvato: {image_path.resolve()}")
    print(f"Metadati salvati: {metadata_path.resolve()}")
    return image_path, metadata_path
PATTERN = "*.tif"

def read_gray_and_crop(path):
  """Legge l'immagine, ritaglia lungo il canale e ruota di 90° CCW."""
  img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
  if img is None:
    raise ValueError(f"Impossibile leggere l'immagine: {path}")
  crop = img[CROP_Y_MIN:CROP_Y_MAX, CROP_X_MIN:CROP_X_MAX]
  return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

def preprocess_for_piv(path):
  """Sottrae il background statico per lasciare solo le particelle in movimento."""
  frame_raw = read_gray_and_crop(path)
  diff = cv2.subtract(frame_raw, BACKGROUND)
  return cv2.GaussianBlur(diff, (3, 3), 0.8)

def get_channel_bounds(x_query):
    """Restituisce y_top(x) e y_bottom(x) per qualsiasi coordinata x."""
    yt = np.interp(x_query, x_t, y_t)
    yb = np.interp(x_query, x_b, y_b)
    return yt, yb

def get_channel_mask(x, y):
    """Restituisce True all'interno del canale, escludendo 3 px da ogni parete."""
    yt, yb = get_channel_bounds(x)
    erosion_px = 2
    return (y >= yt + erosion_px) & (y <= yb - erosion_px)

def analyze_pair(path_a, path_b):
    """Calcola il campo di spostamento PIV mascherato all'interno delle pareti del canale."""
    frame_a = preprocess_for_piv(path_a)
    frame_b = preprocess_for_piv(path_b)


    if frame_a.shape != frame_b.shape:
        raise ValueError(f"Dimensioni disuguali: {frame_a.shape} vs {frame_b.shape}")

    # 1. PIV con OpenPIV
    u, v, s2n = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        search_area_size=SEARCH_AREA_SIZE,
        dt=DT_SECONDS,
        sig2noise_method="peak2peak",
    )

    # 2. Coordinate della griglia
    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=SEARCH_AREA_SIZE,
        overlap=OVERLAP,
    )

    # 3. Maschera del canale: identifica i punti interni
    in_channel = get_channel_mask(x, y)
    # 4. Validazione: scarta vettori con s2n basso e vettori fuori dal canale
    invalid_s2n = validation.sig2noise_val(s2n, threshold=S2N_THRESHOLD)
    invalid = invalid_s2n | (~in_channel)
    #Filtro su velocità fisicamente impossibili (es. vettori > U_MAX)
    # Se il flusso medio è lungo X positivo, u deve essere positivo e v prossimo a 0:
    invalid_vel = (np.abs(v) > 20) | (np.abs(u) > 50)
    invalid = invalid | invalid_vel
    # 5. Sostituisce outlier interpolando solo tra punti validi
    u_clean, v_clean = filters.replace_outliers(
        u,
        v,
        invalid,
        method="localmean",
        max_iter=3,
        kernel_size=3,
    )

    # 6. Maschera i punti fuori dal canale impostandoli a NaN
    u_final = np.where(in_channel, u_clean, np.nan)
    v_final = np.where(in_channel, v_clean, np.nan)

    return x, y, u_final, v_final, s2n, invalid, in_channel



def get_frame_pairs():
    """Crea tutte le coppie con distanza temporale FRAME_STEP."""
    if FRAME_STEP <= 0:
        raise ValueError("FRAME_STEP deve essere un intero positivo")

    frames_paths = sorted(INPUT_DIR.glob(PATTERN))
    frames_paths = frames_paths[FIRST_FRAME:FINAL_FRAME]
    pair_count = max(0, len(frames_paths) - FRAME_STEP)
    frame_pairs = [
        (frames_paths[i], frames_paths[i + FRAME_STEP])
        for i in range(pair_count)
    ]
    return frames_paths, frame_pairs


def plot_single_piv(index_to_plot=0):
    """Visualizza il campo vettoriale PIV per una specifica coppia di frame."""
    _, frame_pairs = get_frame_pairs()

    if not frame_pairs:
        print("Nessuna coppia di frame disponibile per l'analisi PIV!")
        return None, None, None

    if not 0 <= index_to_plot < len(frame_pairs):
        print(f"Indice {index_to_plot} non valido! Disponibili coppie da 0 a {len(frame_pairs)-1}.")
        return

    csv_file = OUTPUT_DIR / f"piv_{index_to_plot:04d}.csv"
    if not csv_file.exists():
        print(f"File {csv_file.name} non trovato in {OUTPUT_DIR}.Esegui prima la cella successiva main() per calcolare la PIV!")
        return

    frame_a_index = FIRST_FRAME + index_to_plot
    frame_b_index = frame_a_index + FRAME_STEP
    print(f"Generazione grafico PIV per la coppia {index_to_plot+1} ({frame_a_index} -> {frame_b_index})...")
    first_frame_path = frame_pairs[index_to_plot][0]
    img = read_gray_and_crop(first_frame_path)

    df = pd.read_csv(csv_file)

    # Filtro robusto per i punti nel canale
    if 'in_channel' in df.columns:
        df_channel = df[df['in_channel'] == True]
    else:
        in_c = get_channel_mask(df['x_px'].values, df['y_px'].values)
        df_channel = df[in_c]

    # Esclude eventuali NaN residui
    df_channel = df_channel[df_channel['u_px_s'].notna()]

    x_columns = np.sort(df_channel["x_px"].unique())
    df_channel = df_channel[df_channel["x_px"].isin(x_columns[::2])]

    x = df_channel['x_px'].values
    y = df_channel['y_px'].values
    u = df_channel['u_px_s'].values
    v = df_channel['v_px_s'].values
    magnitude = np.sqrt(u**2 + v**2) * pixel_size_um

    fig, ax = plt.subplots(figsize=(15, 5))

    # Sfondo: immagine ritagliata e orientata
    ax.imshow(img, cmap="gray", extent=image_extent_um)

    # Pareti del canale e mezzeria
    ax.plot(x_plot * pixel_size_um, yt_plot * pixel_size_um, "-c", lw=2, alpha=0.85, label="Upper wall")
    ax.plot(x_plot * pixel_size_um, yb_plot * pixel_size_um, "-m", lw=2, alpha=0.85, label="Lower wall")
    ax.plot(x_plot * pixel_size_um, y_center_plot * pixel_size_um, "--w", lw=1.2, alpha=0.6, label="Centerline")

    # Frecce PIV all'interno del canale
    Q = ax.quiver(
        x * pixel_size_um, y * pixel_size_um,
        u * pixel_size_um, v * pixel_size_um, magnitude,
        angles="xy",
        scale=400 * pixel_size_um,
        width=0.0025,  # Spessore dell'asta della freccia
        headwidth=2.5,  # Larghezza della punta
        headlength=2.5,  # Lunghezza della punta
        cmap="plasma",
        pivot="tail",
        alpha=0.9,
    )

def main():
    frames, frame_pairs = get_frame_pairs()

    if not frame_pairs:
        raise RuntimeError(
            f"Servono almeno {FRAME_STEP + 1} immagini in {INPUT_DIR} "
            f"con estensione {PATTERN} per creare una coppia PIV"
        )

    # Elimina i risultati precedenti solo dopo aver verificato che l'analisi sia possibile.
    for f in OUTPUT_DIR.glob("piv_*.csv"):
        f.unlink()

    print(f"Trovati {len(frames)} frame ({len(frame_pairs)} coppie PIV)...")

    for index, (path_a, path_b) in enumerate(frame_pairs):
        x, y, u, v, s2n, invalid, in_channel = analyze_pair(path_a, path_b)

        result = pd.DataFrame({
            "x_px": x.ravel(),
            "y_px": y.ravel(),
            "u_px_s": u.ravel(),
            "v_px_s": v.ravel(),
            "signal_to_noise": s2n.ravel(),
            "invalid": invalid.ravel(),
            "in_channel": in_channel.ravel(),
        })

        output_path = OUTPUT_DIR / f"piv_{index:04d}.csv"
        result.to_csv(output_path, index=False)

        valid_in = np.sum((~np.isnan(u.ravel())) & in_channel.ravel())
        frame_a_index = FIRST_FRAME + index
        frame_b_index = frame_a_index + FRAME_STEP
        print(f"Salvata coppia {index+1}/{len(frame_pairs)} ({frame_a_index} -> {frame_b_index}): {valid_in} vettori validi nel canale -> {output_path.name}")

    print("Analisi PIV completata con successo!")

    # Mostra circa 7 grafici distribuiti tra le coppie calcolate.
    plot_interval = max(10, (len(frame_pairs) + 6) // 7)
    for plot_index in range(0, len(frame_pairs), plot_interval):
        plot_single_piv(plot_index)

def plot_piv_results():
    print("Calculating the time-averaged velocity fields...")
    frames_paths, frame_pairs = get_frame_pairs()

    if not frame_pairs:
        print("Nessuna coppia di frame disponibile per l'analisi PIV!")
        return None, None, None

    csv_files = sorted(OUTPUT_DIR.glob("piv_*.csv"))
    if not csv_files:
        print(f"Nessun file CSV trovato in {OUTPUT_DIR}. Esegui prima la cella main()!")
        return None, None, None

    mean_img = read_gray_and_crop(frames_paths[0]).astype(np.float64)
    for frame_path in frames_paths[1:]:
        mean_img += read_gray_and_crop(frame_path)
    mean_img /= len(frames_paths)

    df_list = [pd.read_csv(f) for f in csv_files]
    df_all = pd.concat(df_list)

    # Filtra solo i punti nel canale
    if "in_channel" in df_all.columns:
        df_channel = df_all[df_all["in_channel"] == True]
    else:
        in_c = get_channel_mask(
            df_all["x_px"].values,
            df_all["y_px"].values,
        )
        df_channel = df_all[in_c]

    df_channel = df_channel[df_channel["u_px_s"].notna()]
    df_mean = (
        df_channel.groupby(["x_px", "y_px"])[["u_px_s", "v_px_s"]]
        .mean()
        .reset_index()
    )

    df_profile = df_mean.copy()
    df_profile["y_center_px"] = np.interp(
        df_profile["x_px"],
        x_plot,
        y_center_plot,
    )
    df_profile = df_profile[
        np.abs(df_profile["y_px"] - df_profile["y_center_px"]) <= 6.5
    ]

    velocity_profile = (
        df_profile.groupby("x_px")[["u_px_s", "v_px_s"]]
        .mean()
        .sort_index()
    )
    x_profile = velocity_profile.index.to_numpy()
    u_profile = velocity_profile["u_px_s"].to_numpy()

    smoothing_window = min(
        11,
        u_profile.size if u_profile.size % 2 else u_profile.size - 1,
    )
    if smoothing_window >= 3:
        u_profile_smooth = savgol_filter(
            u_profile,
            window_length=smoothing_window,
            polyorder=min(3, smoothing_window - 1),
        )

    x = df_mean["x_px"].values
    y = df_mean["y_px"].values
    u = df_mean["u_px_s"].values
    v = df_mean["v_px_s"].values
    magnitude = np.sqrt(u**2 + v**2) * pixel_size_um

    # Dirada le colonne X solo per il secondo grafico medio
    x_columns = np.sort(df_mean["x_px"].unique())
    df_mean_sparse = df_mean[df_mean["x_px"].isin(x_columns[::2])]
    x_sparse = df_mean_sparse["x_px"].values
    y_sparse = df_mean_sparse["y_px"].values
    u_sparse = df_mean_sparse["u_px_s"].values
    v_sparse = df_mean_sparse["v_px_s"].values
    magnitude_sparse = np.hypot(u_sparse, v_sparse) * pixel_size_um

    angle_rad = np.arctan2(v, u)
    angle_deg = (np.degrees(angle_rad) + 360) % 360

    print("Generating time-averaged plots...")
    fig, ax1 = plt.subplots(figsize=(15, 5))

    # 1. Campo di velocità medio
    ax1.imshow(mean_img, cmap="gray", extent=image_extent_um)

    arrow_length_um = 45
    speed = np.hypot(u, v)
    u_arrow = (
        np.divide(u, speed, out=np.zeros_like(u), where=speed != 0)
        * arrow_length_um
    )
    v_arrow = (
        np.divide(v, speed, out=np.zeros_like(v), where=speed != 0)
        * arrow_length_um
    )

    Q1 = ax1.quiver(
        x * pixel_size_um,
        y * pixel_size_um,
        u_arrow,
        v_arrow,
        magnitude,
        angles="xy",
        scale_units="xy",
        scale=1,
        cmap="plasma",
        width=0.0017,
        headwidth=3.85,
        headlength=3.74,
        pivot="tail",
        alpha=0.8,
    )
    cbar1 = plt.colorbar(Q1, ax=ax1, shrink=0.7, aspect=20, pad=0.03)
    cbar1.set_label("Mean Velocity Magnitude (µm/s)", fontsize=11)
    ax1.set_title("PIV Time-Averaged Velocity Field in the Channel", fontsize=13)
    ax1.set_ylabel("Y [µm]", fontsize=11)

    plt.tight_layout()
    save_plot_artifacts(
        fig,
        "mean_piv_velocity_field.png",
        {
            "plot_type": "time_averaged_piv_velocity_field",
            "title": "PIV Time-Averaged Velocity Field in the Channel",
            "coordinate_units": "µm",
            "velocity_units": "µm/s",
            "csv_files_averaged": len(csv_files),
            "mean_vector_count": int(len(df_mean)),
        },
    )
    plt.show()

    return x_profile, u_profile, mean_img
