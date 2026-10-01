# ==========================================
# 1. FUNZIONE DI FITTING 
# (Seno+coseno al 1° ordine e Coseni agli ordini successivi)
# ==========================================

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

def fourier_cos(theta, a0, a1, b1, a2, a3, a4):
    return (a0 
            + a1 * np.cos(theta)   + b1 * np.sin(theta) 
            + a2 * np.cos(2*theta) 
            + a3 * np.cos(3*theta) 
            + a4 * np.cos(4*theta))


# ==========================================
# 2. FUNZIONE DI ANALISI 
"""
    Esegue l'analisi morfologica di un frame tramite Active Contour e fitting di Fourier.

    1. Applica un filtro Gaussiano.
    2. Trova il contorno usando Active Contour.
    3. Calcola il centro di massa del contorno trovato.
    4. Converte le coordinate in polari relative al centro di massa.
    5. Esegue il fitting del raggio in funzione dell'angolo con fourier_cos.
    6. Calcola i moduli delle deformazioni.
    
    Parametri:
    - img: Immagine originale del frame.
    - init_contour: Contorno iniziale per l'algoritmo (array Nx2).
    - sigma, alpha, beta, w_edge, w_line: Parametri per preprocessing e active contour.
    
    Ritorna:
    - a0: Raggio medio (ordine 0).
    - c1: Modulo armonico del 1° ordine (traslazione residua/asimmetria).
    - c2: Modulo armonico del 2° ordine (deformazione ellittica).
    - c3: Modulo armonico del 3° ordine (deformazione triangolare).
    - c4: Modulo armonico del 4° ordine (deformazione quadrata).
    - contour: Le coordinate (N, 2) del contorno trovato, utile per plot o verifica.
    """
# ==========================================

def analyze_frame_fourier(img, init_contour, sigma=2, alpha=1.5, beta=10.0, w_edge=5.0, w_line=0.0, plot_profile=False, frame_number=None):
    # 1. Pre-processing
    if img.ndim == 3:
        img_gray = rgb2gray(img)
    else:
        img_gray = img
        
    img_blurred = gaussian(img_gray, sigma=sigma)
    
    # 2. Active Contour
    contour = active_contour(img_blurred, init_contour, alpha=alpha, beta=beta, 
                             w_edge=w_edge, w_line=w_line, boundary_condition='periodic')
    Y_contour = contour[:, 0]
    X_contour = contour[:, 1]
    
    # 3. Centro di massa
    x_center_prov = np.mean(X_contour)
    y_center_prov = np.mean(Y_contour)
    
    X_centered = X_contour - x_center_prov 
    Y_centered = Y_contour - y_center_prov
    
    # 4. Coordinate Polari
    theta_contour = np.arctan2(Y_centered, X_centered)
    radius_contour = np.sqrt(X_centered**2 + Y_centered**2)
    
    # Ordinamento per angolo
    #sort_indices = np.argsort(theta_contour)
    #theta_sorted = theta_contour[sort_indices]
    #radius_sorted = radius_contour[sort_indices]
    
    # 5. Fit di Fourier (1st fit)
    initial_par = [np.mean(radius_contour)] + [0.0] * 5
    fit_parameters, _ = curve_fit(fourier_cos, theta_contour, radius_contour, p0=initial_par)
    
    # 6. centro vero
    x_true = x_center_prov + fit_parameters[1]
    y_true = y_center_prov + fit_parameters[2]
    fit_parameters[1] = 0.0  # azzero a1
    fit_parameters[2] = 0.0  # azzero b1

    # 6. Fit di Fourier (2nd fit)
    theta_fit = np.linspace(-np.pi, np.pi, 400)            #disegno una curva con i parametri del fit
    radius_fit = fourier_cos(theta_fit, *fit_parameters) 
    x_fit = x_true + radius_fit * np.cos(theta_fit)
    y_fit = y_true + radius_fit * np.sin(theta_fit)

    # 7. Ricalcolo coordinate polari con centro vero
    #r_corr = np.sqrt((X_contour - x_true)**2 + (Y_contour - y_true)**2)
    #theta_corr = np.arctan2(Y_contour - y_true, X_contour - x_true)
    
    # Riordinamento (necessario se l'ordine del contorno è cambiato a causa della traslazione)
    #sort_indices = np.argsort(theta_corr)
    #theta_sorted = theta_corr[sort_indices]
    #r_sorted = radius_fit[sort_indices]



    
    # 6. Fit di Fourier (3 fit)
    #guess_iniziale = [np.mean(r_sorted)] + [0.0] * 5
    #fit_parameters, _ = curve_fit(fourier_cos, theta_sorted, r_sorted, p0=guess_iniziale)


    # 7. Estrazione dei Parametri
    a0 = fit_parameters[0]
    a1, b1 = fit_parameters[1], fit_parameters[2]
    a2 = fit_parameters[3]
    a3 = fit_parameters[4]
    a4 = fit_parameters[5]
    
    # 8. Moduli Armonici
    c1 = np.sqrt(a1**2 + b1**2)
    c2 = np.abs(a2)
    c3 = np.abs(a3)
    c4 = np.abs(a4)
    
    if plot_profile:
        theta_contour = np.arctan2(Y_contour - y_true, X_contour - x_true) 
        theta_contour = np.mod(theta_contour, 2 * np.pi)
        r_contour = np.sqrt((X_contour - x_true)**2 + (Y_contour - y_true)**2)
        
        theta_fit_mod = np.mod(theta_fit, 2 * np.pi)
        
        sort_contour_indices = np.argsort(theta_contour)
        theta_contour_sorted = theta_contour[sort_contour_indices]
        r_contour_sorted = r_contour[sort_contour_indices]
        
        sort_fit_indices = np.argsort(theta_fit_mod)
        theta_fit_sorted = theta_fit_mod[sort_fit_indices]
        r_fit_sorted = radius_fit[sort_fit_indices]

        # Creiamo una figura con 2 grafici affiancati (ax1 e ax2)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Se abbiamo passato il numero del frame, lo mettiamo come super-titolo globale
        if frame_number is not None:
            fig.suptitle(f"Analisi del Frame: {frame_number}", fontsize=16, fontweight='bold')

        # --- GRAFICO 1: Immagine e Contorni ---
        # (nota: dentro la funzione l'immagine si chiama img_gray, non ROI)
        ax1.imshow(img_gray, cmap=plt.cm.gray)
        ax1.plot(init_contour[:, 1], init_contour[:, 0], '--r', lw=1, label="Starting point")
        ax1.plot(contour[:, 1], contour[:, 0], '-b', lw=2, label="Active contour")
        ax1.set_xticks([])
        ax1.set_yticks([]) # Nasconde i numeri sugli assi
        ax1.legend()
        
        # --- GRAFICO 2: Profilo Radiale ---
        ax2.plot(theta_contour_sorted, r_contour_sorted, color='blue', linestyle='-', marker='.', label='Contorno')
        ax2.plot(theta_fit_sorted, r_fit_sorted, color='red', linestyle='-', marker='.', label='Fit')
        ax2.axhline(a0, color='black', linestyle='--', alpha=0.7, label='a_0')
        ax2.set_xlabel('Angle θ (rad)')
        ax2.set_ylabel('radius r(θ)')
        ax2.set_title('Radial profile r(θ)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    return a0, a2, a3, a4, c1, c2, c3, c4, contour


# ==========================================
# 3. FUNZIONE DI TEMPLATE MATCHING
    """
    Esegue il template matching per trovare la posizione di un oggetto in un frame.
    
    Parametri:
    - img: Immagine corrente (verrà convertita in scala di grigi se a colori).
    - template: Il template da cercare (immagine croppata, in scala di grigi).
    - threshold: Valore minimo di affidabilità per considerare trovato l'oggetto.
    - plot_result: Se True, mostra un plot con il rettangolo trovato e la heatmap.
    
    Ritorna:
    - found (bool): True se l'oggetto è stato trovato con confidenza >= threshold.
    - center (tuple): Coordinate (x_center, y_center) del centro, oppure None.
    - ROI (ndarray): L'immagine ritagliata trovata, oppure None.
    - confidence (float): L'affidabilità (max_val) del matching.
    """
# ==========================================

def track_droplet_template(img, template, threshold=0.60, plot_result=False):
    # 1. Conversione in scala di grigi
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    h, w = template.shape[:2]
    
    # 2. Template Matching
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 3. Verifica della threshold
    if max_val >= threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        ROI = gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
        
        x_center = top_left[0] + (w // 2)
        y_center = top_left[1] + (h // 2)
        
        if plot_result:
            img_plot = gray.copy()
            # Disegna rettangolo e marker. Nota: gray è 1 canale, cv2.rectangle disegnerà in bianco/nero o se necessario un valore di intensità (es. 255)
            cv2.rectangle(img_plot, top_left, bottom_right, 255, 2)
            cv2.drawMarker(img_plot, (x_center, y_center), 0, cv2.MARKER_TILTED_CROSS, 30, 3)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            ax1.imshow(img_plot, cmap='gray')
            ax1.set_title(f"Oggetto trovato con affidabilità: {max_val:.2f}", fontsize=14)
            ax1.axis('off')
            
            heatmap = ax2.imshow(result, cmap='viridis', vmin=0, vmax=1)
            ax2.set_title(f"Probability Heatmap (Peak: {max_val:.2f})", fontsize=14)
            ax2.axis('off')
            
            cbar = fig.colorbar(heatmap, ax=ax2, fraction=0.02, pad=0.04)
            cbar.set_label('Confidence Score (0.0 to 1.0)', fontsize=12)
            plt.tight_layout()
            plt.show()
            
        return True, (x_center, y_center), ROI, max_val
    else:
        if plot_result:
            print(f"Oggetto non trovato nel frame. Affidabilità massima: {max_val:.2f} < {threshold}")
        return False, None, None, max_val
