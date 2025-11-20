import numpy as np
import cv2
import json
from ouster.sdk import client
import os
import glob
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
directory = "./1756674415/"
trip = "1756674415"
PATH_RES = "1756674415"
metadata_file = "1756674415/metadata.json"

# --- Helper Function ---
def normalize_img(img):
    img = np.nan_to_num(img, nan=0.0)
    img = img.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)

# --- 1. Load Metadata & Create LUT ---
print(f"Loading metadata from: {metadata_file}")
with open(metadata_file, "r") as f:
    raw_json = f.read()
info = client.SensorInfo(raw_json)
xyz_lut = client.XYZLut(info) 

# --- 2. Get Frames ---
range_files = sorted(glob.glob(os.path.join(PATH_RES, "range_*.npy")))
# Asumimos que existen archivos coincidentes para reflec y nir
print(f"Found {len(range_files)} range files.")

i = 0
while 0 <= i < len(range_files):
    rfile = range_files[i]  
    frame_id = os.path.basename(rfile).split("_")[1].split(".")[0]
    f_ref = os.path.join(PATH_RES, f"reflectivity_{frame_id}.npy")
    f_nir = os.path.join(PATH_RES, f"nir_{frame_id}.npy")

    if not (os.path.exists(f_ref) and os.path.exists(f_nir)):
        i += 1
        continue

    # Load Data
    range_px = np.load(rfile)
    refl_px  = np.load(f_ref)
    nir_px   = np.load(f_nir)

    # --- 3. Generate Point Cloud (XYZ) ---
    # xyz_points shape is (H, W, 3)
    xyz_points = xyz_lut(range_px) 

    # --- 4. Visualization (2D) ---
    # Destagger for viewing ONLY (Keep raw data for saving)
    range_img = client.destagger(info, range_px)
    refl_img  = client.destagger(info, refl_px)
    nir_img   = client.destagger(info, nir_px)

    # Visualize
    range_vis = normalize_img(range_img)
    refl_vis  = normalize_img(refl_img)
    nir_vis   = normalize_img(nir_img)

    sep = np.ones((50, range_vis.shape[1]), dtype=np.uint8) * 255
    winname = "Ouster Viewer (Press 's' to save cloud)"
    combined = np.vstack([range_vis, sep, refl_vis, sep, nir_vis])
    
    cv2.imshow(winname, combined)
    print(f"Frame {frame_id} | Press 's' to save, 'q' to quit, arrows to move.")

    # --- 5. Controls ---
    key = cv2.waitKey(0) & 0xFF
    
    if key == ord('q') or key == 27:   # Quit
        break
    
    elif key == ord('s'): # SAVE CON TODOS LOS ATRIBUTOS
        filename = f"cloud_full_{frame_id}.xyz"
        print(f"Procesando guardado para {filename}...")

        # 1. Aplanar los arrays para tener listas de puntos (N puntos)
        # xyz pasa de (H,W,3) a (N, 3)
        xyz_flat = xyz_points.reshape(-1, 3)
        
        # refl y nir pasan de (H,W) a (N, 1). Importante el '1' para poder concatenar
        refl_flat = refl_px.reshape(-1, 1)
        nir_flat  = nir_px.reshape(-1, 1)

        # 2. Crear máscara de puntos válidos (eliminar los que están en el origen 0,0,0)
        norms = np.linalg.norm(xyz_flat, axis=1)
        mask = norms > 0.1  # Esto devuelve True/False para cada punto

        # 3. Aplicar filtro a TODOS los datos para mantener la sincronización
        xyz_valid  = xyz_flat[mask]
        refl_valid = refl_flat[mask]
        nir_valid  = nir_flat[mask]

        # 4. Juntar todo horizontalmente (Horizontal Stack)
        # El resultado será una matriz de (N_validos, 5) -> [X, Y, Z, Refl, NIR]
        data_to_save = np.hstack((xyz_valid, refl_valid, nir_valid))
        filename = f"full_ciudad/{filename}"
        # 5. Guardar
        # fmt="%.4f" guardará todo con 4 decimales. 
        # Es suficiente para XYZ y funciona bien para Refl/NIR aunque sean enteros.
        np.savetxt(filename, data_to_save, fmt="%.4f")
        
        print(f"Guardado exitosamente: {filename} con shape {data_to_save.shape}")
        print("Formato de columnas: X Y Z Reflectivity NIR")

    elif key == ord('d') or key == 83 or key == 54: # Next
        if i < len(range_files) - 1:
            i += 1
        else:
            print("Ultimo frame.")

    elif key == ord('a') or key == 81 or key == 52: # Previous
        if i > 0:
            i -= 1
        else:
            print("Primer frame.")

cv2.destroyAllWindows()