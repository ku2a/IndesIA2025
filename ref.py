import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # <--- Necesario para los mapas de color (Heatmaps)

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
path = "full_ciudad/cloud_full_000378.xyz"
output_filename = "resultado_heatmap_reflectividad.pcd"
MAPA_DE_COLOR = "jet"  # Opciones: 'jet', 'viridis', 'plasma', 'inferno', 'magma'

# -----------------------------
# 1. Cargar Datos con Pandas
# -----------------------------
print(f"Cargando {path}...")

try:
    # Leemos las 5 columnas (X, Y, Z, Ref, Nir)
    df = pd.read_csv(path, sep=' ', header=None, names=['x', 'y', 'z', 'ref', 'nir'])
    
    points = df[['x', 'y', 'z']].values.astype(np.float32)
    reflectivity = df['nir'].values.astype(np.float32)
    
    print(f"Datos cargados: {len(points)} puntos.")

except Exception as e:
    print(f"Error cargando archivo: {e}")
    exit()

# -----------------------------
# 2. Normalizar la Reflectividad (CRÍTICO)
# -----------------------------
# Los mapas de color necesitan valores entre 0.0 y 1.0.
# La reflectividad cruda puede tener picos muy altos que "apagan" el resto de colores.
# Usamos percentiles para ignorar outliers extremos.

min_val = np.percentile(reflectivity, 1)   # El valor mínimo (ignorando el 1% más bajo)
max_val = np.percentile(reflectivity, 99)  # El valor máximo (ignorando el 1% más alto)

print(f"Rango de reflectividad usado: {min_val} a {max_val}")

# Normalización Min-Max
ref_norm = (reflectivity - min_val) / (max_val - min_val)

# Aseguramos que nada se salga de 0 a 1 (clip)
ref_norm = np.clip(ref_norm, 0, 1)

# -----------------------------
# 3. Aplicar el Heatmap
# -----------------------------
print(f"Aplicando mapa de color '{MAPA_DE_COLOR}'...")

# Obtenemos el objeto colormap de Matplotlib
cmap = plt.get_cmap(MAPA_DE_COLOR)

# cmap(val) devuelve (R, G, B, Alpha). Nosotros solo queremos (R, G, B).
colors = cmap(ref_norm)[:, :3]

# -----------------------------
# 4. Crear Nube y Guardar
# -----------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

print(f"Guardando resultado en {output_filename}...")
o3d.io.write_point_cloud(output_filename, pcd)

print("¡Guardado! Ábrelo para ver el mapa de calor.")

# Visualización opcional (si tienes entorno gráfico)
# o3d.visualization.draw_geometries([pcd], window_name="Heatmap Reflectividad")