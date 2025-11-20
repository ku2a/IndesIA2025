import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os  

path = "full_ciudad/cloud_full_000098.xyz"


CARPETA_SALIDA = "señales"

# FACTOR IQR
FACTOR_IQR = 5
DBSCAN_EPS = 0.4
DBSCAN_MIN_POINTS = 10


nombre_archivo = os.path.basename(path)  
try:

    frame_id = nombre_archivo.split('_')[-1].replace('.xyz', '')
except:
    frame_id = "unknown"


os.makedirs(CARPETA_SALIDA, exist_ok=True)


output_filename = os.path.join(CARPETA_SALIDA, f"señal_{frame_id}.pcd")



try:
    df = pd.read_csv(path, sep=' ', header=None, names=['x', 'y', 'z', 'ref', 'nir'])
    points = df[['x', 'y', 'z']].values.astype(np.float32)
    reflectivity = df['ref'].values.astype(np.float32)
    
    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(points)
    
    # Pintar fondo gris
    pcd_final.paint_uniform_color([0.5, 0.5, 0.5]) 
    colors_full = np.asarray(pcd_final.colors)

except Exception as e:
    print(f"Error cargando: {e}")
    exit()

#Rangos intercuartílicos
Q1 = np.percentile(reflectivity, 25)
Q3 = np.percentile(reflectivity, 75)
IQR = Q3 - Q1
umbral_iqr = Q3 + (FACTOR_IQR * IQR)

MIN_SEGURIDAD = 150.0
umbral_final = max(umbral_iqr, MIN_SEGURIDAD)

indices_high_ref_global = np.where(reflectivity > umbral_final)[0]

if len(indices_high_ref_global) > 0:
    # Sub-nube temporal para cálculos
    pcd_high = pcd_final.select_by_index(indices_high_ref_global)
    
    # Limpieza estadística
    pcd_clean, indices_clean_local = pcd_high.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    
    # Mapeo de índices
    indices_clean_global = indices_high_ref_global[indices_clean_local]

    if len(pcd_clean.points) > 0:
        # Clustering
        labels = np.array(pcd_clean.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False))
        max_label = labels.max()
        
        if max_label >= 0:
            print(f"   OBJETOS DETECTADOS: {max_label + 1}")
            
            cmap = plt.get_cmap("hsv")
            count_signals = 0
            
            for i in range(max_label + 1):
                cluster_mask_local = (labels == i)
                num_pts = np.sum(cluster_mask_local)
                

                if 20 <= num_pts <= 2000:
                    # Color único para este cluster
                    color_cluster = cmap(i / (max_label + 1))[:3]
                    
                    # Índices globales para pintar sobre el fondo gris
                    indices_cluster_global = indices_clean_global[cluster_mask_local]
                    colors_full[indices_cluster_global] = color_cluster
                    
                    count_signals += 1
            
            print(f"Señales coloreadas: {count_signals}")
            
            
            print(f"Guardando en {output_filename}...")
            o3d.io.write_point_cloud(output_filename, pcd_final)
            
            # Visualizar
            o3d.visualization.draw_geometries([pcd_final], window_name=f"Señal {frame_id}")
            
        else:
             print("No se formaron clusters válidos.")
    else:
        print("Todo era ruido.")
else:
    print("No hay puntos brillantes.")