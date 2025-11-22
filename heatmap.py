import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  

import os
path = "full_ciudad/cloud_full_000098.xyz"


nombre_archivo = os.path.basename(path)  
try:

    frame_id = nombre_archivo.split('_')[-1].replace('.xyz', '')
except:
    frame_id = "unknown"

output_filename = f"heatmaps/heatmap_{frame_id}.pcd"
MAPA_DE_COLOR = "jet" 




try:
    df = pd.read_csv(path, sep=' ', header=None, names=['x', 'y', 'z', 'ref', 'nir'])
    
    points = df[['x', 'y', 'z']].values.astype(np.float32)
    reflectivity = df['ref'].values.astype(np.float32)
    
    print(f"Datos cargados: {len(points)} puntos.")

except Exception as e:
    print(f"Error cargando archivo: {e}")
    exit()



min_val = reflectivity.min()   
max_val = reflectivity.max()  

print(f"Rango de reflectividad usado: {min_val} a {max_val}")


ref_norm = (reflectivity - min_val) / (max_val - min_val)

ref_norm = np.clip(ref_norm, 0, 1)




cmap = plt.get_cmap(MAPA_DE_COLOR)

colors = cmap(ref_norm)[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

print(f"Guardando resultado en {output_filename}...")
o3d.io.write_point_cloud(output_filename, pcd)


