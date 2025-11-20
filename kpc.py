import open3d as o3d
import numpy as np
import pandas as pd
import open3d.ml.torch as ml
import torch
from open3d.ml.torch.pipelines import SemanticSegmentation
import os, requests


filename="kpconv_semantickitti_202009090354utc.pth"
url = "https://storage.googleapis.com/open3d-releases/model-zoo/kpconv_semantickitti_202009090354utc.pth"



if os.path.exists(filename):
    print("Modelo detectado")
else:
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status() 
        
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)


path = "full_ciudad/cloud_full_000851.xyz"
nombre_archivo = os.path.basename(path)  
try:

    frame_id = nombre_archivo.split('_')[-1].replace('.xyz', '')
except:
    frame_id = "unknown"

pesos = "kpconv_semantickitti_202009090354utc.pth" 

print(f"Configurando modelo KPFCNN con pesos: {pesos}")

kit_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

model = ml.models.KPFCNN(
    name="KPFCNN",
    dim_input=3,         
    num_classes=19,      
    ignored_label_inds=[0],
    lbl_values=kit_labels 
)

try:
    checkpoint = torch.load(pesos)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

except Exception as e:
    print(f"Error cargando pesos: {e}")

    exit()


pipeline = SemanticSegmentation(model=model, device="cuda")



print(f"Cargando {path} con Pandas...")

try:
    # Leemos las 5 columnas (ajusta si tu archivo tiene otro formato)
    df = pd.read_csv(path, sep=' ', header=None, names=['x', 'y', 'z', 'ref', 'nir'])
    
    # Extraemos solo XYZ
    points = df[['x', 'y', 'z']].values.astype(np.float32)
    
    # Creamos objeto Open3D (solo para guardar al final)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    print(f"Puntos cargados: {points.shape}")

except Exception as e:
    print(f"Error cargando archivo: {e}")
    exit()


empty_feats = np.zeros((points.shape[0], 0), dtype=np.float32)

data = {
    'point': points,
    'feat': empty_feats, 
    'label': np.zeros(len(points), dtype=np.int32)
}


print("Ejecutando inferencia con KPConv (Esto puede tardar más que RandLANet)...")

results = pipeline.run_inference(data)
pred = results['predict_labels']

print(f"Inferencia completada. Predicciones: {pred.shape}")


colors_map = np.array([
    [0, 0, 0],          
    [245,150,100],      
    [245,230,100],      
    [150,60,30],        
    [180,30,80],        
    [255,0,0],          
    [30,30,255],        
    [200,40,255],       
    [90,30,150],        
    [255,0,255],       
    [255,150,255],      
    [75,0,75],          
    [75,0,175],         
    [0,200,255],        
    [50,120,255],       
    [0,175,0],          
    [0,60,135],         
    [80,240,150],       
    [150,240,255],      
    [0,0,255],          
]) / 255.0

# Asignamos colores
colored_pred = colors_map[np.clip(pred, 0, len(colors_map)-1)]
pcd.colors = o3d.utility.Vector3dVector(colored_pred)

# Guardar
output_filename = f"resultados_kpconv/kpconv_{frame_id}.pcd"
print(f"Guardando resultado en {output_filename}...")

o3d.io.write_point_cloud(output_filename, pcd)
print("¡Guardado!")