import open3d as o3d
import numpy as np
import pandas as pd
import open3d.ml.torch as ml
import torch
from open3d.ml.torch.pipelines import SemanticSegmentation
import os,requests



filename="randlanet_semantickitti_202201071330utc.pth"
url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_semantickitti_202201071330utc.pth"



if os.path.exists(filename):
    print("Modelo detectado")
else:
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status() 
        
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)


path = "full_ciudad/cloud_full_000585.xyz"

nombre_archivo = os.path.basename(path)  
try:

    frame_id = nombre_archivo.split('_')[-1].replace('.xyz', '')
except:
    frame_id = "unknown"


#modelo preentrenado
pesos = "randlanet_semantickitti_202201071330utc.pth"

model = ml.models.RandLANet(
    name="RandLANet",
    dim_input=3, 
    num_classes=19,
    ignored_label_inds=[0]
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


pipeline = SemanticSegmentation(model=model, device="cuda", max_knn_search=30)




try:

    df = pd.read_csv(path, sep=' ', header=None, names=['x', 'y', 'z', 'ref', 'nir'])
    
    # Extraemos solo las coordenadas XYZ como array de Numpy (float32)

    points = df[['x', 'y', 'z']].values.astype(np.float32)
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    

except Exception as e:
    print(e)
    exit()


empty_feats = np.zeros((points.shape[0], 0), dtype=np.float32)

data = {
    'point': points,
    'feat': empty_feats, 
    'label': np.zeros(len(points), dtype=np.int32)
}



results = pipeline.run_inference(data)
pred = results['predict_labels']




colors_map = np.array([
    [0, 0, 0],          # 0: ignored
    [245,150,100],      # 1: car
    [245,230,100],      # 2: bicycle
    [150,60,30],        # 3: motorcycle
    [180,30,80],        # 4: truck
    [255,0,0],          # 5: other-vehicle
    [30,30,255],        # 6: person
    [200,40,255],       # 7: bicyclist
    [90,30,150],        # 8: motorcyclist
    [255,0,255],        # 9: road
    [255,150,255],      # 10: parking
    [75,0,75],          # 11: sidewalk
    [75,0,175],         # 12: other-ground
    [0,200,255],        # 13: building
    [50,120,255],       # 14: fence
    [0,175,0],          # 15: vegetation
    [0,60,135],         # 16: trunk
    [80,240,150],       # 17: terrain
    [150,240,255],      # 18: pole
    [0,0,255],          # 19: traffic-sign
]) / 255.0

# Asignamos colores según la predicción
colored_pred = colors_map[np.clip(pred, 0, len(colors_map)-1)]
pcd.colors = o3d.utility.Vector3dVector(colored_pred)

# Guardar resultado
output_filename = f"resultados_randlanet/randlanet_{frame_id}.pcd"
print(f"Guardando resultado en {output_filename}...")

o3d.io.write_point_cloud(output_filename, pcd)

