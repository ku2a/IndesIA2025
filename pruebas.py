import open3d as o3d
import numpy as np
import pandas as pd
import open3d.ml.torch as ml
import torch
from open3d.ml.torch.pipelines import SemanticSegmentation

# -----------------------------
# 1. Configurar y Cargar modelo
# -----------------------------
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
    print("Pesos cargados correctamente.")
except Exception as e:
    print(f"Error cargando pesos: {e}")
    exit()

# -----------------------------
# 2. Inicializar el Pipeline
# -----------------------------
pipeline = SemanticSegmentation(model=model, device="cuda", max_knn_search=30)

# -----------------------------
# 3. Cargar nube de puntos CON PANDAS
# -----------------------------
path = "full_ciudad/cloud_full_000798.xyz"
print(f"Cargando {path} con Pandas...")

try:
    # Leemos el archivo como CSV separado por espacios.
    # header=None porque np.savetxt no guarda cabeceras por defecto.
    # Asignamos nombres para ser ordenados.
    df = pd.read_csv(path, sep=' ', header=None, names=['x', 'y', 'z', 'ref', 'nir'])
    
    # Extraemos solo las coordenadas XYZ como array de Numpy (float32)
    # .values convierte el DataFrame a Numpy array
    points = df[['x', 'y', 'z']].values.astype(np.float32)
    
    # (Opcional) Si quisieras usar la reflectividad más tarde, la tendrías aquí:
    # reflectividad = df['ref'].values.astype(np.float32)

    # Creamos el objeto Open3D manualmente
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    print(f"Puntos cargados: {points.shape}")

except Exception as e:
    print(f"Error crítico cargando el archivo con Pandas: {e}")
    print("Asegúrate de que el archivo existe y está separado por espacios.")
    exit()

# -----------------------------
# 4. Preparar datos para inferencia
# -----------------------------
# Creamos features vacíos para que RandLANet use solo la geometría
empty_feats = np.zeros((points.shape[0], 0), dtype=np.float32)

data = {
    'point': points,
    'feat': empty_feats, 
    'label': np.zeros(len(points), dtype=np.int32)
}

# -----------------------------
# 5. Inferencia
# -----------------------------
print("Ejecutando inferencia (calculando vecinos y capas)...")
results = pipeline.run_inference(data)
pred = results['predict_labels']

print(f"Inferencia completada. Predicciones: {pred.shape}")

# -----------------------------
# 6. Guardado y Coloreado
# -----------------------------
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
output_filename = "resultado_segmentacion.pcd"
print(f"Guardando resultado en {output_filename}...")

o3d.io.write_point_cloud(output_filename, pcd)

print("¡Guardado! Descarga el archivo .pcd y ábrelo en CloudCompare.")