import open3d as o3d
import numpy as np
import pandas as pd
import open3d.ml.torch as ml
import torch
from open3d.ml.torch.pipelines import SemanticSegmentation

# -----------------------------
# 1. Configurar y Cargar modelo (KPConv)
# -----------------------------
# NOTA: El nombre oficial suele terminar en '1354utc.pth'. 
# Asegúrate de que tu archivo se llama exactamente así:
pesos = "kpconv_semantickitti_202009090354utc.pth" 

print(f"Configurando modelo KPFCNN con pesos: {pesos}")

# KPConv requiere saber qué valores de etiqueta existen (0 a 19 en SemanticKITTI)
kit_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

model = ml.models.KPFCNN(
    name="KPFCNN",
    dim_input=3,         # X, Y, Z
    num_classes=19,      # Número de clases activas
    ignored_label_inds=[0],
    lbl_values=kit_labels # <--- IMPORTANTE: KPConv necesita esto
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
    print("CONSEJO: Verifica si el nombre del archivo termina en '1354utc.pth' o '0354utc.pth'")
    exit()

# -----------------------------
# 2. Inicializar el Pipeline
# -----------------------------
# KPFCNN usa búsqueda de radio, pero el pipeline maneja la configuración.
pipeline = SemanticSegmentation(model=model, device="cuda")

# -----------------------------
# 3. Cargar nube de puntos CON PANDAS
# -----------------------------
path = "full_ciudad/cloud_full_000793.xyz"
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

# -----------------------------
# 4. Preparar datos para inferencia
# -----------------------------
# KPConv preentrenado en SemanticKITTI espera solo geometría (3 dims).
# Así que pasamos 'feat' vacío para que no concatene nada extra a las coordenadas.
empty_feats = np.zeros((points.shape[0], 0), dtype=np.float32)

data = {
    'point': points,
    'feat': empty_feats, 
    'label': np.zeros(len(points), dtype=np.int32)
}

# -----------------------------
# 5. Inferencia
# -----------------------------
print("Ejecutando inferencia con KPConv (Esto puede tardar más que RandLANet)...")
# KPConv es más pesado, ten paciencia si tarda unos segundos extra
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

# Asignamos colores
colored_pred = colors_map[np.clip(pred, 0, len(colors_map)-1)]
pcd.colors = o3d.utility.Vector3dVector(colored_pred)

# Guardar
output_filename = "resultado_kpconv.pcd"
print(f"Guardando resultado en {output_filename}...")

o3d.io.write_point_cloud(output_filename, pcd)
print("¡Guardado!")