import open3d as o3d
import numpy as np
import open3d.ml.torch as ml
import torch
from open3d.ml.torch.pipelines import SemanticSegmentation # <--- IMPORTANTE
import glob, os


archivos = glob.glob("full_ciudad/*.xyz")
os.makedirs("resultados_procesados", exist_ok=True)
# -----------------------------
# 1. Configurar y Cargar modelo
# -----------------------------
pesos = "randlanet_semantickitti_202201071330utc.pth"

# Definimos el modelo. dim_input=3 significa que usaremos X,Y,Z como features.
model = ml.models.RandLANet(
    name="RandLANet",
    dim_input=3, 
    num_classes=19,
    ignored_label_inds=[0] # La clase 0 suele ser "ignorada" en SemanticKITTI
)

# Cargar pesos (Tu lógica estaba bien aquí)
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
# El pipeline envuelve al modelo y se encarga del pre-procesamiento (vecinos, etc.)
pipeline = SemanticSegmentation(model=model, device="cuda", max_knn_search=30)

# -----------------------------
# 3. Cargar nube de puntos
# -----------------------------
for path in archivos:

    # Intento robusto de carga (Open3D a veces falla con XYZ simples)
    try:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            # Fallback a numpy si Open3D falla leyendo el header
            points = np.loadtxt(path, dtype=np.float32)[:, :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
        else:
            points = np.asarray(pcd.points).astype(np.float32)
            
        print(f"Puntos cargados: {points.shape}")

    except Exception as e:
        print(f"No se pudo cargar el archivo: {e}")
        exit()

    # -----------------------------
    # 4. Preparar datos para inferencia
    # -----------------------------
    # El pipeline espera un diccionario con claves específicas.
    empty_feats = np.zeros((points.shape[0], 0), dtype=np.float32)
    data = {
        'point': points,
        # IMPORTANTE: RandLANet espera features de entrada. 
        # Si no tienes color, usamos las mismas coordenadas XYZ como features (Nx3).
        # Tu código anterior usaba np.ones((N,1)), eso fallaría porque el modelo espera dim 3.
        'feat': empty_feats, 
        'label': np.zeros(len(points), dtype=np.int32) # Dummy labels requeridos por la API
    }

    # -----------------------------
    # 5. Inferencia
    # -----------------------------
    print("Ejecutando inferencia (calculando vecinos y capas)...")
    results = pipeline.run_inference(data)
    pred = results['predict_labels'] # Array con las clases (0-18)

    print(f"Inferencia completada. Predicciones: {pred.shape}")

    # -----------------------------
    # 6. Visualización
    # -----------------------------

    # Paleta de colores SemanticKITTI (Extendida para seguridad)
    # Aseguramos que sea float 0-1
    colors_map = np.array([
        [0, 0, 0],          # 0: ignored/unlabeled
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

    # Asignamos colores. Usamos 'clip' para evitar errores si predice algo fuera de rango
    colored_pred = colors_map[np.clip(pred, 0, len(colors_map)-1)]

    pcd.colors = o3d.utility.Vector3dVector(colored_pred)

    print("Abriendo visualizador...")
    o3d.visualization.draw_geometries([pcd], window_name="Resultado Segmentación")


    # ... (Toda la parte de inferencia y colors_map se queda igual) ...

    # Asignamos colores según la predicción
    colored_pred = colors_map[np.clip(pred, 0, len(colors_map)-1)]
    pcd.colors = o3d.utility.Vector3dVector(colored_pred)

    # --- CAMBIO AQUÍ: EN LUGAR DE DIBUJAR, GUARDAMOS ---

    nombre_base = os.path.basename(path) # "cloud_001.xyz"
    ruta_salida = os.path.join("resultados_procesados", nombre_base.replace(".xyz", ".pcd"))

    print(f"Guardando resultado en {ruta_salida}...")

    # Guardar la nube coloreada
    o3d.io.write_point_cloud(ruta_salida, pcd)

    print("¡Guardado! Descarga el archivo .pcd y ábrelo en CloudCompare o MeshLab.")

    # Comentamos la línea que da error
    # o3d.visualization.draw_geometries([pcd], window_name="Resultado Segmentación")