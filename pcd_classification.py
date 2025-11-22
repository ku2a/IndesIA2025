import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

path1="resultados_kpconv/kpconv_000373.pcd"
path2="resultados_kpconv/kpconv_000261.pcd"
path3="resultados_kpconv/kpconv_000230.pcd"
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

def view_semantic_pcd(filename):
    print(f"Loading {filename}...")
    
    pcd = o3d.io.read_point_cloud(filename)
    
    if pcd.is_empty():
        print("Error: Cloud is empty.")
        return

    if not pcd.has_colors():
        points = np.asarray(pcd.points)
        
        if np.asarray(pcd.colors).shape[0] > 0:
             # If loaded as grayscale RGB, take the R channel as the ID
             # (Multiply by 255 if it was normalized on load)
             labels = np.asarray(pcd.colors)[:, 0]
             # Heuristic: If values are small floats (0.0-1.0), they might be normalized
             if labels.max() <= 1.0: 
                 labels = (labels * 255).astype(int)
             else:
                 labels = labels.astype(int)
        else:
             print("No colors/intensity found to use as labels.")
             return

        print(f"Found {len(points)} points.")
        print(f"Unique Class IDs found in file: {np.unique(labels)}")

        # 3. Apply Colors based on the Map
        new_colors = np.zeros((len(points), 3))
        
        # Vectorized coloring
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            if lbl in colors_map:
                new_colors[labels == lbl] = colors_map[lbl]
            else:
                # Default to gray for unknown labels
                new_colors[labels == lbl] = [0.5, 0.5, 0.5] 

        pcd.colors = o3d.utility.Vector3dVector(new_colors)
    
    else:
        print("Cloud already has colors, displaying as is...")

    o3d.visualization.draw_geometries([pcd], window_name="Semantic Segmentation Results")
    return pcd


## Sidewalk
def detect_sidewalk(path):
    pcd = view_semantic_pcd(path)

    TARGET_RGB = [255,150,255]

    float_colors = np.asarray(pcd.colors)
    int_colors = np.round(float_colors * 255).astype(int)

    mask = np.all(int_colors == TARGET_RGB, axis=1)
    road_indices = np.where(mask)[0]

    print(f"Found {len(road_indices)} road points.")
    road_cloud = pcd.select_by_index(road_indices)
    o3d.visualization.draw_geometries([road_cloud])

    # Noise cleaning
    clean_cloud, ind = road_cloud.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.5)
    o3d.visualization.draw_geometries([clean_cloud])

    obb = clean_cloud.get_oriented_bounding_box()
    obb.color = (0, 0, 0)

    extent = obb.extent
    sorted_extent = np.sort(extent)
    
    height = sorted_extent[0] # Road thickness (should be small)
    width  = sorted_extent[1] # Road width
    length = sorted_extent[2] # Road length

    print(f"Segment Length: {length:.2f} m")
    print(f"Segment Width:  {width:.2f} m")

    road_cloud.paint_uniform_color([1, 0, 1]) 
    pcd.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([clean_cloud, obb])

    labels = np.array(clean_cloud.cluster_dbscan(eps=2.0, min_points=100, print_progress=True))
    max_label = labels.max()
    print(f"Found {max_label + 1} distinct sidewalk segments.")

    if max_label < 1:
        print("[ERROR] Less than two sidewalk segments found. Cannot measure width.")
        return
    
    # Measure sidewalk
    indices_A = np.where(labels == 0)[0]
    cloud_A = clean_cloud.select_by_index(indices_A)
    
    indices_B = np.where(labels == 1)[0]
    cloud_B = clean_cloud.select_by_index(indices_B)

    distance_A_to_B = cloud_A.compute_point_cloud_distance(cloud_B)
    distance_B_to_A = cloud_B.compute_point_cloud_distance(cloud_A)

    closest_dist_A_to_B = np.asarray(distance_A_to_B)
    idx_in_A = np.argmin(closest_dist_A_to_B)
    closest_dist_B_to_A = np.asarray(distance_B_to_A)
    idx_in_B = np.argmin(closest_dist_B_to_A)

    points_A = np.asarray(cloud_A.points)
    points_B = np.asarray(cloud_B.points)
    
    inner_edge_A = points_A[idx_in_A]
    inner_edge_B = points_B[idx_in_B]
    
    # Calculate the Euclidean distance between the two points
    road_width_3D = np.linalg.norm(inner_edge_A - inner_edge_B)

    print(f"Curb-to-Curb Road Width: {road_width_3D:.2f} m")

    cluster = 1
    side_indices = np.where(labels == cluster)[0]
    single_side_cloud = clean_cloud.select_by_index(side_indices)
    o3d.visualization.draw_geometries([single_side_cloud])

    obb = single_side_cloud.get_oriented_bounding_box()
    obb.color = (0, 0, 0)

    single_side_cloud, ind = single_side_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)
    o3d.visualization.draw_geometries([single_side_cloud])

    clean_cloud.paint_uniform_color([1, 0, 0])
    single_side_cloud.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([single_side_cloud, obb])

detect_sidewalk("resultados_randlanet/randlanet_000585.pcd")
detect_sidewalk("resultados_randlanet/randlanet_000047.pcd")


## Troncos
def detect_trunks(path):


    nombre_archivo = os.path.basename(path)  
    try:

        frame_id = nombre_archivo.split('_')[-1].replace('.xyz', '')
    except:
        frame_id = "unknown"

    output_name = f"class/trunks_{frame_id}"
    pcd = view_semantic_pcd(path)
    
    TARGET_RGB = [0,175,0]

    float_colors = np.asarray(pcd.colors)
    int_colors = np.round(float_colors * 255).astype(int)

    mask = np.all(int_colors == TARGET_RGB, axis=1)
    trunk_indices = np.where(mask)[0]

    print(f"Found {len(trunk_indices)} road points.")
    trunk_cloud = pcd.select_by_index(trunk_indices)
    o3d.visualization.draw_geometries([trunk_cloud])

    clean_cloud_trunks, ind = trunk_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)
    
    labels = np.array(clean_cloud_trunks.cluster_dbscan(eps=2, min_points=50, print_progress=True))
    max_label = labels.max()
    print(f"Found {max_label + 1} distinct trees segments.")

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    clean_cloud_trunks.colors = o3d.utility.Vector3dVector(colors[:, :3])
    pcd.paint_uniform_color([0,0,0])

    dists = pcd.compute_point_cloud_distance(clean_cloud_trunks)
    dists = np.asarray(dists)
    non_trunk_indices = np.where(dists > 0.001)[0]
    background_cloud = pcd.select_by_index(non_trunk_indices)

    background_cloud.paint_uniform_color([0.4, 0.4, 0.4])
    o3d.visualization.draw_geometries([background_cloud, clean_cloud_trunks])

    visuals = [background_cloud, clean_cloud_trunks]
    
    # Save point cloud to file
    combined_cloud = visuals[0] 
    for pcd_obj in visuals[1:]:
        combined_cloud += pcd_obj

    for i in range(max_label+1):
        ind = np.where(labels == i)[0]
        single_trunk_cloud = clean_cloud_trunks.select_by_index(ind)

        points = np.asarray(single_trunk_cloud.points)
        zs = points[:, 2]
        min_z = np.min(zs)
        max_z = np.max(zs) 
        
        aabb = single_trunk_cloud.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)
        extent = aabb.get_extent()
        
        height = extent[2] 
        width  = max(extent[0], extent[1]) 
        center = aabb.get_center()

        if height > 0.5:
            print(f"Tree {i}: Height = {height:.2f}m | Diameter ~ {width:.2f}m | Loc: ({center[0]:.1f}, {center[1]:.1f})")
            
            # Add the box to our list of visuals
            visuals.append(aabb)

    o3d.visualization.draw_geometries(visuals)
    return output_name,combined_cloud


visuals = detect_trunks(path1)
o3d.io.write_point_cloud(visuals[0],visuals[1])
visuals = detect_trunks(path2)
o3d.io.write_point_cloud(visuals[0],visuals[1])
visuals = detect_trunks(path3)
o3d.io.write_point_cloud(visuals[0],visuals[1])


## Trees with randlanet model

path_to_file = "resultados_randlanet/randlanet_001014.pcd" 
pcd = view_semantic_pcd(path_to_file)

TARGET_RGB = [0,175,0]

float_colors = np.asarray(pcd.colors)
int_colors = np.round(float_colors * 255).astype(int)

mask = np.all(int_colors == TARGET_RGB, axis=1)
tree_indices = np.where(mask)[0]

print(f"Found {len(tree_indices)} tree points.")
tree_cloud = pcd.select_by_index(tree_indices)
o3d.visualization.draw_geometries([tree_cloud])

clean_cloud, ind = tree_cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=2)
o3d.visualization.draw_geometries([clean_cloud])

labels = np.array(clean_cloud.cluster_dbscan(eps=2.2, min_points=100, print_progress=True))
max_label = labels.max()
print(f"Found {max_label + 1} distinct trees segments.")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
clean_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([clean_cloud])

clusters = []
for i in range(1,3):
    ind = np.where(labels == i)[0]
    clusters.append(clean_cloud.select_by_index(ind))
o3d.visualization.draw_geometries(clusters)

