import os
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
import math
import argparse
import re  # Added for regular expressions

def to_snake_case(text):
    """Convert a string to snake case (lowercase with underscores instead of spaces)"""
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    # Convert to lowercase
    text = text.lower()
    # Replace any remaining non-alphanumeric characters (except underscore) with underscores
    text = re.sub(r'[^\w]', '_', text)
    # Replace multiple consecutive underscores with a single one
    text = re.sub(r'_+', '_', text)
    # Remove leading/trailing underscores
    text = text.strip('_')
    return text


def read_ply_organized_pc(ply_path, width=None, height=None):
    """Read a PLY file and convert to organized point cloud format"""
    # Load the PLY file using Open3D
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    num_points = len(points)
    
    # Check if this might be a 1024x1024x3 point cloud
    if num_points == 1024*1024:
        return points.reshape(1024, 1024, 3)
    # Check if this might be a 1536x1024x3 point cloud
    elif num_points == 1536*1024:
        return points.reshape(1536, 1024, 3)
    # Check if this might be a 1536x2048x3 point cloud (3,145,728 points)
    elif num_points == 1536*2048:
        return points.reshape(1536, 2048, 3)
    # Check if this is a perfect square
    elif int(np.sqrt(num_points))**2 == num_points:
        dim = int(np.sqrt(num_points))
        return points.reshape(dim, dim, 3)
    
    # If no exact match, try to infer dimensions
    if width is None or height is None:
        # Try to find the closest factors of num_points that make sense for a grid
        # This is a more sophisticated approach than just taking the square root
        possible_heights = []
        for i in range(int(np.sqrt(num_points)), int(np.sqrt(num_points)/2), -1):
            if num_points % i == 0:
                possible_heights.append(i)
                
        if possible_heights:
            height = possible_heights[0]  # Take the largest factor
            width = num_points // height
            print(f"Inferred dimensions: {height}x{width} for {num_points} points")
        else:
            # If no exact factors, use the square root approach with padding/truncation
            height = int(np.sqrt(num_points))
            width = height
            print(f"No exact factors found. Using approximate dimensions: {height}x{width}")
    
    # Reshape to organized format
    try:
        organized_pc = points.reshape(height, width, 3)
        return organized_pc
    except ValueError:
        print(f"Warning: Could not reshape point cloud with {len(points)} points to {height}x{width}x3")
        # Create a properly sized array
        target_size = height * width
        if num_points > target_size:
            # Truncate the points
            print(f"Truncating {num_points - target_size} points")
            padded_points = points[:target_size]
        else:
            # Pad with zeros
            print(f"Padding with {target_size - num_points} zero points")
            padded_points = np.zeros((target_size, 3))
            padded_points[:num_points] = points
            
        return padded_points.reshape(height, width, 3)


def get_edges_of_pc(organized_pc):
    unorganized_edges_pc = organized_pc[0:10, :, :].reshape(organized_pc[0:10, :, :].shape[0]*organized_pc[0:10, :, :].shape[1],organized_pc[0:10, :, :].shape[2])
    unorganized_edges_pc = np.concatenate([unorganized_edges_pc,organized_pc[-10:, :, :].reshape(organized_pc[-10:, :, :].shape[0] * organized_pc[-10:, :, :].shape[1],organized_pc[-10:, :, :].shape[2])],axis=0)
    unorganized_edges_pc = np.concatenate([unorganized_edges_pc, organized_pc[:, 0:10, :].reshape(organized_pc[:, 0:10, :].shape[0] * organized_pc[:, 0:10, :].shape[1],organized_pc[:, 0:10, :].shape[2])], axis=0)
    unorganized_edges_pc = np.concatenate([unorganized_edges_pc, organized_pc[:, -10:, :].reshape(organized_pc[:, -10:, :].shape[0] * organized_pc[:, -10:, :].shape[1],organized_pc[:, -10:, :].shape[2])], axis=0)
    unorganized_edges_pc = unorganized_edges_pc[np.nonzero(np.all(unorganized_edges_pc != 0, axis=1))[0],:]
    return unorganized_edges_pc


def get_plane_eq(unorganized_pc, ransac_n_pts=50):
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc))
    plane_model, inliers = o3d_pc.segment_plane(distance_threshold=0.004, ransac_n=ransac_n_pts, num_iterations=1000)
    return plane_model


def organized_pc_to_unorganized_pc(organized_pc):
    """Convert organized point cloud to unorganized format"""
    shape = organized_pc.shape
    if len(shape) == 3:
        return organized_pc.reshape(-1, shape[2])
    else:
        return organized_pc.reshape(-1)


def remove_plane(organized_pc_clean, organized_rgb, distance_threshold=0.005):
    # PREP PC
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc_clean)
    unorganized_rgb = organized_pc_to_unorganized_pc(organized_rgb)
    clean_planeless_unorganized_pc = unorganized_pc.copy()
    planeless_unorganized_rgb = unorganized_rgb.copy()

    # REMOVE PLANE
    plane_model = get_plane_eq(get_edges_of_pc(organized_pc_clean))
    distances = np.abs(np.dot(np.array(plane_model), np.hstack((clean_planeless_unorganized_pc, np.ones((clean_planeless_unorganized_pc.shape[0], 1)))).T))
    plane_indices = np.argwhere(distances < distance_threshold)

    planeless_unorganized_rgb[plane_indices] = 0
    clean_planeless_unorganized_pc[plane_indices] = 0
    clean_planeless_organized_pc = clean_planeless_unorganized_pc.reshape(organized_pc_clean.shape[0],
                                                                          organized_pc_clean.shape[1],
                                                                          organized_pc_clean.shape[2])
    planeless_organized_rgb = planeless_unorganized_rgb.reshape(organized_rgb.shape[0],
                                                                          organized_rgb.shape[1],
                                                                          organized_rgb.shape[2])
    return clean_planeless_organized_pc, planeless_organized_rgb


def connected_components_cleaning(organized_pc, organized_rgb, image_path):
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc)
    unorganized_rgb = organized_pc_to_unorganized_pc(organized_rgb)

    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    labels = np.array(o3d_pc.cluster_dbscan(eps=0.006, min_points=30, print_progress=False))

    unique_cluster_ids, cluster_size = np.unique(labels, return_counts=True)
    max_label = labels.max()
    if max_label > 0:
        print("##########################################################################")
        print(f"Point cloud file {image_path} has {max_label + 1} clusters")
        print(f"Cluster ids: {unique_cluster_ids}. Cluster size {cluster_size}")
        print("##########################################################################\n\n")

    largest_cluster_id = unique_cluster_ids[np.argmax(cluster_size)]
    outlier_indices_nonzero_array = np.argwhere(labels != largest_cluster_id)
    outlier_indices_original_pc_array = nonzero_indices[outlier_indices_nonzero_array]
    unorganized_pc[outlier_indices_original_pc_array] = 0
    unorganized_rgb[outlier_indices_original_pc_array] = 0
    organized_clustered_pc = unorganized_pc.reshape(organized_pc.shape[0],
                                                    organized_pc.shape[1],
                                                    organized_pc.shape[2])
    organized_clustered_rgb = unorganized_rgb.reshape(organized_rgb.shape[0],
                                                    organized_rgb.shape[1],
                                                    organized_rgb.shape[2])
    return organized_clustered_pc, organized_clustered_rgb


def roundup_next_100(x):
    return int(math.ceil(x / 100.0)) * 100


def pad_cropped_pc(cropped_pc, single_channel=False):
    orig_h, orig_w = cropped_pc.shape[0], cropped_pc.shape[1]
    round_orig_h = roundup_next_100(orig_h)
    round_orig_w = roundup_next_100(orig_w)
    large_side = max(round_orig_h, round_orig_w)

    a = (large_side - orig_h) // 2
    aa = large_side - a - orig_h

    b = (large_side - orig_w) // 2
    bb = large_side - b - orig_w
    if single_channel:
        return np.pad(cropped_pc, pad_width=((a, aa), (b, bb)), mode='constant')
    else:
        return np.pad(cropped_pc, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')


def preprocess_pc(ply_path):
    # READ FILES
    organized_pc = read_ply_organized_pc(ply_path)
    
    # Handle the complex directory structure:
    # - PLY files are in a 'xyz' directory with "_cloud.ply" suffix
    # - RGB images are in a parallel 'rgb' directory with "_image.png" suffix
    # - GT images are in a parallel 'gt' directory with "defect_type_defect_n.png" pattern
    
    # First, extract the base filename without extension
    base_filename = os.path.basename(ply_path).replace("_cloud.ply", "")
    
    # Now get parent directory of xyz (which contains subdirectories)
    xyz_dir = os.path.dirname(ply_path)
    parent_dir = os.path.dirname(xyz_dir)  # One level up from xyz
    
    # Extract defect_type from parent directory name if possible
    defect_type = os.path.basename(parent_dir)  # This might be the defect type
    # Convert defect_type to snake case
    defect_type = to_snake_case(defect_type)
    print(f"Detected defect type (from directory): {defect_type}")
    
    # Construct paths for RGB and possible GT images
    rgb_dir = os.path.join(parent_dir, "rgb")
    gt_dir = os.path.join(parent_dir, "gt")
    
    # Path to corresponding RGB image
    rgb_path = os.path.join(rgb_dir, f"{base_filename}_image.png")
    print(f"Looking for RGB image at: {rgb_path}")
    
    # For GT, we need to check for files that match the pattern
    gt_exists = False
    gt_path = None
    
    if os.path.exists(gt_dir):
        # First try the exact pattern if defect_type is available
        potential_gt_pattern = f"{defect_type}_defect_*.png"
        potential_gt_files = list(Path(gt_dir).glob(potential_gt_pattern))
        
        # If no files found, try a more generic pattern
        if not potential_gt_files:
            potential_gt_files = list(Path(gt_dir).glob("*_defect_*.png"))
        
        # If we found any potential GT files, try to match to base filename
        if potential_gt_files:
            # Try to find a match based on indices/numbers in the filename
            # Extract the numeric part from base_filename
            base_num_match = re.search(r'\d+', base_filename)
            if base_num_match:
                base_num = base_num_match.group()
                
                # Look for GT files with the same number
                for gt_file in potential_gt_files:
                    if base_num in str(gt_file):
                        gt_path = str(gt_file)
                        gt_exists = True
                        print(f"Found matching GT file: {gt_path}")
                        break
            
            # If no match found but GT files exist, just use the first one
            if not gt_exists and potential_gt_files:
                gt_path = str(potential_gt_files[0])
                gt_exists = True
                print(f"Using first available GT file: {gt_path}")
    
    if os.path.exists(rgb_path):
        organized_rgb = np.array(Image.open(rgb_path))
    else:
        # If no RGB image exists, create a dummy one
        organized_rgb = np.zeros((organized_pc.shape[0], organized_pc.shape[1], 3), dtype=np.uint8)
        print(f"Warning: RGB file not found at {rgb_path}. Creating dummy RGB.")

    organized_gt = None
    if gt_exists and gt_path and os.path.isfile(gt_path):
        organized_gt = np.array(Image.open(gt_path))
        print(f"Loaded ground truth from {gt_path}")
    else:
        print("No ground truth file found or matched")

    # REMOVE PLANE
    planeless_organized_pc, planeless_organized_rgb = remove_plane(organized_pc, organized_rgb)

    # PAD WITH ZEROS TO LARGEST SIDE (SO THAT THE FINAL IMAGE IS SQUARE)
    padded_planeless_organized_pc = pad_cropped_pc(planeless_organized_pc, single_channel=False)
    padded_planeless_organized_rgb = pad_cropped_pc(planeless_organized_rgb, single_channel=False)
    if gt_exists:
        padded_organized_gt = pad_cropped_pc(organized_gt, single_channel=True)

    organized_clustered_pc, organized_clustered_rgb = connected_components_cleaning(
        padded_planeless_organized_pc, padded_planeless_organized_rgb, ply_path)
    
    # SAVE PREPROCESSED FILES
    # First, let's save a debug image of the point cloud depth
    depth_image = np.linalg.norm(organized_clustered_pc, axis=2)
    normalized_depth = np.zeros_like(depth_image)
    if np.max(depth_image) > 0:  # Avoid division by zero
        normalized_depth = (depth_image / np.max(depth_image) * 255).astype(np.uint8)
    
    # Save the debug image in the same directory as the original PLY
    debug_path = str(ply_path).replace("_cloud.ply", "_depth_debug.png")
    Image.fromarray(normalized_depth).save(debug_path)
    print(f"Saved depth debug image to {debug_path}")
    
    # Convert to point cloud for PLY saving
    points = organized_clustered_pc.reshape(-1, 3)
    colors = organized_clustered_rgb.reshape(-1, 3) / 255.0  # Normalize colors to [0,1]
    
    # Filter out zero points
    valid_indices = np.where(np.all(points != 0, axis=1))[0]
    if len(valid_indices) == 0:
        print(f"WARNING: No valid points found in processed point cloud for {ply_path}")
        valid_points = points  # Use all points even if they're zeros
        valid_colors = colors
    else:
        valid_points = points[valid_indices]
        valid_colors = colors[valid_indices]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    pcd.colors = o3d.utility.Vector3dVector(valid_colors)
    
    # Save the processed point cloud in-place
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"Saved processed point cloud with {len(valid_points)} points to {ply_path}")
    
    # Save the processed RGB in the rgb directory
    if os.path.exists(rgb_path):
        Image.fromarray(organized_clustered_rgb).save(rgb_path)
        print(f"Saved processed RGB to {rgb_path}")
    
    # Save GT if it exists (in the same location as the original GT)
    if gt_exists and gt_path:
        Image.fromarray(padded_organized_gt).save(gt_path)
        print(f"Saved processed ground truth to {gt_path}")
    
    # Save RGB
    Image.fromarray(organized_clustered_rgb).save(rgb_path)
    
    # Save GT if it exists
    if gt_exists:
        Image.fromarray(padded_organized_gt).save(gt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess 3D point cloud data from PLY files')
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='The root path of the dataset. The preprocessing is done inplace')
    args = parser.parse_args()

    root_path = args.dataset_path
    paths = list(Path(root_path).rglob('*_cloud.ply'))  # Updated to match the specific naming convention
    print(f"Found {len(paths)} PLY files in {root_path}")
    
    processed_files = 0
    for path in paths:
        try:
            print(f"Processing {path}...")
            preprocess_pc(path)
            processed_files += 1
            if processed_files % 10 == 0:
                print(f"Processed {processed_files}/{len(paths)} PLY files...")
        except Exception as e:
            print(f'Error while processing {path}: {str(e)}')
    
    print(f"Preprocessing complete. Processed {processed_files} PLY files.")