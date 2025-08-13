import os
import argparse
import re
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Consolidate defect masks to match labeller format.')
    
    parser.add_argument('dataset_dir', help='Path to the original dataset directory')
    parser.add_argument('output_dir', help='Path to the output directory for consolidated masks')
    
    return parser.parse_args()

def extract_part_info(filename):
    """Extract part name and image identifier from a labelled mask filename."""
    # Expected format: PartName_Nano_00X_labelled.png
    match = re.match(r'(.+_Nano)_(\d+)_labelled\.png', filename)
    if match:
        part_name_with_nano = match.group(1)
        part_name = part_name_with_nano.replace('_Nano', '')  # Remove the _Nano suffix for dataset matching
        image_id = match.group(2)
        return part_name, part_name_with_nano, image_id
    return None, None, None

def find_defect_masks(dir, part_name, image_id):
    """Find all defect masks for a specific part and image ID in the dataset."""
    # Expected path: dataset_dir/PartName/MechMind-Nano/defect_masks/
    # Expected files: PartName_Nano_00X_{defect_type}_defect_n.png

    if not os.path.exists(dir):
        print(f"Warning: Defect masks directory not found: {dir}")
        return []
    
    # Look for all masks with the matching image ID
    mask_pattern = re.compile(f"{part_name}_Nano_{image_id}_.*_defect_\\d+\\.png")
    
    matching_masks = []
    for filename in os.listdir(dir):
        if mask_pattern.match(filename):
            matching_masks.append(os.path.join(dir, filename))
    
    if not matching_masks:
        print(f"Warning: No matching defect masks found for {part_name}_Nano_{image_id}")
    
    return matching_masks

def combine_masks(mask_paths):
    """Combine multiple binary masks into a single mask."""
    if not mask_paths:
        return None
    
    # Load the first mask to get dimensions
    first_mask = Image.open(mask_paths[0])
    width, height = first_mask.size
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Combine all masks
    for mask_path in mask_paths:
        mask = Image.open(mask_path)
        # Convert to grayscale if it's not already
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Convert to numpy array and binarize
        mask_array = np.array(mask)
        binary_mask = (mask_array > 0).astype(np.uint8) * 255
        
        # Combine with OR operation (any non-zero value becomes 255)
        combined_mask = np.maximum(combined_mask, binary_mask)
    
    # Convert back to PIL Image, ensuring it's the right data type
    return Image.fromarray(combined_mask.astype(np.uint8))

def process_annotations(dataset_dir, annotation_dir, output_dir, camera_name):
    """Process all annotations and create consolidated masks."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics
    total_processed = 0
    successful_matches = 0
    
    # Iterate through all directories in the annotation set
    for root, dirs, files in os.walk(annotation_dir):
        for filename in files:
            if filename.endswith('_labelled.png'):
                part_name, part_name_with_nano, image_id = extract_part_info(filename)
                
                if not part_name:
                    print(f"Warning: Could not parse filename format: {filename}")
                    continue
                
                total_processed += 1
                
                # Find corresponding defect masks in the dataset
                defect_masks = find_defect_masks(dataset_dir, part_name, camera_name, image_id)
                
                if defect_masks:
                    # Combine the masks
                    combined_mask = combine_masks(defect_masks)
                    
                    if combined_mask:
                        # Create output directory structure matching annotation set
                        rel_path = os.path.relpath(root, annotation_dir)
                        output_subdir = os.path.join(output_dir, rel_path)
                        os.makedirs(output_subdir, exist_ok=True)
                        
                        # Save the combined mask
                        output_path = os.path.join(output_subdir, filename.replace('_labelled.png', '_combined.png'))
                        combined_mask.save(output_path)
                        
                        print(f"Created combined mask: {output_path} from {len(defect_masks)} masks")
                        successful_matches += 1
    
    print(f"\nProcessing complete!")
    print(f"Total annotations processed: {total_processed}")
    print(f"Successful matches: {successful_matches}")
    print(f"Failed matches: {total_processed - successful_matches}")

def main():
    args = parse_arguments()
    
    # Ensure input directories exist
    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' does not exist.")
        return
    
    if not os.path.isdir(args.annotation_dir):
        print(f"Error: Annotation directory '{args.annotation_dir}' does not exist.")
        return
    
    # Process the annotations
    process_annotations(args.dataset_dir, args.annotation_dir, args.output_dir, args.camera_name)

if __name__ == "__main__":
    main()