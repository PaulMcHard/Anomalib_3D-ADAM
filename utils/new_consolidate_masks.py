#!/usr/bin/env python3
import os
import argparse
import re
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Consolidate defect masks based on a new naming schema.')
    parser.add_argument('input_dir', help='Path to the directory containing all defect masks')
    parser.add_argument('output_dir', help='Path to the output directory for consolidated masks')
    return parser.parse_args()

def extract_part_info(filename):
    """
    Extract part information from the filename using the new schema.
    Schema: {PartName}_{InstanceLetter}_{DefectClass}_{CameraName}_{ImageNumber}_image_{defectType}_defect_{defectID}.png
    """
    pattern = re.compile(r'(.+?)_([A-Z])_(\w+)_(\w+)_(?P<image_number>\d{3})_image_(\w+)_defect_(\d+)\.png')
    match = pattern.match(filename)
    if match:
        return match.group('image_number')
    return None

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

def process_masks(input_dir, output_dir):
    """Process all masks in the input directory and create consolidated masks."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group masks by their image number
    mask_groups = {}
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            image_number = extract_part_info(filename)
            if image_number:
                if image_number not in mask_groups:
                    mask_groups[image_number] = []
                mask_groups[image_number].append(os.path.join(input_dir, filename))
    
    # Process each group
    successful_consolidations = 0
    total_groups = len(mask_groups)
    
    for image_number, mask_paths in mask_groups.items():
        if len(mask_paths) > 1:
            print(f"Consolidating {len(mask_paths)} masks for image number {image_number}...")
            combined_mask = combine_masks(mask_paths)
            if combined_mask:
                output_path = os.path.join(output_dir, f"{image_number}_mask.png")
                combined_mask.save(output_path)
                successful_consolidations += 1
                print(f"Successfully created: {output_path}")
        else:
            print(f"Skipping image number {image_number} as it only has one mask.")
            
    print("\nProcessing complete!")
    print(f"Total image groups found: {total_groups}")
    print(f"Successfully consolidated groups: {successful_consolidations}")
    print(f"Groups skipped (single mask): {total_groups - successful_consolidations}")

def main():
    args = parse_arguments()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return
    
    process_masks(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()