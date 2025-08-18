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
    parser = argparse.ArgumentParser(description='Traverse dataset and consolidate defect masks in "gt" directories.')
    parser.add_argument('dataset_root', help='Path to the root directory of the dataset')
    parser.add_argument('output_root', help='Path to the root output directory for consolidated masks')
    parser.add_argument('--gt_dir_name', default='gt', help='Name of the ground truth directories to look for (default: "gt")')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be processed without actually doing it')
    return parser.parse_args()

def find_gt_directories(dataset_root, gt_dir_name='gt'):
    """
    Recursively find all directories named 'gt' (or specified name) in the dataset.
    Returns a list of tuples: (gt_directory_path, relative_path_from_root)
    """
    gt_directories = []
    
    for root, dirs, files in os.walk(dataset_root):
        if gt_dir_name in dirs:
            gt_path = os.path.join(root, gt_dir_name)
            # Calculate relative path from dataset root to the parent of gt directory
            rel_path = os.path.relpath(root, dataset_root)
            gt_directories.append((gt_path, rel_path))
    
    return gt_directories

def extract_grouping_key(filename):
    """
    Extract the grouping key (PartName_InstanceLetter and ImageNumber) from the filename.
    
    Examples:
    - Tapa2M1_C_Cut_MechMind-Nano_Tapa2M1_C_Cut_Nano_009_image_cut_defect_2.png -> ('Tapa2M1_C', '009')
    - SpurGear_D_Warped_MechMind-Nano_SpurGear_D_Warped_Nano_006_warped_defect_1.png -> ('SpurGear_D', '006')
    - Gripper_Closed_C_Hole_MechMind-Nano_Gripper_Closed_C_Hole_Nano_000_image_hole_defect_1.png -> ('Gripper_Closed_C', '000')
    """
    # Remove .png extension
    name_without_ext = filename.replace('.png', '')
    
    # Look for all 3-digit numbers (potential image numbers)
    image_number_pattern = re.compile(r'\d{3}')
    image_number_matches = image_number_pattern.findall(name_without_ext)
    
    if not image_number_matches:
        return None
    
    # Use the last 3-digit number as the image number (most reliable)
    image_number = image_number_matches[-1]
    
    # Split by underscores
    parts = name_without_ext.split('_')
    
    if len(parts) < 3:
        return None
    
    # Strategy: Look for single uppercase letter that appears to be an instance letter
    # The instance letter should be a single uppercase letter (A, B, C, D, etc.)
    # and it should appear early in the filename structure
    
    for i in range(1, min(len(parts), 5)):  # Check first few positions for instance letter
        if len(parts[i]) == 1 and parts[i].isupper() and parts[i].isalpha():
            # Found potential instance letter, construct part name
            part_name_components = parts[:i+1]  # Include everything up to and including the instance letter
            part_name_instance = '_'.join(part_name_components)
            return (part_name_instance, image_number)
    
    # Fallback: if no single letter found in expected positions, 
    # try the original logic (for backwards compatibility)
    if len(parts) >= 2 and len(parts[1]) == 1 and parts[1].isupper() and parts[1].isalpha():
        part_name_instance = f"{parts[0]}_{parts[1]}"
        return (part_name_instance, image_number)
    
    return None

def combine_masks(mask_paths):
    """Combine multiple binary masks into a single mask."""
    if not mask_paths:
        return None
    
    print(f"    Combining {len(mask_paths)} masks:")
    for path in mask_paths:
        print(f"      - {os.path.basename(path)}")
    
    # Load the first mask to get dimensions
    try:
        first_mask = Image.open(mask_paths[0])
        width, height = first_mask.size
        combined_mask = np.zeros((height, width), dtype=np.uint8)
    except Exception as e:
        print(f"    Error loading first mask {mask_paths[0]}: {e}")
        return None
    
    # Combine all masks
    for mask_path in mask_paths:
        try:
            mask = Image.open(mask_path)
            # Convert to grayscale if it's not already
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # Convert to numpy array and binarize
            mask_array = np.array(mask)
            binary_mask = (mask_array > 0).astype(np.uint8) * 255
            
            # Combine with OR operation (any non-zero value becomes 255)
            combined_mask = np.maximum(combined_mask, binary_mask)
        except Exception as e:
            print(f"      Warning: Could not process {mask_path}: {e}")
            continue
    
    # Convert back to PIL Image
    return Image.fromarray(combined_mask.astype(np.uint8), mode='L')

def process_gt_directory(gt_dir_path, output_dir, dry_run=False):
    """Process all masks in a single gt directory and create consolidated masks."""
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if directory has PNG files
    png_files = [f for f in os.listdir(gt_dir_path) if f.endswith('.png')]
    if not png_files:
        print(f"    No PNG files found in {gt_dir_path}")
        return 0, 0, 0
    
    # Group masks by their (PartName_InstanceLetter, ImageNumber) key
    mask_groups = {}
    skipped_files = []
    
    for filename in png_files:
        grouping_key = extract_grouping_key(filename)
        if grouping_key:
            if grouping_key not in mask_groups:
                mask_groups[grouping_key] = []
            mask_groups[grouping_key].append(os.path.join(gt_dir_path, filename))
        else:
            skipped_files.append(filename)
            print(f"    Warning: Could not extract grouping key from {filename}")
            # Debug: show the first few parts of the filename to help troubleshoot
            parts = filename.replace('.png', '').split('_')
            print(f"      First few parts: {parts[:min(6, len(parts))]}")
    
    if not mask_groups:
        print(f"    No processable masks found in {gt_dir_path}")
        return 0, 0, len(skipped_files)
    
    print(f"    Found {len(mask_groups)} unique image groups")
    if skipped_files:
        print(f"    Skipped {len(skipped_files)} files due to naming issues")
    
    # Process each group
    successful_consolidations = 0
    total_groups = len(mask_groups)
    
    for (part_name_instance, image_number), mask_paths in mask_groups.items():
        group_name = f"{part_name_instance} - Image {image_number}"
        
        if len(mask_paths) > 1:
            print(f"    Processing group: {group_name} ({len(mask_paths)} masks)")
            
            if not dry_run:
                combined_mask = combine_masks(mask_paths)
                if combined_mask:
                    output_filename = f"{part_name_instance}_{image_number}_consolidated_mask.png"
                    output_path = os.path.join(output_dir, output_filename)
                    combined_mask.save(output_path)
                    successful_consolidations += 1
                    print(f"      Successfully created: {output_filename}")
                else:
                    print(f"      Failed to create consolidated mask for {group_name}")
            else:
                print(f"      [DRY RUN] Would consolidate into: {part_name_instance}_{image_number}_consolidated_mask.png")
                successful_consolidations += 1
        else:
            print(f"    Processing group: {group_name} (single mask)")
            
            if not dry_run:
                # Copy the single mask to output directory with new naming
                source_path = mask_paths[0]
                output_filename = f"{part_name_instance}_{image_number}_single_mask.png"
                output_path = os.path.join(output_dir, output_filename)
                shutil.copy2(source_path, output_path)
                print(f"      Copied: {output_filename}")
            else:
                print(f"      [DRY RUN] Would copy as: {part_name_instance}_{image_number}_single_mask.png")
    
    return total_groups, successful_consolidations, len(skipped_files)

def process_dataset(dataset_root, output_root, gt_dir_name='gt', dry_run=False):
    """Process entire dataset, finding and processing all gt directories."""
    print(f"Searching for '{gt_dir_name}' directories in: {dataset_root}")
    
    gt_directories = find_gt_directories(dataset_root, gt_dir_name)
    
    if not gt_directories:
        print(f"No '{gt_dir_name}' directories found in the dataset!")
        return
    
    print(f"Found {len(gt_directories)} '{gt_dir_name}' directories:")
    for gt_path, rel_path in gt_directories:
        print(f"  {rel_path}/{gt_dir_name}")
    
    if dry_run:
        print(f"\n[DRY RUN MODE] - No files will be modified")
    
    print(f"\n{'='*80}")
    print("PROCESSING GT DIRECTORIES")
    print(f"{'='*80}")
    
    total_groups_all = 0
    total_consolidations_all = 0
    total_skipped_all = 0
    
    for i, (gt_path, rel_path) in enumerate(gt_directories, 1):
        print(f"\n[{i}/{len(gt_directories)}] Processing: {rel_path}/{gt_dir_name}")
        
        # Create corresponding output directory
        output_dir = os.path.join(output_root, rel_path, gt_dir_name + "_consolidated")
        print(f"  Output directory: {os.path.relpath(output_dir, output_root)}")
        
        groups, consolidations, skipped = process_gt_directory(gt_path, output_dir, dry_run)
        
        total_groups_all += groups
        total_consolidations_all += consolidations
        total_skipped_all += skipped
        
        print(f"  Result: {groups} groups, {consolidations} consolidated, {skipped} skipped")
    
    print(f"\n{'='*80}")
    print("DATASET PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total GT directories processed: {len(gt_directories)}")
    print(f"Total image groups found: {total_groups_all}")
    print(f"Total groups consolidated: {total_consolidations_all}")
    print(f"Total files skipped: {total_skipped_all}")
    
    if not dry_run:
        print(f"\nConsolidated masks saved to: {output_root}")
    else:
        print(f"\n[DRY RUN] Run without --dry_run to actually process files")

def main():
    args = parse_arguments()
    
    if not os.path.isdir(args.dataset_root):
        print(f"Error: Dataset root directory '{args.dataset_root}' does not exist.")
        return
    
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output root: {args.output_root}")
    print(f"Looking for directories named: '{args.gt_dir_name}'")
    
    process_dataset(args.dataset_root, args.output_root, args.gt_dir_name, args.dry_run)

if __name__ == "__main__":
    main()