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
    parser.add_argument('dataset_root', help='Path to the root dataset directory')
    parser.add_argument('--output_root', help='Path to the output root directory (default: overwrites in place)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually performing operations')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def extract_base_filename(filename):
    """
    Extract the base filename by removing the _defect_{n}.png suffix.
    Schema: {PartName}_{InstanceLetter}_{DefectClass}_{CameraName}_{ImageNumber}_image_{defectType}_defect_{defectID}.png
    Returns: base filename without _defect_{defectID}.png and the defect ID
    """
    pattern = re.compile(r'(.+?)_defect_(\d+)\.png$')
    match = pattern.match(filename)
    if match:
        base_filename = match.group(1)
        defect_id = match.group(2)
        return base_filename, defect_id
    return None, None

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

def find_gt_directories(dataset_root):
    """Find all 'gt' subdirectories within defectType directories."""
    gt_dirs = []
    dataset_path = Path(dataset_root)
    
    # Look for gt directories at various levels
    for path in dataset_path.rglob('gt'):
        if path.is_dir():
            gt_dirs.append(path)
    
    return gt_dirs

def process_gt_directory(gt_dir, output_dir=None, dry_run=False, verbose=False):
    """Process all masks in a single gt directory and create consolidated masks."""
    if output_dir is None:
        output_dir = gt_dir  # In-place processing
    elif not dry_run:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing GT directory: {gt_dir}")
    if dry_run:
        print("  [DRY RUN MODE - No files will be modified]")
    
    # Group masks by their base filename
    mask_groups = {}
    png_files = list(gt_dir.glob('*.png'))
    
    if verbose:
        print(f"  Found {len(png_files)} PNG files")
    
    for file_path in png_files:
        filename = file_path.name
        base_filename, defect_id = extract_base_filename(filename)
        
        if base_filename and defect_id:
            if base_filename not in mask_groups:
                mask_groups[base_filename] = []
            mask_groups[base_filename].append(file_path)
            if verbose:
                print(f"    Grouped {filename} -> base: {base_filename}, defect_id: {defect_id}")
        elif verbose:
            print(f"    Skipped {filename} (doesn't match defect mask pattern)")
    
    # Process each group
    successful_consolidations = 0
    total_groups = len(mask_groups)
    
    for base_filename, mask_paths in mask_groups.items():
        if len(mask_paths) > 1:
            print(f"  Consolidating {len(mask_paths)} masks for base: {base_filename}")
            if verbose:
                for mask_path in mask_paths:
                    print(f"    Input: {mask_path.name}")
            
            if not dry_run:
                combined_mask = combine_masks(mask_paths)
                if combined_mask:
                    # Create output filename by replacing _defect_{n} with _mask
                    output_filename = f"{base_filename}_mask.png"
                    if output_dir == gt_dir:
                        # In-place: save consolidated mask and remove individual masks
                        output_path = gt_dir / output_filename
                    else:
                        # Separate output directory
                        output_path = Path(output_dir) / output_filename
                    
                    combined_mask.save(output_path)
                    successful_consolidations += 1
                    print(f"    Created: {output_path}")
                    
                    # If processing in-place, remove the individual mask files
                    if output_dir == gt_dir:
                        for mask_path in mask_paths:
                            try:
                                mask_path.unlink()
                                if verbose:
                                    print(f"    Removed: {mask_path.name}")
                            except Exception as e:
                                print(f"    Warning: Could not remove {mask_path.name}: {e}")
                else:
                    print(f"    Error: Failed to combine masks for {base_filename}")
            else:
                # Dry run mode
                output_filename = f"{base_filename}_mask.png"
                if output_dir == gt_dir:
                    output_path = gt_dir / output_filename
                    print(f"    Would create: {output_path}")
                    print(f"    Would remove {len(mask_paths)} individual mask files:")
                    for mask_path in mask_paths:
                        print(f"      - {mask_path.name}")
                else:
                    output_path = Path(output_dir) / output_filename
                    print(f"    Would create: {output_path}")
                successful_consolidations += 1  # Count for dry run statistics
        else:
            # Single mask - rename it to _mask.png format if needed
            mask_path = mask_paths[0]
            base_filename, defect_id = extract_base_filename(mask_path.name)
            if base_filename:
                new_filename = f"{base_filename}_mask.png"
                if output_dir == gt_dir:
                    # In-place rename
                    new_path = gt_dir / new_filename
                    if mask_path.name != new_filename:  # Only rename if different
                        if not dry_run:
                            mask_path.rename(new_path)
                            print(f"  Renamed single mask: {mask_path.name} -> {new_filename}")
                        else:
                            print(f"  Would rename single mask: {mask_path.name} -> {new_filename}")
                    elif verbose:
                        print(f"  Single mask already has correct name: {mask_path.name}")
                else:
                    # Copy to output directory
                    new_path = Path(output_dir) / new_filename
                    if not dry_run:
                        shutil.copy2(mask_path, new_path)
                        print(f"  Copied single mask: {mask_path.name} -> {new_path}")
                    else:
                        print(f"  Would copy single mask: {mask_path.name} -> {new_path}")
    
    print(f"  Summary - Total groups: {total_groups}, Consolidated: {successful_consolidations}, Single masks: {total_groups - successful_consolidations}")
    return successful_consolidations, total_groups

def main():
    args = parse_arguments()
    
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        print(f"Error: Dataset root directory '{dataset_root}' does not exist.")
        return
    
    # Find all gt directories
    gt_directories = find_gt_directories(dataset_root)
    
    if not gt_directories:
        print(f"No 'gt' directories found in {dataset_root}")
        return
    
    print(f"Found {len(gt_directories)} GT directories to process:")
    for gt_dir in gt_directories:
        print(f"  {gt_dir}")
    
    total_consolidations = 0
    total_groups = 0
    
    # Process each gt directory
    for gt_dir in gt_directories:
        if args.output_root:
            # Create corresponding output directory structure
            rel_path = gt_dir.relative_to(dataset_root)
            output_dir = Path(args.output_root) / rel_path
        else:
            output_dir = None  # In-place processing
        
        consolidations, groups = process_gt_directory(gt_dir, output_dir)
        total_consolidations += consolidations
        total_groups += groups
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Processed {len(gt_directories)} GT directories")
    print(f"Total image groups found: {total_groups}")
    print(f"Successfully consolidated groups: {total_consolidations}")
    print(f"Single mask groups: {total_groups - total_consolidations}")
    
    if args.output_root:
        print(f"Output saved to: {args.output_root}")
    else:
        print("Processing completed in-place")

if __name__ == "__main__":
    main()