#!/usr/bin/env python3
import os
import argparse
import re
from pathlib import Path
import collections

def shorten_filenames(directory, dry_run):
    """
    Looks into all subdirectories and shortens filenames to just the image number,
    using the full Instance ID and part name to resolve any potential clashes.
    """
    print(f"Starting filename shortening in '{directory}'. Dry run mode: {dry_run}\n")
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        if not files:
            continue

        print(f"--- Processing directory: {root} ---")
        
        # Dictionary to group filenames by image number
        grouped_files = collections.defaultdict(list)
        
        # First pass: Group files by image number and check for valid format
        for filename in files:
            # Expected pattern: {PartName}_{InstanceID}_..._{ImageNumber}_...png
            match = re.match(r'(.+?)_([A-Z]\d*)_.+_(\d{3})_.*\.png', filename)
            if match:
                part_name = match.group(1)
                instance_id = match.group(2)
                image_number = match.group(3)
                grouped_files[image_number].append({
                    'original_name': filename, 
                    'part_name': part_name, 
                    'instance_id': instance_id
                })
            else:
                print(f"  - Warning: Skipping file '{filename}' with an unexpected naming format.")
        
        # Second pass: Process groups and rename files
        for image_number, file_list in grouped_files.items():
            if len(file_list) > 1:
                # Clash detected, use Instance ID and part name to resolve
                print(f"  - CLASH DETECTED for image number '{image_number}'. Resolving with custom suffixes.")
                
                # Check for the specific Gripper case
                #gripper_clash = all('Gripper' in f['part_name'] for f in file_list) and all('C' in f['instance_id'] for f in file_list)
#
                #if gripper_clash:
                #    for file_info in file_list:
                #        original_name = file_info['original_name']
                #        part_name = file_info['part_name']
                #        
                #        suffix = None
                #        if 'Closed' in part_name:
                #            suffix = 'C1'
                #        elif 'Open' in part_name:
                #            suffix = 'C2'
                #        
                #        if suffix:
                #            new_name = f"{image_number}_{suffix}.png"
                #            src = os.path.join(root, original_name)
                #            dst = os.path.join(root, new_name)
                #            print(f"    - {'[DRY RUN] ' if dry_run else ''}Renaming '{original_name}' to '{new_name}'")
                #            if not dry_run:
                #                os.rename(src, dst)
                #        else:
                #            print(f"    - Error: Cannot resolve Gripper clash for file '{original_name}'. Skipping.")
                #else:
                # Generic clash resolution using the full Instance ID
                for file_info in file_list:
                    original_name = file_info['original_name']
                    instance_id = file_info['instance_id']
                    new_name = f"{image_number}_{instance_id}.png"
                    src = os.path.join(root, original_name)
                    dst = os.path.join(root, new_name)
                    print(f"    - {'[DRY RUN] ' if dry_run else ''}Renaming '{original_name}' to '{new_name}'")
                    if not dry_run:
                        os.rename(src, dst)
            else:
                # No clash, perform simple rename
                file_info = file_list[0]
                original_name = file_info['original_name']
                new_name = f"{image_number}.png"
                src = os.path.join(root, original_name)
                dst = os.path.join(root, new_name)
                print(f"  - {'[DRY RUN] ' if dry_run else ''}Renaming '{original_name}' to '{new_name}'")
                if not dry_run:
                    os.rename(src, dst)
    
    print("\nFilename shortening complete.")
    if dry_run:
        print("This was a dry run. No files were actually renamed.")

def main():
    parser = argparse.ArgumentParser(description="Shorten image and mask filenames to just the image number.")
    parser.add_argument("target_dir", type=str, help="Path to the directory containing 'test' and 'ground_truth' folders.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a trial run with no files renamed.")
    
    args = parser.parse_args()
    
    target_path = Path(args.target_dir)

    if not target_path.is_dir():
        print(f"Error: Target directory '{args.target_dir}' not found.")
        return
    
    # Process 'test' directory
    test_dir = target_path / 'test'
    if test_dir.is_dir():
        shorten_filenames(str(test_dir), args.dry_run)
    else:
        print(f"Warning: 'test' directory not found in '{args.target_dir}'. Skipping.")
        
    # Process 'ground_truth' directory
    ground_truth_dir = target_path / 'gt'
    if ground_truth_dir.is_dir():
        shorten_filenames(str(ground_truth_dir), args.dry_run)
    else:
        print(f"Warning: 'ground_truth' directory not found in '{args.target_dir}'. Skipping.")

if __name__ == "__main__":
    main()