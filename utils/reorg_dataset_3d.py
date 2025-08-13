#!/usr/bin/env python3
import os
import argparse
import shutil
import re
from pathlib import Path
import collections

def parse_filename(filename):
    """
    Parses a filename to extract part_name, instance_id, and image_number.
    Handles different naming conventions.
    """
    # Pattern for PartName_InstanceID_..._ImageNumber_...png
    match = re.match(r'(.+?)_([A-Z]\d*)_.+_(\d{3})_.*\.png', filename)
    if match:
        part_name = match.group(1)
        instance_id = match.group(2)
        image_number = match.group(3)
        return part_name, instance_id, image_number
    return None, None, None

def transform_dataset(source_dir, output_dir, dry_run):
    """
    Transforms the dataset structure from 3d-adam to mvtec3d format.
    """
    print(f"Starting dataset transformation. Dry run mode: {dry_run}\n")

    for category in os.listdir(source_dir):
        source_category_path = Path(source_dir) / category
        if not source_category_path.is_dir():
            continue

        print(f"--- Processing category: {category} ---")
        output_category_path = Path(output_dir) / category

        if not dry_run:
            output_category_path.mkdir(parents=True, exist_ok=True)

        for split in ['test', 'train']:
            source_split_path = source_category_path / split
            if not source_split_path.is_dir():
                continue

            print(f"  - Processing split: {split}...")

            # Define new destination directories
            output_rgb_path = output_category_path / split / 'rgb'
            output_xyz_path = output_category_path / split / 'xyz'
            output_gt_path = output_category_path / split / 'gt'

            if not dry_run:
                output_rgb_path.mkdir(parents=True, exist_ok=True)
                output_xyz_path.mkdir(parents=True, exist_ok=True)
                output_gt_path.mkdir(parents=True, exist_ok=True)

            # Dictionary to handle filename clashes
            clash_groups = collections.defaultdict(list)

            # First pass: Collect all files and their metadata
            for root, _, files in os.walk(source_split_path):
                for filename in files:
                    part_name, instance_id, image_number = parse_filename(filename)
                    if image_number:
                        file_info = {
                            'original_name': filename,
                            'part_name': part_name,
                            'instance_id': instance_id,
                            'image_number': image_number,
                            'source_path': Path(root) / filename
                        }
                        clash_groups[image_number].append(file_info)
            
            # Second pass: Process groups and move/rename files
            for image_number, file_list in clash_groups.items():
                if len(file_list) > 1:
                    print(f"  - CLASH DETECTED for image number '{image_number}'. Resolving with custom suffixes.")
                    
                    # Check for the specific Gripper case
                    is_gripper_clash = all('Gripper' in f['part_name'] for f in file_list) and \
                                       all('C' in f['instance_id'] for f in file_list)

                    for file_info in file_list:
                        original_name = file_info['original_name']
                        new_name = ''

                        if is_gripper_clash:
                            if 'Closed' in file_info['part_name']:
                                new_name = f"{image_number}_C1.png"
                            elif 'Open' in file_info['part_name']:
                                new_name = f"{image_number}_C2.png"
                            else:
                                new_name = f"{image_number}_{file_info['instance_id']}.png"
                        else:
                            new_name = f"{image_number}_{file_info['instance_id']}.png"

                        source_path = file_info['source_path']
                        
                        if new_name:
                            if original_name.endswith('.png'):
                                if 'defect' in original_name:
                                    destination_path = output_gt_path / new_name
                                else:
                                    destination_path = output_rgb_path / new_name
                            elif original_name.endswith(('.tiff', '.ply')):
                                destination_path = output_xyz_path / new_name.replace('.png', f'{Path(original_name).suffix}')

                            print(f"    - {'[DRY RUN] ' if dry_run else ''}Moving '{source_path}' to '{destination_path}'")
                            if not dry_run:
                                shutil.move(source_path, destination_path)
                        else:
                             print(f"    - Error: Cannot resolve clash for file '{original_name}'. Skipping.")

                else:
                    # No clash, perform simple rename and move
                    file_info = file_list[0]
                    original_name = file_info['original_name']
                    source_path = file_info['source_path']
                    image_number = file_info['image_number']
                    new_name = f"{image_number}.png"

                    if original_name.endswith('.png'):
                        if 'defect' in original_name:
                            destination_path = output_gt_path / new_name
                        else:
                            destination_path = output_rgb_path / new_name
                    elif original_name.endswith(('.tiff', '.ply')):
                         new_name = f"{image_number}{Path(original_name).suffix}"
                         destination_path = output_xyz_path / new_name
                    else:
                        continue
                    
                    print(f"  - {'[DRY RUN] ' if dry_run else ''}Moving '{source_path}' to '{destination_path}'")
                    if not dry_run:
                        shutil.move(source_path, destination_path)

            # Cleanup old directories
            if not dry_run:
                shutil.rmtree(source_split_path)

    print("\nDataset transformation complete.")
    if dry_run:
        print("This was a dry run. No files were actually moved or copied.")


def main():
    parser = argparse.ArgumentParser(description="Transform a dataset into a new directory structure.")
    parser.add_argument("source_dir", type=str, help="Path to the source dataset directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for the new dataset structure.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a trial run with no files moved or copied.")

    args = parser.parse_args()

    if not Path(args.source_dir).is_dir():
        print(f"Error: Source directory '{args.source_dir}' not found.")
        return
    
    transform_dataset(args.source_dir, args.output_dir, args.dry_run)

if __name__ == "__main__":
    main()