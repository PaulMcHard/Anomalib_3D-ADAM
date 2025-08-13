#!/usr/bin/env python3
import os
import argparse
import shutil
import re
from pathlib import Path

def camel_to_snake(name):
    """Convert CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def process_dataset(source_dir, output_dir, dry_run):
    """
    Transforms the dataset structure according to the specified rules.
    """
    print(f"Starting dataset transformation. Dry run mode: {dry_run}\n")

    # Iterate through categories at the top level
    for category in os.listdir(source_dir):
        category_path = Path(source_dir) / category
        if not category_path.is_dir():
            continue

        print(f"--- Processing category: {category} ---")
        output_category_path = Path(output_dir) / category
        if not dry_run:
            output_category_path.mkdir(parents=True, exist_ok=True)

        # Task 1: Consolidate 'train' images
        train_source_path = category_path / 'train'
        if train_source_path.is_dir():
            print(f"  - Consolidating 'train' images...")
            output_train_path = output_category_path / 'train'
            if not dry_run:
                output_train_path.mkdir(parents=True, exist_ok=True)

            for part_instance in os.listdir(train_source_path):
                part_instance_path = train_source_path / part_instance
                if not part_instance_path.is_dir():
                    continue

                images_source_path = part_instance_path / 'images'
                if images_source_path.is_dir():
                    for image_file in os.listdir(images_source_path):
                        src = images_source_path / image_file
                        dst = output_train_path / image_file
                        print(f"    - {'[DRY RUN] ' if dry_run else ''}Copying {src} to {dst}")
                        if not dry_run:
                            shutil.copy2(src, dst)

        # Tasks 2 & 3: Process 'test' and 'val' splits for images and masks
        for split in ['test', 'val']:
            split_source_path = category_path / split
            if not split_source_path.is_dir():
                continue

            print(f"  - Processing '{split}' split for images and masks...")

            for part_instance in os.listdir(split_source_path):
                part_instance_path = split_source_path / part_instance
                if not part_instance_path.is_dir():
                    continue

                # Extract defect type from part instance name
                try:
                    defect_camel_case = part_instance.split('_')[-1]
                    defect_snake_case = camel_to_snake(defect_camel_case)
                except IndexError:
                    print(f"    - Warning: Skipping '{part_instance}' due to unexpected naming format.")
                    continue

                # Task 2: Consolidate images into 'test' directory
                images_source_path = part_instance_path / 'images'
                if images_source_path.is_dir():
                    output_images_path = output_category_path / 'test' / defect_snake_case
                    if not dry_run:
                        output_images_path.mkdir(parents=True, exist_ok=True)

                    for image_file in os.listdir(images_source_path):
                        src = images_source_path / image_file
                        dst = output_images_path / image_file
                        print(f"    - {'[DRY RUN] ' if dry_run else ''}Copying image {src} to {dst}")
                        if not dry_run:
                            shutil.copy2(src, dst)

                # Task 3: Consolidate ground truth masks into 'ground_truth' directory
                masks_source_path = part_instance_path / 'defect_masks'
                if masks_source_path.is_dir():
                    output_masks_path = output_category_path / 'ground_truth' / defect_snake_case
                    if not dry_run:
                        output_masks_path.mkdir(parents=True, exist_ok=True)

                    for mask_file in os.listdir(masks_source_path):
                        src = masks_source_path / mask_file
                        dst = output_masks_path / mask_file
                        print(f"    - {'[DRY RUN] ' if dry_run else ''}Copying mask {src} to {dst}")
                        if not dry_run:
                            shutil.copy2(src, dst)

    print("\nDataset transformation complete.")
    if dry_run:
        print("This was a dry run. No files were actually moved or copied.")

def main():
    parser = argparse.ArgumentParser(description="Transform a dataset into a new directory structure.")
    parser.add_argument("source_dir", type=str, help="Path to the source dataset directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for the new dataset structure.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a trial run with no files moved or copied.")

    args = parser.parse_args()

    # Ensure source directory exists
    if not Path(args.source_dir).is_dir():
        print(f"Error: Source directory '{args.source_dir}' not found.")
        return

    process_dataset(args.source_dir, args.output_dir, args.dry_run)

if __name__ == "__main__":
    main()