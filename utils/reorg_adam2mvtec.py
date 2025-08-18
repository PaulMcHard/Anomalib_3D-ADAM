#!/usr/bin/env python3
"""
Script to reorganize Adam dataset structure to match MVTec 3D AD structure.

Usage:
    python reorganize_dataset.py /path/to/adam/dataset /path/to/output/mvtec --dry-run --verbose --create-missing-masks
"""

import os
import shutil
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
import logging
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DatasetReorganizer:
    def __init__(self, source_dir, output_dir, dry_run=False, verbose=False, create_missing_masks=False):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.verbose = verbose
        self.create_missing_masks = create_missing_masks
        
        # Set logging level based on verbose flag
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Valid defect types that exist in MVTec structure
        self.valid_defects = {
            'good', 'hole', 'cut', 'crack', 'bulge', 'contamination', 
            'over_extrusion', 'under_extrusion', 'combined', 'bent', 
            'thread', 'color', 'open'
        }
        
        # Valid data types (ignore binary_masks, machine_elements)
        self.valid_data_types = {'rgb', 'xyz', 'ground_truth'}
        
        # Split ratios
        self.train_ratio = 0.8
        self.val_ratio = 0.2
        self.test_ratio = 0.0
        
        # Statistics
        self.stats = {
            'objects_processed': 0,
            'files_copied': 0,
            'masks_created': 0,
            'defects_found': defaultdict(int),
            'splits_created': defaultdict(lambda: defaultdict(int))
        }
    
    def extract_part_id(self, filename):
        """Extract the part ID from filename to group related files."""
        # Remove extension
        base_name = Path(filename).stem
        
        # Split by underscore to find the suffix (A, B, C, etc.)
        if '_' in base_name:
            parts = base_name.split('_')
            # The suffix is everything after the first underscore
            # e.g., "000_A" -> suffix is "A", "000_A_extra" -> suffix is "A_extra"
            suffix = '_'.join(parts[1:])
            return suffix
        
        # If no underscore, this represents the "no suffix" group
        return ""
    
    def get_part_groups(self, file_list):
        """Group files by part ID to avoid data leakage."""
        part_groups = defaultdict(list)
        
        for file_path in file_list:
            part_id = self.extract_part_id(file_path.name)
            part_groups[part_id].append(file_path)
        
        return dict(part_groups)
    
    def split_good_samples(self, good_files):
        """Split good samples into train/val/test while keeping part groups together."""
        if not good_files:
            return {}, {}, {}
        
        # Group files by part ID
        part_groups = self.get_part_groups(good_files)
        part_ids = list(part_groups.keys())
        
        # Shuffle part IDs for random split
        random.shuffle(part_ids)
        
        # Calculate split indices
        total_parts = len(part_ids)
        train_end = int(total_parts * self.train_ratio)
        val_end = train_end + int(total_parts * self.val_ratio)
        
        # Split part IDs
        train_parts = part_ids[:train_end]
        val_parts = part_ids[train_end:]
        #test_parts = part_ids[val_end:]
        
        # Group files by split
        train_files = []
        val_files = []
        test_files = []
        
        for part_id in train_parts:
            train_files.extend(part_groups[part_id])
        
        for part_id in val_parts:
            val_files.extend(part_groups[part_id])
        
        #for part_id in test_parts:
        #    test_files.extend(part_groups[part_id])
        
        logger.debug(f"Split {total_parts} parts: {len(train_parts)} train, {len(val_parts)} val, {len(test_files)} test")
        
        return train_files, val_files , test_files
    
    def find_corresponding_file(self, file_list, base_filename, extensions):
        """Find a file with the same base name but different extension."""
        for file_path in file_list:
            file_stem = file_path.stem
            file_ext = file_path.suffix.lower()
            
            # Check if the base filename matches and extension is in allowed list
            if file_stem == base_filename and file_ext in extensions:
                return file_path
        
        return None
    
    def get_image_dimensions(self, image_path):
        """Get the dimensions (width, height) of an image file."""
        try:
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            logger.error(f"Failed to get dimensions for {image_path}: {e}")
            return None
    
    def create_empty_mask(self, width, height, output_path):
        """Create an empty (all black) single-channel PNG mask."""
        try:
            if self.dry_run:
                logger.debug(f"[DRY RUN] Would create empty mask: {output_path} ({width}x{height})")
                self.stats['masks_created'] += 1
                return True
            
            # Create empty mask (all zeros, single channel)
            mask_array = np.zeros((height, width), dtype=np.uint8)
            
            # Convert to PIL Image and save as PNG
            mask_image = Image.fromarray(mask_array, mode='L')  # 'L' for grayscale
            
            # Create destination directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the mask
            mask_image.save(output_path)
            
            logger.debug(f"Created empty mask: {output_path} ({width}x{height})")
            self.stats['masks_created'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to create empty mask {output_path}: {e}")
            return False
    
    def copy_file(self, src_path, dst_path):
        """Copy a single file with proper error handling."""
        try:
            if self.dry_run:
                logger.debug(f"[DRY RUN] Would copy: {src_path} -> {dst_path}")
            else:
                # Create destination directory if it doesn't exist
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                logger.debug(f"Copied: {src_path} -> {dst_path}")
            
            self.stats['files_copied'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
            return False
    
    def process_object_type(self, object_path):
        """Process a single object type directory."""
        object_name = object_path.name
        logger.info(f"Processing object: {object_name}")
        
        # Collect all files by defect type and data type
        defect_data = defaultdict(lambda: defaultdict(list))
        
        for defect_dir in object_path.iterdir():
            if not defect_dir.is_dir():
                continue
            
            defect_name = defect_dir.name
            
            # Skip invalid defect types
            if defect_name not in self.valid_defects:
                logger.debug(f"Skipping invalid defect: {defect_name}")
                continue
            
            self.stats['defects_found'][defect_name] += 1
            
            for data_type_dir in defect_dir.iterdir():
                if not data_type_dir.is_dir():
                    continue
                
                data_type = data_type_dir.name
                
                # Skip invalid data types
                if data_type not in self.valid_data_types:
                    logger.debug(f"Skipping invalid data type: {data_type}")
                    continue
                
                # Collect all files in this directory
                files = list(data_type_dir.glob('*'))
                files = [f for f in files if f.is_file()]
                
                defect_data[defect_name][data_type].extend(files)
                
                logger.debug(f"Found {len(files)} files in {object_name}/{defect_name}/{data_type}")
        
        # Process good samples first (need to split)
        if 'good' in defect_data:
            self.process_good_samples(object_name, defect_data['good'])
        
        # Process all defective samples (go to test only)
        for defect_name, data_types in defect_data.items():
            if defect_name != 'good':
                self.process_defective_samples(object_name, defect_name, data_types)
        
        self.stats['objects_processed'] += 1
    
    def process_good_samples(self, object_name, good_data):
        """Process good samples and split into train/val/test."""
        logger.info(f"Processing good samples for {object_name}")
        
        # Use RGB as the master directory for splitting decisions
        if 'rgb' not in good_data or not good_data['rgb']:
            logger.warning(f"No RGB files found for {object_name}/good - skipping")
            return
        
        # Split RGB files while keeping part groups together
        train_rgb, val_rgb, test_rgb = self.split_good_samples(good_data['rgb'])
        
        # Create filename mappings for each split
        splits_data = {
            'train': train_rgb,
            'validation': val_rgb,
            'test': test_rgb
        }
        
        # For each split, copy corresponding files from all data types
        for split_name, rgb_files in splits_data.items():
            logger.info(f"Processing {split_name} split with {len(rgb_files)} RGB files")
            
            for rgb_file in rgb_files:
                # Copy the RGB file
                dst_rgb_path = (self.output_dir / object_name / split_name / 'good' / 
                               'rgb' / rgb_file.name)
                if self.copy_file(rgb_file, dst_rgb_path):
                    self.stats['splits_created'][split_name]['good'] += 1
                
                # Find and copy corresponding files from other data types
                base_filename = rgb_file.stem  # filename without extension
                
                # Copy corresponding XYZ file (same name but .tiff extension)
                if 'xyz' in good_data:
                    xyz_file = self.find_corresponding_file(good_data['xyz'], base_filename, ['.tiff', '.tif'])
                    if xyz_file:
                        dst_xyz_path = (self.output_dir / object_name / split_name / 'good' / 
                                       'xyz' / xyz_file.name)
                        self.copy_file(xyz_file, dst_xyz_path)
                    else:
                        logger.warning(f"No corresponding XYZ file found for {rgb_file.name}")
                
                # Copy corresponding ground_truth file OR create empty mask
                if 'ground_truth' in good_data:
                    ground_truth_file = self.find_corresponding_file(good_data['ground_truth'], base_filename, ['.png', '.tiff', '.tif'])
                    if ground_truth_file:
                        dst_ground_truth_path = (self.output_dir / object_name / split_name / 'good' / 
                                      'ground_truth' / ground_truth_file.name)
                        self.copy_file(ground_truth_file, dst_ground_truth_path)
                    else:
                        # No corresponding ground truth file found
                        if self.create_missing_masks:
                            # Get RGB image dimensions
                            rgb_dimensions = self.get_image_dimensions(rgb_file)
                            if rgb_dimensions:
                                width, height = rgb_dimensions
                                # Create empty mask with same base filename but .png extension
                                mask_filename = f"{base_filename}.png"
                                dst_mask_path = (self.output_dir / object_name / split_name / 'good' / 
                                               'ground_truth' / mask_filename)
                                self.create_empty_mask(width, height, dst_mask_path)
                                logger.info(f"Created empty mask for {rgb_file.name}")
                            else:
                                logger.error(f"Could not get dimensions for {rgb_file.name} - cannot create mask")
                        else:
                            logger.warning(f"No corresponding ground_truth file found for {rgb_file.name}")
        
        logger.info(f"Split good samples: {len(train_rgb)} train, {len(val_rgb)} val, {len(test_rgb)} test")
    
    def process_defective_samples(self, object_name, defect_name, defect_data):
        """Process defective samples (all go to test split)."""
        logger.info(f"Processing {defect_name} samples for {object_name}")
        
        # Use RGB as the master directory for file selection
        if 'rgb' not in defect_data or not defect_data['rgb']:
            logger.warning(f"No RGB files found for {object_name}/{defect_name} - skipping")
            return
        
        rgb_files = defect_data['rgb']
        logger.info(f"Processing {len(rgb_files)} RGB files for {defect_name}")
        
        for rgb_file in rgb_files:
            # Copy the RGB file
            dst_rgb_path = (self.output_dir / object_name / 'test' / defect_name / 
                           'rgb' / rgb_file.name)
            if self.copy_file(rgb_file, dst_rgb_path):
                self.stats['splits_created']['test'][defect_name] += 1
            
            # Find and copy corresponding files from other data types
            base_filename = rgb_file.stem  # filename without extension
            
            # Copy corresponding XYZ file (same name but .tiff extension)
            if 'xyz' in defect_data:
                xyz_file = self.find_corresponding_file(defect_data['xyz'], base_filename, ['.tiff', '.tif'])
                if xyz_file:
                    dst_xyz_path = (self.output_dir / object_name / 'test' / defect_name / 
                                   'xyz' / xyz_file.name)
                    self.copy_file(xyz_file, dst_xyz_path)
                else:
                    logger.warning(f"No corresponding XYZ file found for {rgb_file.name}")
            
            # Copy corresponding ground_truth file OR create empty mask
            if 'ground_truth' in defect_data:
                ground_truth_file = self.find_corresponding_file(defect_data['ground_truth'], base_filename, ['.png', '.tiff', '.tif'])
                if ground_truth_file:
                    dst_ground_truth_path = (self.output_dir / object_name / 'test' / defect_name / 
                                  'ground_truth' / ground_truth_file.name)
                    self.copy_file(ground_truth_file, dst_ground_truth_path)
                else:
                    # No corresponding ground truth file found
                    if self.create_missing_masks:
                        # Get RGB image dimensions
                        rgb_dimensions = self.get_image_dimensions(rgb_file)
                        if rgb_dimensions:
                            width, height = rgb_dimensions
                            # Create empty mask with same base filename but .png extension
                            mask_filename = f"{base_filename}.png"
                            dst_mask_path = (self.output_dir / object_name / 'test' / defect_name / 
                                           'ground_truth' / mask_filename)
                            self.create_empty_mask(width, height, dst_mask_path)
                            logger.info(f"Created empty mask for {rgb_file.name}")
                        else:
                            logger.error(f"Could not get dimensions for {rgb_file.name} - cannot create mask")
                    else:
                        logger.warning(f"No corresponding ground_truth file found for {rgb_file.name}")
        
        total_files = len(rgb_files) * len([dt for dt in ['rgb', 'xyz', 'ground_truth'] if dt in defect_data])
        logger.info(f"Copied up to {total_files} {defect_name} files to test split")
    
    def create_calibration_dirs(self):
        """Create empty calibration directories for each object type."""
        for object_path in self.source_dir.iterdir():
            if not object_path.is_dir():
                continue
            
            object_name = object_path.name
            calibration_dir = self.output_dir / object_name / 'calibration'
            
            if self.dry_run:
                logger.debug(f"[DRY RUN] Would create calibration dir: {calibration_dir}")
            else:
                calibration_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created calibration directory: {calibration_dir}")
    
    def print_statistics(self):
        """Print reorganization statistics."""
        logger.info("\n" + "="*50)
        logger.info("REORGANIZATION STATISTICS")
        logger.info("="*50)
        logger.info(f"Objects processed: {self.stats['objects_processed']}")
        logger.info(f"Files copied: {self.stats['files_copied']}")
        
        if self.create_missing_masks:
            logger.info(f"Empty masks created: {self.stats['masks_created']}")
        
        logger.info("\nDefect types found:")
        for defect, count in sorted(self.stats['defects_found'].items()):
            logger.info(f"  {defect}: {count} directories")
        
        logger.info("\nFiles per split:")
        for split_name, defects in self.stats['splits_created'].items():
            logger.info(f"  {split_name}:")
            for defect, count in sorted(defects.items()):
                logger.info(f"    {defect}: {count} files")
        
        if self.dry_run:
            logger.info("\n[DRY RUN] No files were actually copied or created.")
    
    def reorganize(self):
        """Main reorganization function."""
        logger.info(f"Starting dataset reorganization...")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Create missing masks: {self.create_missing_masks}")
        
        # Set random seed for reproducible splits
        random.seed(42)
        
        # Validate source directory
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        
        # Create output directory
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each object type directory
        for object_path in self.source_dir.iterdir():
            if not object_path.is_dir():
                continue
            
            try:
                self.process_object_type(object_path)
            except Exception as e:
                logger.error(f"Failed to process {object_path.name}: {e}")
                continue
        
        # Create calibration directories
        self.create_calibration_dirs()
        
        # Print statistics
        self.print_statistics()
        
        logger.info("Reorganization completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize Adam dataset to match MVTec 3D AD structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reorganize_dataset.py /data/adam /data/mvtec_format
  python reorganize_dataset.py /data/adam /data/mvtec_format --dry-run --verbose
  python reorganize_dataset.py /data/adam /data/mvtec_format --create-missing-masks
        """
    )
    
    parser.add_argument('source_dir', 
                       help='Path to the source Adam dataset directory')
    parser.add_argument('output_dir', 
                       help='Path to the output directory for reorganized dataset')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually copying files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--create-missing-masks', action='store_true',
                       help='Create empty black PNG masks when ground truth files are missing')
    
    args = parser.parse_args()
    
    try:
        reorganizer = DatasetReorganizer(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            verbose=args.verbose,
            create_missing_masks=args.create_missing_masks
        )
        
        reorganizer.reorganize()
        
    except Exception as e:
        logger.error(f"Reorganization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())