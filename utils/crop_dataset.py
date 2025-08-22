#!/usr/bin/env python3
"""
Crop all images in the reorganized Adam dataset to center 256x256 pixels.
Handles RGB (.png), depth (.tiff), and ground truth (.png) images.
"""

import cv2
import numpy as np
import tifffile
import os
import argparse
from pathlib import Path
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DatasetImageCropper:
    def __init__(self, crop_size=256, backup=True, verbose=False):
        """
        Initialize the image cropper.
        
        Args:
            crop_size: Size of the square crop (default: 256)
            backup: Create backup of original files
            verbose: Enable verbose logging
        """
        self.crop_size = crop_size
        self.backup = backup
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        self.stats = {
            'total_files': 0,
            'cropped_rgb': 0,
            'cropped_tiff': 0,
            'cropped_gt': 0,
            'already_correct_size': 0,
            'too_small': 0,
            'failed': 0,
            'backed_up': 0
        }
    
    def center_crop(self, image, target_size):
        """
        Crop image to center square of target_size.
        
        Args:
            image: Input image as numpy array
            target_size: Size of square crop
            
        Returns:
            numpy.ndarray: Cropped image, or None if image too small
        """
        height, width = image.shape[:2]
        
        # Check if image is large enough
        if height < target_size or width < target_size:
            logger.warning(f"Image too small ({height}x{width}) for {target_size}x{target_size} crop")
            return None
        
        # Calculate crop coordinates for center crop
        start_y = (height - target_size) // 2
        start_x = (width - target_size) // 2
        end_y = start_y + target_size
        end_x = start_x + target_size
        
        # Crop the image
        if len(image.shape) == 3:
            # Multi-channel image (RGB, RGBA, or multi-channel TIFF)
            cropped = image[start_y:end_y, start_x:end_x, :]
        else:
            # Single-channel image (grayscale GT or single-channel depth)
            cropped = image[start_y:end_y, start_x:end_x]
        
        logger.debug(f"Cropped from {height}x{width} to {cropped.shape}")
        return cropped
    
    def process_png_image(self, file_path):
        """
        Process a PNG image (RGB or ground truth).
        
        Args:
            file_path: Path to PNG file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read image with all channels preserved
            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                logger.error(f"Could not read PNG: {file_path}")
                return False
            
            original_shape = img.shape
            logger.debug(f"PNG image shape: {original_shape}")
            
            # Check if already correct size
            if (img.shape[0] == self.crop_size and img.shape[1] == self.crop_size):
                logger.debug(f"PNG already {self.crop_size}x{self.crop_size}: {file_path}")
                self.stats['already_correct_size'] += 1
                return True
            
            # Crop to center
            cropped_img = self.center_crop(img, self.crop_size)
            
            if cropped_img is None:
                self.stats['too_small'] += 1
                return False
            
            # Create backup if requested
            if self.backup:
                backup_path = file_path.with_suffix(f".backup{file_path.suffix}")
                if not backup_path.exists():
                    shutil.copy2(file_path, backup_path)
                    logger.debug(f"Created backup: {backup_path}")
                    self.stats['backed_up'] += 1
            
            # Save cropped image
            success = cv2.imwrite(str(file_path), cropped_img)
            
            if success:
                # Determine if RGB or GT based on channels
                if len(cropped_img.shape) == 2 or (len(cropped_img.shape) == 3 and cropped_img.shape[2] == 1):
                    self.stats['cropped_gt'] += 1
                    image_type = "GT"
                else:
                    self.stats['cropped_rgb'] += 1
                    image_type = "RGB"
                
                logger.debug(f"Cropped {image_type} PNG: {file_path}")
                return True
            else:
                logger.error(f"Failed to save PNG: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing PNG {file_path}: {e}")
            return False
    
    def process_tiff_image(self, file_path):
        """
        Process a TIFF image (depth data).
        
        Args:
            file_path: Path to TIFF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read TIFF using tifffile (preserves all data types and channels)
            img = tifffile.imread(str(file_path))
            
            original_shape = img.shape
            logger.debug(f"TIFF image shape: {original_shape}, dtype: {img.dtype}")
            
            # Check if already correct size
            if (img.shape[0] == self.crop_size and img.shape[1] == self.crop_size):
                logger.debug(f"TIFF already {self.crop_size}x{self.crop_size}: {file_path}")
                self.stats['already_correct_size'] += 1
                return True
            
            # Crop to center
            cropped_img = self.center_crop(img, self.crop_size)
            
            if cropped_img is None:
                self.stats['too_small'] += 1
                return False
            
            # Create backup if requested
            if self.backup:
                backup_path = file_path.with_suffix(f".backup{file_path.suffix}")
                if not backup_path.exists():
                    shutil.copy2(file_path, backup_path)
                    logger.debug(f"Created backup: {backup_path}")
                    self.stats['backed_up'] += 1
            
            # Save cropped TIFF with same properties as original
            tifffile.imwrite(
                str(file_path),
                cropped_img,
                compression='lzw',
                photometric='rgb' if len(cropped_img.shape) == 3 else 'minisblack'
            )
            
            self.stats['cropped_tiff'] += 1
            logger.debug(f"Cropped TIFF: {file_path}")
            return True
                
        except Exception as e:
            logger.error(f"Error processing TIFF {file_path}: {e}")
            return False
    
    def process_file(self, file_path):
        """
        Process a single image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.stats['total_files'] += 1
        
        file_extension = file_path.suffix.lower()
        
        logger.info(f"Processing: {file_path}")
        
        if file_extension == '.png':
            success = self.process_png_image(file_path)
        elif file_extension in ['.tiff', '.tif']:
            success = self.process_tiff_image(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return False
        
        if not success:
            self.stats['failed'] += 1
            logger.error(f"❌ Failed to process: {file_path}")
        else:
            logger.info(f"✅ Successfully processed: {file_path}")
        
        return success
    
    def process_dataset(self, dataset_root):
        """
        Process the entire reorganized Adam dataset.
        
        Args:
            dataset_root: Root directory of reorganized dataset
            
        Returns:
            dict: Processing statistics
        """
        dataset_path = Path(dataset_root)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
        
        logger.info(f"Processing dataset: {dataset_root}")
        logger.info(f"Cropping all images to {self.crop_size}x{self.crop_size}")
        
        # Find all image files in the dataset
        # Based on our understanding: object_type/split/defect_type/data_type/images
        image_files = []
        
        # Find PNG files (RGB and GT)
        png_files = list(dataset_path.rglob("*.png"))
        image_files.extend(png_files)
        
        # Find TIFF files (depth)
        tiff_files = list(dataset_path.rglob("*.tiff"))
        tiff_files.extend(list(dataset_path.rglob("*.tif")))
        image_files.extend(tiff_files)
        
        if not image_files:
            logger.warning(f"No image files found in {dataset_root}")
            return self.stats
        
        logger.info(f"Found {len(image_files)} image files to process")
        logger.info(f"  PNG files: {len(png_files)}")
        logger.info(f"  TIFF files: {len(tiff_files)}")
        
        # Group files by location for better logging
        rgb_files = []
        xyz_files = []
        gt_files = []
        
        for img_file in image_files:
            parent_dir = img_file.parent.name
            if parent_dir == 'rgb':
                rgb_files.append(img_file)
            elif parent_dir == 'xyz':
                xyz_files.append(img_file)
            elif parent_dir == 'gt':
                gt_files.append(img_file)
            else:
                # File not in expected location, but process anyway
                logger.debug(f"Unexpected location: {img_file}")
                if img_file.suffix.lower() == '.png':
                    rgb_files.append(img_file)
                else:
                    xyz_files.append(img_file)
        
        logger.info(f"Dataset structure analysis:")
        logger.info(f"  RGB directory files: {len(rgb_files)}")
        logger.info(f"  XYZ directory files: {len(xyz_files)}")
        logger.info(f"  GT directory files: {len(gt_files)}")
        
        # Process all files
        for img_file in image_files:
            try:
                self.process_file(img_file)
            except Exception as e:
                logger.error(f"Unexpected error processing {img_file}: {e}")
                self.stats['failed'] += 1
        
        return self.stats
    
    def print_summary(self):
        """Print processing summary."""
        logger.info("\n" + "="*60)
        logger.info("CROPPING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"RGB images cropped: {self.stats['cropped_rgb']}")
        logger.info(f"TIFF images cropped: {self.stats['cropped_tiff']}")
        logger.info(f"GT images cropped: {self.stats['cropped_gt']}")
        logger.info(f"Already correct size: {self.stats['already_correct_size']}")
        logger.info(f"Too small to crop: {self.stats['too_small']}")
        logger.info(f"Failed to process: {self.stats['failed']}")
        logger.info(f"Backup files created: {self.stats['backed_up']}")
        
        total_cropped = (self.stats['cropped_rgb'] + 
                        self.stats['cropped_tiff'] + 
                        self.stats['cropped_gt'])
        
        if self.stats['failed'] > 0:
            logger.warning(f"⚠️  {self.stats['failed']} files failed to process!")
        
        if self.stats['too_small'] > 0:
            logger.warning(f"⚠️  {self.stats['too_small']} files were too small to crop!")
        
        if total_cropped > 0:
            logger.info(f"✅ Successfully cropped {total_cropped} images to {self.crop_size}x{self.crop_size}")
        
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Crop all images in reorganized Adam dataset to center 256x256",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop entire dataset with backups
  python crop_dataset_images.py /path/to/reorganized_adam
  
  # Crop without backups (faster, but riskier)
  python crop_dataset_images.py /path/to/reorganized_adam --no-backup
  
  # Custom crop size
  python crop_dataset_images.py /path/to/reorganized_adam --crop-size 512
  
  # Verbose output
  python crop_dataset_images.py /path/to/reorganized_adam --verbose

This script will process:
- RGB images: /object/split/defect/rgb/*.png
- Depth images: /object/split/defect/xyz/*.tiff
- Ground truth: /object/split/defect/gt/*.png
        """
    )
    
    parser.add_argument('dataset_root', 
                       help='Root directory of the reorganized Adam dataset')
    parser.add_argument('--crop-size', type=int, default=256,
                       help='Size of square crop (default: 256)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup files (faster but riskier)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Create cropper
        cropper = DatasetImageCropper(
            crop_size=args.crop_size,
            backup=not args.no_backup,
            verbose=args.verbose
        )
        
        # Process dataset
        stats = cropper.process_dataset(args.dataset_root)
        
        # Print summary
        cropper.print_summary()
        
        # Exit with appropriate code
        if stats['failed'] > 0:
            logger.error("Some files failed to process!")
            exit(1)
        else:
            logger.info("All files processed successfully!")
            exit(0)
            
    except Exception as e:
        logger.error(f"Cropping failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()