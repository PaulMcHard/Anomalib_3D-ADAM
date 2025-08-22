#!/usr/bin/env python3
"""
Downsample square images from 1024x1024 or 512x512 to 256x256.
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

class ImageDownsampler:
    def __init__(self, target_size=256, backup=True, verbose=False):
        """
        Initialize the image downsampler.
        
        Args:
            target_size: Target size for square images (default: 256)
            backup: Create backup of original files
            verbose: Enable verbose logging
        """
        self.target_size = target_size
        self.backup = backup
        
        # Supported source sizes
        self.supported_sizes = [1024, 512]
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        self.stats = {
            'total_files': 0,
            'downsampled_1024': 0,
            'downsampled_512': 0,
            'already_target_size': 0,
            'unsupported_size': 0,
            'non_square': 0,
            'failed': 0,
            'backed_up': 0
        }
    
    def downsample_image(self, image, current_size, target_size, interpolation_method):
        """
        Downsample image to target size.
        
        Args:
            image: Input image as numpy array
            current_size: Current size (assumed square)
            target_size: Target size
            interpolation_method: OpenCV interpolation method
            
        Returns:
            numpy.ndarray: Downsampled image
        """
        if len(image.shape) == 3:
            # Multi-channel image
            height, width, channels = image.shape
        else:
            # Single-channel image
            height, width = image.shape
            channels = 1
        
        logger.debug(f"Downsampling from {current_size}x{current_size} to {target_size}x{target_size}")
        
        # Resize image
        downsampled = cv2.resize(
            image, 
            (target_size, target_size), 
            interpolation=interpolation_method
        )
        
        logger.debug(f"Downsampled shape: {downsampled.shape}")
        return downsampled
    
    def get_interpolation_method(self, image_type, is_ground_truth=False):
        """
        Get appropriate interpolation method based on image type.
        
        Args:
            image_type: 'png' or 'tiff'
            is_ground_truth: Whether this is a ground truth mask
            
        Returns:
            OpenCV interpolation constant
        """
        if is_ground_truth:
            # Ground truth masks should use nearest neighbor to preserve labels
            return cv2.INTER_NEAREST
        else:
            # RGB and depth images can use higher quality interpolation
            return cv2.INTER_AREA  # Good for downsampling
    
    def is_ground_truth_image(self, file_path):
        """
        Determine if image is likely a ground truth mask.
        
        Args:
            file_path: Path to image file
            
        Returns:
            bool: True if likely ground truth
        """
        # Check if in 'gt' directory
        return file_path.parent.name == 'gt'
    
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
            
            height, width = img.shape[:2]
            logger.debug(f"PNG image shape: {img.shape}")
            
            # Check if square
            if height != width:
                logger.warning(f"Non-square image {height}x{width}: {file_path}")
                self.stats['non_square'] += 1
                return False
            
            current_size = height
            
            # Check if already target size
            if current_size == self.target_size:
                logger.debug(f"PNG already {self.target_size}x{self.target_size}: {file_path}")
                self.stats['already_target_size'] += 1
                return True
            
            # Check if supported size
            if current_size not in self.supported_sizes:
                logger.warning(f"Unsupported size {current_size}x{current_size}: {file_path}")
                self.stats['unsupported_size'] += 1
                return False
            
            # Determine if ground truth
            is_gt = self.is_ground_truth_image(file_path)
            
            # Get appropriate interpolation method
            interp_method = self.get_interpolation_method('png', is_gt)
            
            # Downsample image
            downsampled_img = self.downsample_image(
                img, current_size, self.target_size, interp_method
            )
            
            # Create backup if requested
            if self.backup:
                backup_path = file_path.with_suffix(f".backup{file_path.suffix}")
                if not backup_path.exists():
                    shutil.copy2(file_path, backup_path)
                    logger.debug(f"Created backup: {backup_path}")
                    self.stats['backed_up'] += 1
            
            # Save downsampled image
            success = cv2.imwrite(str(file_path), downsampled_img)
            
            if success:
                # Update statistics
                if current_size == 1024:
                    self.stats['downsampled_1024'] += 1
                elif current_size == 512:
                    self.stats['downsampled_512'] += 1
                
                image_type = "GT" if is_gt else "RGB"
                logger.debug(f"Downsampled {image_type} PNG from {current_size} to {self.target_size}: {file_path}")
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
            
            height, width = img.shape[:2]
            logger.debug(f"TIFF image shape: {img.shape}, dtype: {img.dtype}")
            
            # Check if square
            if height != width:
                logger.warning(f"Non-square image {height}x{width}: {file_path}")
                self.stats['non_square'] += 1
                return False
            
            current_size = height
            
            # Check if already target size
            if current_size == self.target_size:
                logger.debug(f"TIFF already {self.target_size}x{self.target_size}: {file_path}")
                self.stats['already_target_size'] += 1
                return True
            
            # Check if supported size
            if current_size not in self.supported_sizes:
                logger.warning(f"Unsupported size {current_size}x{current_size}: {file_path}")
                self.stats['unsupported_size'] += 1
                return False
            
            # Get interpolation method (depth images use INTER_AREA)
            interp_method = self.get_interpolation_method('tiff', False)
            
            # Downsample image
            downsampled_img = self.downsample_image(
                img, current_size, self.target_size, interp_method
            )
            
            # Create backup if requested
            if self.backup:
                backup_path = file_path.with_suffix(f".backup{file_path.suffix}")
                if not backup_path.exists():
                    shutil.copy2(file_path, backup_path)
                    logger.debug(f"Created backup: {backup_path}")
                    self.stats['backed_up'] += 1
            
            # Save downsampled TIFF with same properties as original
            tifffile.imwrite(
                str(file_path),
                downsampled_img.astype(img.dtype),  # Preserve original data type
                compression='lzw',
                photometric='rgb' if len(downsampled_img.shape) == 3 else 'minisblack'
            )
            
            # Update statistics
            if current_size == 1024:
                self.stats['downsampled_1024'] += 1
            elif current_size == 512:
                self.stats['downsampled_512'] += 1
            
            logger.debug(f"Downsampled TIFF from {current_size} to {self.target_size}: {file_path}")
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
    
    def analyze_dataset_sizes(self, dataset_root):
        """
        Analyze the current sizes of images in the dataset.
        
        Args:
            dataset_root: Root directory of dataset
        """
        dataset_path = Path(dataset_root)
        
        # Find all image files
        png_files = list(dataset_path.rglob("*.png"))
        tiff_files = list(dataset_path.rglob("*.tiff")) + list(dataset_path.rglob("*.tif"))
        
        size_counts = {}
        total_files = len(png_files) + len(tiff_files)
        
        logger.info(f"Analyzing {total_files} images in dataset...")
        
        all_files = png_files + tiff_files
        
        for img_file in all_files:
            try:
                if img_file.suffix.lower() == '.png':
                    img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        height, width = img.shape[:2]
                else:
                    img = tifffile.imread(str(img_file))
                    height, width = img.shape[:2]
                
                if height == width:  # Square images only
                    size = height
                    if size not in size_counts:
                        size_counts[size] = 0
                    size_counts[size] += 1
                        
            except Exception as e:
                logger.debug(f"Could not analyze {img_file}: {e}")
        
        logger.info("\nDataset Size Analysis:")
        logger.info("=" * 30)
        
        total_square = sum(size_counts.values())
        
        for size in sorted(size_counts.keys(), reverse=True):
            count = size_counts[size]
            percentage = (count / total_square) * 100 if total_square > 0 else 0
            
            if size in self.supported_sizes:
                status = f"→ Will downsample to {self.target_size}x{self.target_size}"
            elif size == self.target_size:
                status = "✅ Already target size"
            else:
                status = "⚠️  Unsupported size"
            
            logger.info(f"{size}x{size}: {count:4d} images ({percentage:5.1f}%) {status}")
        
        non_square = total_files - total_square
        if non_square > 0:
            logger.info(f"Non-square: {non_square:4d} images (will be skipped)")
        
        # Calculate what will be processed
        will_process = sum(size_counts.get(size, 0) for size in self.supported_sizes)
        logger.info(f"\nWill downsample: {will_process} images")
        logger.info(f"Already correct: {size_counts.get(self.target_size, 0)} images")
        logger.info(f"Will skip: {total_files - will_process - size_counts.get(self.target_size, 0)} images")
    
    def process_dataset(self, dataset_root, analyze_first=True):
        """
        Process the entire dataset.
        
        Args:
            dataset_root: Root directory of dataset
            analyze_first: Whether to analyze sizes first
            
        Returns:
            dict: Processing statistics
        """
        dataset_path = Path(dataset_root)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
        
        if analyze_first:
            self.analyze_dataset_sizes(dataset_root)
            
            response = input(f"\nProceed with downsampling to {self.target_size}x{self.target_size}? (y/N): ")
            if response.lower() != 'y':
                logger.info("Downsampling cancelled.")
                return self.stats
        
        logger.info(f"\nProcessing dataset: {dataset_root}")
        logger.info(f"Downsampling {self.supported_sizes} → {self.target_size}x{self.target_size}")
        
        # Find all image files
        image_files = []
        png_files = list(dataset_path.rglob("*.png"))
        tiff_files = list(dataset_path.rglob("*.tiff")) + list(dataset_path.rglob("*.tif"))
        image_files = png_files + tiff_files
        
        if not image_files:
            logger.warning(f"No image files found in {dataset_root}")
            return self.stats
        
        logger.info(f"Found {len(image_files)} image files to check")
        
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
        logger.info("DOWNSAMPLING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"1024→{self.target_size} downsampled: {self.stats['downsampled_1024']}")
        logger.info(f"512→{self.target_size} downsampled: {self.stats['downsampled_512']}")
        logger.info(f"Already target size: {self.stats['already_target_size']}")
        logger.info(f"Unsupported sizes: {self.stats['unsupported_size']}")
        logger.info(f"Non-square images: {self.stats['non_square']}")
        logger.info(f"Failed to process: {self.stats['failed']}")
        logger.info(f"Backup files created: {self.stats['backed_up']}")
        
        total_downsampled = self.stats['downsampled_1024'] + self.stats['downsampled_512']
        
        if self.stats['failed'] > 0:
            logger.warning(f"⚠️  {self.stats['failed']} files failed to process!")
        
        if total_downsampled > 0:
            logger.info(f"✅ Successfully downsampled {total_downsampled} images to {self.target_size}x{self.target_size}")
        
        logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Downsample square images from 1024x1024 or 512x512 to 256x256",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze first, then downsample with backups
  python downsample_images.py /path/to/dataset
  
  # Downsample without analysis or backups
  python downsample_images.py /path/to/dataset --no-analyze --no-backup
  
  # Custom target size
  python downsample_images.py /path/to/dataset --target-size 128
  
  # Verbose output
  python downsample_images.py /path/to/dataset --verbose

Supported operations:
- 1024x1024 → 256x256 (4x downsampling)
- 512x512 → 256x256 (2x downsampling)
- Ground truth masks use nearest neighbor interpolation
- RGB/depth images use area interpolation for quality
        """
    )
    
    parser.add_argument('dataset_root', 
                       help='Root directory of the dataset')
    parser.add_argument('--target-size', type=int, default=256,
                       help='Target size for downsampling (default: 256)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup files')
    parser.add_argument('--no-analyze', action='store_true',
                       help='Skip analysis and confirmation')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Create downsampler
        downsampler = ImageDownsampler(
            target_size=args.target_size,
            backup=not args.no_backup,
            verbose=args.verbose
        )
        
        # Process dataset
        stats = downsampler.process_dataset(
            args.dataset_root,
            analyze_first=not args.no_analyze
        )
        
        # Print summary
        downsampler.print_summary()
        
        # Exit with appropriate code
        if stats['failed'] > 0:
            logger.error("Some files failed to process!")
            exit(1)
        else:
            logger.info("Processing completed successfully!")
            exit(0)
            
    except Exception as e:
        logger.error(f"Downsampling failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()