#!/usr/bin/env python3
"""
Convert RGBA images to RGB by removing the alpha channel.
This fixes the "Image must have 3 channels, got 4" error in anomalib.
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RGBAToRGBConverter:
    def __init__(self, backup=True, verbose=False):
        """
        Initialize the converter.
        
        Args:
            backup: Create backup of original files
            verbose: Enable verbose logging
        """
        self.backup = backup
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        self.stats = {
            'total_files': 0,
            'converted': 0,
            'already_rgb': 0,
            'single_channel_skipped': 0,
            'failed': 0,
            'backed_up': 0
        }
    
    def check_image_channels(self, image_path):
        """
        Check how many channels an image has.
        
        Args:
            image_path: Path to image file
            
        Returns:
            int: Number of channels, or None if error
        """
        try:
            # Read image without any conversion
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            
            # Check dimensions
            if len(img.shape) == 2:
                return 1  # Grayscale
            elif len(img.shape) == 3:
                return img.shape[2]  # Number of channels
            else:
                logger.error(f"Unexpected image shape: {img.shape}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking {image_path}: {e}")
            return None
    
    def convert_rgba_to_rgb(self, input_path, output_path=None):
        """
        Convert RGBA image to RGB.
        
        Args:
            input_path: Path to input RGBA image
            output_path: Path for output RGB image (None to overwrite)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read image with all channels
            img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                logger.error(f"Could not read image: {input_path}")
                return False
            
            logger.debug(f"Original image shape: {img.shape}, dtype: {img.dtype}")
            
            # Check if conversion is needed
            if len(img.shape) == 3 and img.shape[2] == 4:
                # RGBA -> RGB: Drop alpha channel
                img_rgb = img[:, :, :3]
                logger.debug(f"Converted RGBA to RGB: {img_rgb.shape}")
                
            elif len(img.shape) == 3 and img.shape[2] == 3:
                # Already RGB
                img_rgb = img
                logger.debug("Image already RGB")
                self.stats['already_rgb'] += 1
                return True
                
            elif len(img.shape) == 2:
                # Single-channel (grayscale) - DO NOT CONVERT
                # These are likely ground truth labels
                logger.debug("Single-channel image (likely GT label) - skipping conversion")
                self.stats['single_channel_skipped'] += 1
                return True
                
            else:
                logger.error(f"Unsupported image format: {img.shape}")
                return False
            
            # Determine output path
            if output_path is None:
                output_path = input_path
                
                # Create backup if requested and we're overwriting
                if self.backup:
                    backup_path = input_path.with_suffix(f".backup{input_path.suffix}")
                    if not backup_path.exists():
                        shutil.copy2(input_path, backup_path)
                        logger.debug(f"Created backup: {backup_path}")
                        self.stats['backed_up'] += 1
            
            # Save RGB image
            success = cv2.imwrite(str(output_path), img_rgb)
            
            if success:
                logger.debug(f"Saved RGB image: {output_path}")
                self.stats['converted'] += 1
                return True
            else:
                logger.error(f"Failed to save: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error converting {input_path}: {e}")
            return False
    
    def process_file(self, file_path, output_dir=None):
        """
        Process a single image file.
        
        Args:
            file_path: Path to image file
            output_dir: Output directory (None to overwrite in place)
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.stats['total_files'] += 1
        
        # Check if file needs conversion
        channels = self.check_image_channels(file_path)
        
        if channels is None:
            self.stats['failed'] += 1
            return False
        
        if channels == 3:
            logger.debug(f"File already has 3 channels: {file_path}")
            self.stats['already_rgb'] += 1
            return True
        
        if channels == 1:
            logger.debug(f"Single-channel image (GT label) - skipping: {file_path}")
            self.stats['single_channel_skipped'] += 1
            return True
        
        logger.info(f"Converting {channels}-channel image: {file_path}")
        
        # Determine output path
        if output_dir:
            output_path = Path(output_dir) / file_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = None  # Will overwrite in place
        
        # Convert the file
        if self.convert_rgba_to_rgb(file_path, output_path):
            logger.info(f"✅ Successfully converted: {file_path}")
            return True
        else:
            logger.error(f"❌ Failed to convert: {file_path}")
            self.stats['failed'] += 1
            return False
    
    def process_directory(self, input_dir, output_dir=None, recursive=True, 
                         file_extensions=None):
        """
        Process all image files in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory (None to overwrite in place)
            recursive: Process subdirectories recursively
            file_extensions: List of extensions to process
            
        Returns:
            dict: Processing statistics
        """
        if file_extensions is None:
            file_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
        
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Find all image files
        image_files = []
        for ext in file_extensions:
            if recursive:
                image_files.extend(input_path.rglob(f"*{ext}"))
                image_files.extend(input_path.rglob(f"*{ext.upper()}"))
            else:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return self.stats
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        # Process each file
        for img_file in image_files:
            try:
                # Determine output file path
                if output_dir and recursive:
                    # Preserve directory structure
                    rel_path = img_file.relative_to(input_path)
                    out_dir = Path(output_dir) / rel_path.parent
                    out_dir.mkdir(parents=True, exist_ok=True)
                    self.process_file(img_file, out_dir)
                else:
                    self.process_file(img_file, output_dir)
                    
            except Exception as e:
                logger.error(f"Unexpected error processing {img_file}: {e}")
                self.stats['failed'] += 1
        
        return self.stats
    
    def print_summary(self):
        """Print processing summary."""
        logger.info("\n" + "="*50)
        logger.info("CONVERSION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Successfully converted: {self.stats['converted']}")
        logger.info(f"Already RGB (no change): {self.stats['already_rgb']}")
        logger.info(f"Single-channel skipped (GT labels): {self.stats['single_channel_skipped']}")
        logger.info(f"Failed conversions: {self.stats['failed']}")
        logger.info(f"Backup files created: {self.stats['backed_up']}")
        
        if self.stats['failed'] > 0:
            logger.warning(f"{self.stats['failed']} files failed to convert!")
        else:
            logger.info("✅ All files processed successfully!")

def main():
    parser = argparse.ArgumentParser(
        description="Convert RGBA images to RGB for anomalib compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix all images in Adam dataset recursively (with backups)
  python fix_rgba_to_rgb.py /path/to/reorganized_adam --recursive
  
  # Fix specific directory without backups
  python fix_rgba_to_rgb.py /path/to/rgb/directory --no-backup
  
  # Fix and save to new location
  python fix_rgba_to_rgb.py /path/to/input -o /path/to/output --recursive
  
  # Check what needs fixing (dry run)
  python fix_rgba_to_rgb.py /path/to/images --check-only
        """
    )
    
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('-o', '--output-dir', help='Output directory (default: overwrite in place)')
    parser.add_argument('-r', '--recursive', action='store_true', default=True,
                       help='Process subdirectories recursively (default: True)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup files when overwriting')
    parser.add_argument('--extensions', nargs='+', 
                       default=['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'],
                       help='File extensions to process (default: .png .jpg .jpeg .tiff .tif .bmp)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check image channels, do not convert')
    
    args = parser.parse_args()
    
    try:
        if args.check_only:
            # Just check and report
            check_directory_channels(args.input_dir, args.recursive, args.extensions)
        else:
            # Convert files
            converter = RGBAToRGBConverter(
                backup=not args.no_backup,
                verbose=args.verbose
            )
            
            converter.process_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                recursive=args.recursive,
                file_extensions=args.extensions
            )
            
            converter.print_summary()
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        exit(1)

def check_directory_channels(input_dir, recursive=True, extensions=None):
    """
    Check and report on image channels in directory without converting.
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']
    
    input_path = Path(input_dir)
    
    # Find all image files
    image_files = []
    for ext in extensions:
        if recursive:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        else:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Check channels
    channel_counts = {1: 0, 3: 0, 4: 0, 'error': 0}
    problematic_files = []
    gt_files = []
    
    for img_file in image_files:
        try:
            img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
            if img is None:
                channel_counts['error'] += 1
                continue
            
            if len(img.shape) == 2:
                channels = 1
            elif len(img.shape) == 3:
                channels = img.shape[2]
            else:
                channel_counts['error'] += 1
                continue
            
            if channels in channel_counts:
                channel_counts[channels] += 1
            else:
                channel_counts['error'] += 1
            
            if channels == 4:
                problematic_files.append(img_file)
            elif channels == 1:
                gt_files.append(img_file)
                
        except Exception:
            channel_counts['error'] += 1
    
    # Report
    print(f"\nImage Channel Analysis for {input_dir}")
    print("="*50)
    print(f"Total files: {len(image_files)}")
    print(f"Grayscale (1 channel): {channel_counts[1]} (GT labels - will be skipped)")
    print(f"RGB (3 channels): {channel_counts[3]} ✅")
    print(f"RGBA (4 channels): {channel_counts[4]} ⚠️")
    print(f"Errors/Unknown: {channel_counts['error']}")
    
    if channel_counts[1] > 0:
        print(f"\n📋 Found {channel_counts[1]} single-channel images (ground truth labels)")
        print("These will be SKIPPED during conversion (as they should be).")
    
    if channel_counts[4] > 0:
        print(f"\n❌ Found {channel_counts[4]} RGBA images that need conversion!")
        print("These will cause 'Image must have 3 channels, got 4' errors in anomalib.")
        
        if len(problematic_files) <= 10:
            print("\nRGBA files that need conversion:")
            for f in problematic_files:
                print(f"  {f}")
        else:
            print(f"\nFirst 10 RGBA files that need conversion:")
            for f in problematic_files[:10]:
                print(f"  {f}")
            print(f"  ... and {len(problematic_files) - 10} more")
    else:
        print("\n✅ All RGB images are already 3-channel compatible!")

if __name__ == "__main__":
    main()