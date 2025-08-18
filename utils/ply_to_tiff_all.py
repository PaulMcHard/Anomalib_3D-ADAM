#!/usr/bin/env python3
"""
Convert PLY point cloud files to 3-channel TIFF depth images.
Built from scratch for anomalib compatibility.
"""

import open3d as o3d
import numpy as np
import tifffile
import os
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PLYToTIFFConverter:
    def __init__(self, width=1280, height=1024, focal_length=550.0):
        """
        Initialize the converter.
        
        Args:
            width: Output image width (default: 1280)
            height: Output image height (default: 1024)
            focal_length: Camera focal length (default: 550.0)
        """
        self.width = width
        self.height = height
        self.focal_length = focal_length
        
        logger.info(f"Initialized converter with dimensions: {width}x{height}, focal length: {focal_length}")
    
    def create_depth_image_from_pointcloud(self, pcd):
        """
        Convert point cloud to depth image using orthographic projection.
        
        Args:
            pcd: Open3D point cloud object
            
        Returns:
            numpy.ndarray: 2D depth image (height, width) as float32
        """
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            logger.warning("Empty point cloud, returning zero depth image")
            return np.zeros((self.height, self.width), dtype=np.float32)
        
        # Extract coordinates
        x_coords = points[:, 0]
        y_coords = points[:, 1]  
        z_coords = points[:, 2]
        
        # Get bounding box
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        logger.debug(f"Point cloud bounds: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}], Z[{z_min:.3f}, {z_max:.3f}]")
        
        # Handle degenerate cases
        if x_max <= x_min:
            x_max = x_min + 1e-6
        if y_max <= y_min:
            y_max = y_min + 1e-6
        if z_max <= z_min:
            z_max = z_min + 1e-6
        
        # Map 3D coordinates to 2D pixel coordinates
        pixel_x = ((x_coords - x_min) / (x_max - x_min) * (self.width - 1)).astype(np.int32)
        pixel_y = ((y_coords - y_min) / (y_max - y_min) * (self.height - 1)).astype(np.int32)
        
        # Normalize depth values to [0, 1]
        depth_normalized = (z_coords - z_min) / (z_max - z_min)
        
        # Create depth image
        depth_image = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Fill depth values (handle overlapping pixels by keeping closest depth)
        valid_mask = (pixel_x >= 0) & (pixel_x < self.width) & (pixel_y >= 0) & (pixel_y < self.height)
        
        for i in np.where(valid_mask)[0]:
            px, py = pixel_x[i], pixel_y[i]
            current_depth = depth_normalized[i]
            
            # If pixel is empty or current point is closer, update
            if depth_image[py, px] == 0.0 or current_depth < depth_image[py, px]:
                depth_image[py, px] = current_depth
        
        logger.debug(f"Filled {np.count_nonzero(depth_image)} pixels with depth values")
        return depth_image
    
    def convert_to_3channel_tiff(self, depth_image):
        """
        Convert single-channel depth image to 3-channel TIFF format.
        
        Args:
            depth_image: 2D numpy array with depth values [0, 1]
            
        Returns:
            numpy.ndarray: 3-channel depth image (height, width, 3) as float32
        """
        # Ensure input is float32
        depth_float32 = depth_image.astype(np.float32)
        
        # Stack the same depth values into 3 channels (R=G=B=depth)
        depth_3channel = np.stack([depth_float32, depth_float32, depth_float32], axis=2)
        
        logger.debug(f"Created 3-channel image with shape: {depth_3channel.shape}")
        logger.debug(f"Data type: {depth_3channel.dtype}")
        logger.debug(f"Value range: [{np.min(depth_3channel):.6f}, {np.max(depth_3channel):.6f}]")
        
        return depth_3channel
    
    def save_tiff(self, image_3channel, output_path):
        """
        Save 3-channel image as TIFF using tifffile.
        
        Args:
            image_3channel: 3-channel numpy array (height, width, 3)
            output_path: Path to save TIFF file
        """
        try:
            # Ensure the image has exactly 3 channels
            if len(image_3channel.shape) != 3 or image_3channel.shape[2] != 3:
                raise ValueError(f"Expected 3-channel image, got shape: {image_3channel.shape}")
            
            # Save with tifffile using LZW compression (lossless and widely supported)
            tifffile.imwrite(
                str(output_path),
                image_3channel,
                compression='lzw',
                photometric='rgb',  # Explicitly specify as RGB
                planarconfig='contig'  # Interleaved RGB format
            )
            
            logger.debug(f"Saved TIFF: {output_path}")
            
            # Verify the saved file
            self.verify_tiff_file(output_path)
            
        except Exception as e:
            logger.error(f"Failed to save TIFF {output_path}: {e}")
            raise
    
    def verify_tiff_file(self, tiff_path):
        """
        Verify that the saved TIFF file has correct properties.
        
        Args:
            tiff_path: Path to TIFF file to verify
        """
        try:
            # Read back the saved file
            saved_image = tifffile.imread(str(tiff_path))
            
            logger.debug(f"Verification - Shape: {saved_image.shape}, dtype: {saved_image.dtype}")
            
            # Check properties
            if len(saved_image.shape) != 3:
                raise ValueError(f"Saved image should be 3D, got shape: {saved_image.shape}")
            
            if saved_image.shape[2] != 3:
                raise ValueError(f"Saved image should have 3 channels, got: {saved_image.shape[2]}")
            
            if saved_image.shape[:2] != (self.height, self.width):
                raise ValueError(f"Saved image has wrong dimensions: {saved_image.shape[:2]} != ({self.height}, {self.width})")
            
            logger.debug(f"✓ TIFF verification passed: {tiff_path}")
            
        except Exception as e:
            logger.error(f"TIFF verification failed for {tiff_path}: {e}")
            raise
    
    def convert_ply_file(self, ply_path, output_path):
        """
        Convert a single PLY file to 3-channel TIFF.
        
        Args:
            ply_path: Path to input PLY file
            output_path: Path for output TIFF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Processing: {ply_path}")
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(ply_path))
            
            if len(pcd.points) == 0:
                logger.warning(f"No points found in {ply_path}")
                return False
            
            logger.debug(f"Loaded point cloud with {len(pcd.points)} points")
            
            # Convert to depth image
            depth_image = self.create_depth_image_from_pointcloud(pcd)
            
            # Convert to 3-channel format
            depth_3channel = self.convert_to_3channel_tiff(depth_image)
            
            # Save as TIFF
            self.save_tiff(depth_3channel, output_path)
            
            logger.info(f"✓ Successfully converted: {ply_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert {ply_path}: {e}")
            return False
    
    def process_directory(self, input_dir, output_dir=None, recursive=False):
        """
        Process all PLY files in a directory.
        
        Args:
            input_dir: Input directory containing PLY files
            output_dir: Output directory (if None and not recursive, saves in same directory)
            recursive: Whether to process subdirectories
            
        Returns:
            dict: Statistics about processing
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Set up output directory (only used for non-recursive mode)
        if output_dir is not None and not recursive:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = input_path
        
        # Find PLY files
        if recursive:
            ply_files = list(input_path.rglob("*.ply"))
        else:
            ply_files = list(input_path.glob("*.ply"))
        
        if not ply_files:
            logger.warning(f"No PLY files found in {input_dir}")
            return {"total": 0, "success": 0, "failed": 0}
        
        logger.info(f"Found {len(ply_files)} PLY files to process")
        
        # Process files
        success_count = 0
        failed_count = 0
        
        for ply_file in ply_files:
            try:
                # Determine output file path
                if recursive:
                    # When recursive, always create tiff in same directory as ply file
                    tiff_file = ply_file.with_suffix('.tiff')
                elif output_dir is not None:
                    # Non-recursive with output dir specified
                    tiff_file = output_path / f"{ply_file.stem}.tiff"
                else:
                    # Non-recursive, same directory
                    tiff_file = ply_file.with_suffix('.tiff')
                
                # Convert file
                if self.convert_ply_file(ply_file, tiff_file):
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"Unexpected error processing {ply_file}: {e}")
                failed_count += 1
        
        # Report results
        stats = {"total": len(ply_files), "success": success_count, "failed": failed_count}
        logger.info(f"Processing complete: {success_count}/{len(ply_files)} files converted successfully")
        
        if failed_count > 0:
            logger.warning(f"{failed_count} files failed to convert")
        
        return stats

def main():
    parser = argparse.ArgumentParser(
        description="Convert PLY point clouds to 3-channel TIFF depth images for anomalib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PLY files in current directory
  python convert_ply_to_tiff.py /path/to/ply/files
  
  # Convert recursively (TIFF files created in same dirs as PLY files)
  python convert_ply_to_tiff.py /path/to/ply/files --recursive
  
  # Convert to specific output directory (non-recursive only)
  python convert_ply_to_tiff.py /path/to/ply/files -o /output/path
  
  # Custom dimensions and focal length
  python convert_ply_to_tiff.py /path/to/ply/files --width 640 --height 480 --focal-length 500

Note: When using --recursive, output TIFF files are always created in the same
      directory as their corresponding PLY files, regardless of --output-dir setting.
        """
    )
    
    parser.add_argument('input_dir', help='Input directory containing PLY files')
    parser.add_argument('-o', '--output-dir', help='Output directory (ignored when using --recursive)')
    parser.add_argument('-r', '--recursive', action='store_true', 
                       help='Process subdirectories recursively (TIFF files created in same dirs as PLY files)')
    parser.add_argument('--width', type=int, default=1280,
                       help='Output image width (default: 1280)')
    parser.add_argument('--height', type=int, default=1024,
                       help='Output image height (default: 1024)')
    parser.add_argument('--focal-length', type=float, default=550.0,
                       help='Camera focal length (default: 550.0)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Warn user about output-dir being ignored in recursive mode
    if args.recursive and args.output_dir:
        logger.warning("--output-dir is ignored when using --recursive mode. TIFF files will be created in the same directories as PLY files.")
    
    try:
        # Create converter
        converter = PLYToTIFFConverter(
            width=args.width,
            height=args.height,
            focal_length=args.focal_length
        )
        
        # Process directory
        stats = converter.process_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            recursive=args.recursive
        )
        
        # Exit with appropriate code
        if stats["failed"] > 0:
            exit(1)
        else:
            exit(0)
            
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()