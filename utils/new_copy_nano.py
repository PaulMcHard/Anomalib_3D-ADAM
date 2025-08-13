#!/usr/bin/env python3
"""
Dataset File Extractor - Extract files containing 'Nano' keyword
Preserves original directory structure while copying only matching files.
Special rules:
- From 'xyz' subdirectories, only .tiff files are copied
- Files in 'tiff' subdirectories are skipped if equivalent exists in 'xyz' subdirectories
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple


def find_nano_files(source_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Find all files containing 'Nano' in their filename.
    Rules:
    - For files in 'xyz' subdirectories, only include .tiff files
    - Skip files in 'tiff' subdirectories if equivalent file exists in 'xyz' subdirectory
    
    Args:
        source_dir: Source directory to search
        
    Returns:
        List of tuples containing (source_path, relative_path)
    """
    nano_files = []
    xyz_files = set()  # Track files found in xyz directories
    
    # First pass: collect all .tiff files from xyz directories
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if 'Nano' in file:
                source_path = Path(root) / file
                relative_path = source_path.relative_to(source_dir)
                
                if 'xyz' in relative_path.parts and source_path.suffix.lower() == '.ply':
                    nano_files.append((source_path, relative_path))
                    # Store the filename for duplicate checking
                    xyz_files.add(file)
    
    # Second pass: collect files from other directories, excluding duplicates from tiff dirs
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if 'Nano' in file:
                source_path = Path(root) / file
                relative_path = source_path.relative_to(source_dir)
                
                # Skip if already processed in xyz directory
                if 'xyz' in relative_path.parts:
                    continue
                
                # Check if this is a tiff directory with a duplicate in xyz
                if 'tiff' in relative_path.parts and file in xyz_files:
                    continue  # Skip this file as it exists in xyz directory
                
                # Include all other files
                nano_files.append((source_path, relative_path))
    
    return nano_files


def create_directory_structure(target_dir: Path, relative_path: Path) -> None:
    """
    Create the directory structure for a given relative path.
    
    Args:
        target_dir: Target base directory
        relative_path: Relative path from source
    """
    target_path = target_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)


def copy_nano_files(source_dir: Path, target_dir: Path, dry_run: bool = False) -> None:
    """
    Copy all files containing 'Nano' to target directory preserving structure.
    
    Args:
        source_dir: Source dataset directory
        target_dir: Target directory for extracted files
        dry_run: If True, only print what would be copied without actually copying
    """
    print(f"Searching for files containing 'Nano' in: {source_dir}")
    print("Rules applied:")
    print("- From 'xyz' subdirectories, only .tiff files will be included")
    print("- Files in 'tiff' subdirectories will be skipped if equivalent exists in 'xyz'")
    print("- All other directories (binary_masks, rgb, gt, etc.) will include all 'Nano' files")
    
    nano_files = find_nano_files(source_dir)
    
    if not nano_files:
        print("No files containing 'Nano' found!")
        return
    
    print(f"Found {len(nano_files)} files containing 'Nano'")
    
    # Show breakdown by directory type for debugging
    dir_breakdown = {}
    for source_path, relative_path in nano_files:
        parent_dir = relative_path.parent.name
        if parent_dir not in dir_breakdown:
            dir_breakdown[parent_dir] = 0
        dir_breakdown[parent_dir] += 1
    
    print("Files found by directory:")
    for dir_name, count in sorted(dir_breakdown.items()):
        print(f"  {dir_name}: {count} files")
    
    if dry_run:
        print("\n--- DRY RUN MODE - No files will be copied ---")
    
    copied_count = 0
    skipped_count = 0
    
    for source_path, relative_path in nano_files:
        target_path = target_dir / relative_path
        
        if dry_run:
            print(f"Would copy: {relative_path}")
            continue
        
        # Create directory structure
        create_directory_structure(target_dir, relative_path)
        
        try:
            # Check if file already exists
            if target_path.exists():
                print(f"Skipping (exists): {relative_path}")
                skipped_count += 1
                continue
            
            # Copy the file
            shutil.copy2(source_path, target_path)
            print(f"Copied: {relative_path}")
            copied_count += 1
            
        except Exception as e:
            print(f"Error copying {relative_path}: {e}")
    
    if not dry_run:
        print(f"\nCopy complete! {copied_count} files copied, {skipped_count} files skipped")


def analyze_nano_files(source_dir: Path) -> None:
    """
    Analyze and categorize files containing 'Nano'.
    Shows statistics including files excluded due to deduplication rules.
    
    Args:
        source_dir: Source dataset directory
    """
    nano_files = find_nano_files(source_dir)
    
    if not nano_files:
        print("No files containing 'Nano' found!")
        return
    
    # Categorize by file type and directory
    categories = {}
    file_types = {}
    stats = {
        'xyz_tiff_included': 0,
        'xyz_other_excluded': 0,
        'tiff_duplicates_excluded': 0,
        'tiff_unique_included': 0
    }
    
    # Collect all Nano files and analyze exclusions
    xyz_files = set()
    tiff_files = set()
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if 'Nano' in file:
                source_path = Path(root) / file
                relative_path = source_path.relative_to(source_dir)
                
                if 'xyz' in relative_path.parts:
                    if source_path.suffix.lower() == '.tiff':
                        stats['xyz_tiff_included'] += 1
                        xyz_files.add(file)
                    else:
                        stats['xyz_other_excluded'] += 1
                elif 'tiff' in relative_path.parts:
                    if file in xyz_files:
                        stats['tiff_duplicates_excluded'] += 1
                    else:
                        stats['tiff_unique_included'] += 1
                        tiff_files.add(file)
    
    # Analyze the files that will actually be copied
    for source_path, relative_path in nano_files:
        # Categorize by parent directory
        parent_dir = relative_path.parent.name
        if parent_dir not in categories:
            categories[parent_dir] = 0
        categories[parent_dir] += 1
        
        # Categorize by file extension
        ext = source_path.suffix.lower()
        if ext not in file_types:
            file_types[ext] = 0
        file_types[ext] += 1
    
    print(f"\nAnalysis of files containing 'Nano':")
    print(f"Total files to be copied: {len(nano_files)}")
    
    print(f"\nRule-based exclusions:")
    if stats['xyz_other_excluded'] > 0:
        print(f"  Files excluded from 'xyz' directories (non-.tiff): {stats['xyz_other_excluded']}")
    if stats['tiff_duplicates_excluded'] > 0:
        print(f"  Files excluded from 'tiff' directories (duplicates in xyz): {stats['tiff_duplicates_excluded']}")
    
    print(f"\nFiles included:")
    if stats['xyz_tiff_included'] > 0:
        print(f"  From 'xyz' directories (.tiff only): {stats['xyz_tiff_included']}")
    if stats['tiff_unique_included'] > 0:
        print(f"  From 'tiff' directories (unique files): {stats['tiff_unique_included']}")
    
    print("\nFiles by directory:")
    for directory, count in sorted(categories.items()):
        print(f"  {directory}: {count} files")
    
    print("\nFiles by type:")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count} files")


def test_file_detection(source_dir: Path, sample_size: int = 10) -> None:
    """
    Test function to show sample files that would be detected.
    Useful for debugging file detection logic.
    
    Args:
        source_dir: Source dataset directory
        sample_size: Number of sample files to show per directory
    """
    print(f"Testing file detection in: {source_dir}")
    print("Showing sample files that contain 'Nano'...\n")
    
    # Collect files by directory type
    files_by_dir = {}
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if 'Nano' in file:
                source_path = Path(root) / file
                relative_path = source_path.relative_to(source_dir)
                parent_dir = relative_path.parent.name
                
                if parent_dir not in files_by_dir:
                    files_by_dir[parent_dir] = []
                files_by_dir[parent_dir].append(relative_path)
    
    # Show samples from each directory
    for dir_name, file_list in sorted(files_by_dir.items()):
        print(f"{dir_name} directory ({len(file_list)} total files):")
        for i, file_path in enumerate(file_list[:sample_size]):
            # Check if this file would be included by our rules
            would_include = True
            reason = ""
            
            if 'xyz' in file_path.parts:
                if not file_path.suffix.lower() == '.tiff':
                    would_include = False
                    reason = "(excluded: xyz dir, not .tiff)"
            elif 'tiff' in file_path.parts:
                # This is simplified - in real logic we'd check for duplicates
                reason = "(would check for xyz duplicate)"
            
            status = "✅" if would_include else "❌"
            print(f"  {status} {file_path} {reason}")
            
        if len(file_list) > sample_size:
            print(f"  ... and {len(file_list) - sample_size} more files")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract files containing 'Nano' from dataset while preserving directory structure"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Source dataset directory"
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Target directory for extracted files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without actually copying"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze files before copying"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test file detection and show samples"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in target directory"
    )
    
    args = parser.parse_args()
    
    # Validate source directory
    if not args.source_dir.exists():
        print(f"Error: Source directory '{args.source_dir}' does not exist!")
        return 1
    
    if not args.source_dir.is_dir():
        print(f"Error: '{args.source_dir}' is not a directory!")
        return 1
    
    # Create target directory if it doesn't exist (unless just testing)
    if not args.dry_run and not args.test:
        args.target_dir.mkdir(parents=True, exist_ok=True)
    
    # Test file detection if requested
    if args.test:
        test_file_detection(args.source_dir)
        if not args.analyze and not args.dry_run:
            return 0
    
    # Analyze if requested
    if args.analyze:
        analyze_nano_files(args.source_dir)
        print()
    
    # Copy files (unless only testing)
    if not args.test or args.dry_run:
        copy_nano_files(args.source_dir, args.target_dir, args.dry_run)
    
    return 0


if __name__ == "__main__":
    exit(main())