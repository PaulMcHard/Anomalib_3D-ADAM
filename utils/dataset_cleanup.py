#!/usr/bin/env python3
"""
Dataset cleanup script to perform the following tasks in order:
1. Remove "tiff" endpoint directories
2. Remove "gt" directories  
3. Rename "gt_consolidated" directories to "ground_truth"
4. Remove .ply files from "xyz" folders only if corresponding .tiff file exists

Provides --dry-run option to preview changes without executing them.
"""

import os
import argparse
import shutil
from pathlib import Path

def find_directories_by_name(root_path, target_name):
    """
    Find all directories with the specified name in the directory tree.
    Returns a list of full paths to matching directories.
    """
    matching_dirs = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        if target_name in dirnames:
            matching_dir = os.path.join(dirpath, target_name)
            matching_dirs.append(matching_dir)
    
    return matching_dirs

def remove_tiff_directories(root_path, dry_run=True):
    """
    Task 1: Remove "tiff" endpoint directories
    """
    print("\n" + "="*60)
    print("TASK 1: Removing 'tiff' endpoint directories")
    print("="*60)
    
    tiff_dirs = find_directories_by_name(root_path, "tiff")
    
    if not tiff_dirs:
        print("✅ No 'tiff' directories found")
        return True
    
    print(f"Found {len(tiff_dirs)} 'tiff' directories:")
    
    removed_count = 0
    for tiff_dir in tiff_dirs:
        rel_path = os.path.relpath(tiff_dir, root_path)
        
        if dry_run:
            print(f"   📝 Would remove: {rel_path}")
        else:
            try:
                shutil.rmtree(tiff_dir)
                print(f"   ✅ Removed: {rel_path}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {rel_path}: {e}")
                return False
    
    if dry_run:
        print(f"📊 Would remove {len(tiff_dirs)} 'tiff' directories")
    else:
        print(f"📊 Successfully removed {removed_count}/{len(tiff_dirs)} 'tiff' directories")
    
    return True

def remove_gt_directories(root_path, dry_run=True):
    """
    Task 2: Remove "gt" directories
    """
    print("\n" + "="*60)
    print("TASK 2: Removing 'gt' directories")
    print("="*60)
    
    gt_dirs = find_directories_by_name(root_path, "gt")
    
    if not gt_dirs:
        print("✅ No 'gt' directories found")
        return True
    
    print(f"Found {len(gt_dirs)} 'gt' directories:")
    
    removed_count = 0
    for gt_dir in gt_dirs:
        rel_path = os.path.relpath(gt_dir, root_path)
        
        if dry_run:
            print(f"   📝 Would remove: {rel_path}")
        else:
            try:
                shutil.rmtree(gt_dir)
                print(f"   ✅ Removed: {rel_path}")
                removed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to remove {rel_path}: {e}")
                return False
    
    if dry_run:
        print(f"📊 Would remove {len(gt_dirs)} 'gt' directories")
    else:
        print(f"📊 Successfully removed {removed_count}/{len(gt_dirs)} 'gt' directories")
    
    return True

def rename_gt_consolidated_directories(root_path, dry_run=True):
    """
    Task 3: Rename "gt_consolidated" directories to "ground_truth"
    """
    print("\n" + "="*60)
    print("TASK 3: Renaming 'gt_consolidated' directories to 'ground_truth'")
    print("="*60)
    
    gt_consolidated_dirs = find_directories_by_name(root_path, "gt_consolidated")
    
    if not gt_consolidated_dirs:
        print("✅ No 'gt_consolidated' directories found")
        return True
    
    print(f"Found {len(gt_consolidated_dirs)} 'gt_consolidated' directories:")
    
    renamed_count = 0
    for gt_consolidated_dir in gt_consolidated_dirs:
        parent_dir = os.path.dirname(gt_consolidated_dir)
        new_path = os.path.join(parent_dir, "ground_truth")
        
        rel_old_path = os.path.relpath(gt_consolidated_dir, root_path)
        rel_new_path = os.path.relpath(new_path, root_path)
        
        # Check if target already exists
        if os.path.exists(new_path):
            print(f"   ⚠️  Cannot rename {rel_old_path} -> {rel_new_path} (target exists)")
            continue
        
        if dry_run:
            print(f"   📝 Would rename: {rel_old_path} -> {rel_new_path}")
        else:
            try:
                os.rename(gt_consolidated_dir, new_path)
                print(f"   ✅ Renamed: {rel_old_path} -> {rel_new_path}")
                renamed_count += 1
            except Exception as e:
                print(f"   ❌ Failed to rename {rel_old_path}: {e}")
                return False
    
    if dry_run:
        print(f"📊 Would rename {len(gt_consolidated_dirs)} directories")
    else:
        print(f"📊 Successfully renamed {renamed_count}/{len(gt_consolidated_dirs)} directories")
    
    return True

def cleanup_ply_files_in_xyz_folders(root_path, dry_run=True):
    """
    Task 4: Remove .ply files from "xyz" folders only if corresponding .tiff file exists
    """
    print("\n" + "="*60)
    print("TASK 4: Cleaning up .ply files in 'xyz' folders")
    print("="*60)
    
    xyz_dirs = find_directories_by_name(root_path, "xyz")
    
    if not xyz_dirs:
        print("✅ No 'xyz' directories found")
        return True
    
    print(f"Found {len(xyz_dirs)} 'xyz' directories:")
    
    total_ply_files = 0
    removed_ply_files = 0
    warnings = []
    
    for xyz_dir in xyz_dirs:
        rel_xyz_path = os.path.relpath(xyz_dir, root_path)
        print(f"\n   📁 Processing: {rel_xyz_path}")
        
        # Get all .ply files in this xyz directory
        ply_files = []
        try:
            for filename in os.listdir(xyz_dir):
                if filename.lower().endswith('.ply'):
                    ply_files.append(filename)
        except Exception as e:
            print(f"      ❌ Error reading directory {rel_xyz_path}: {e}")
            continue
        
        if not ply_files:
            print(f"      ✅ No .ply files found")
            continue
        
        print(f"      Found {len(ply_files)} .ply files")
        total_ply_files += len(ply_files)
        
        # Check each .ply file for corresponding .tiff file
        for ply_filename in ply_files:
            # Get base name without extension
            base_name = os.path.splitext(ply_filename)[0]
            tiff_filename = base_name + '.tiff'
            
            ply_path = os.path.join(xyz_dir, ply_filename)
            tiff_path = os.path.join(xyz_dir, tiff_filename)
            
            if os.path.exists(tiff_path):
                # .tiff file exists, safe to remove .ply
                if dry_run:
                    print(f"      📝 Would remove: {ply_filename} (has corresponding {tiff_filename})")
                else:
                    try:
                        os.remove(ply_path)
                        print(f"      ✅ Removed: {ply_filename} (has corresponding {tiff_filename})")
                        removed_ply_files += 1
                    except Exception as e:
                        print(f"      ❌ Failed to remove {ply_filename}: {e}")
            else:
                # No corresponding .tiff file found - warning
                warning_msg = f"{rel_xyz_path}/{ply_filename} - no corresponding .tiff file found"
                warnings.append(warning_msg)
                print(f"      ⚠️  WARNING: {ply_filename} - no corresponding {tiff_filename} found")
    
    # Summary for task 4
    if dry_run:
        print(f"\n📊 Would process {total_ply_files} .ply files")
        if warnings:
            print(f"⚠️  {len(warnings)} warnings (no corresponding .tiff files)")
    else:
        print(f"\n📊 Successfully removed {removed_ply_files}/{total_ply_files} .ply files")
        if warnings:
            print(f"⚠️  {len(warnings)} warnings (no corresponding .tiff files)")
    
    # Show all warnings at the end
    if warnings:
        print(f"\n⚠️  WARNINGS - .ply files without corresponding .tiff files:")
        for warning in warnings:
            print(f"   - {warning}")
    
    return True

def cleanup_dataset(root_path, dry_run=True):
    """
    Main cleanup function that executes all tasks in order.
    """
    root_path = Path(root_path)
    
    if not root_path.exists():
        print(f"❌ Error: Path '{root_path}' does not exist.")
        return False
    
    if not root_path.is_dir():
        print(f"❌ Error: Path '{root_path}' is not a directory.")
        return False
    
    print(f"🧹 {'DRY RUN: ' if dry_run else ''}Dataset Cleanup")
    print(f"📁 Root path: {root_path}")
    print(f"🎯 Tasks to perform:")
    print(f"   1. Remove 'tiff' endpoint directories")
    print(f"   2. Remove 'gt' directories")
    print(f"   3. Rename 'gt_consolidated' directories to 'ground_truth'")
    print(f"   4. Clean up .ply files in 'xyz' folders")
    
    # Execute tasks in order
    success = True
    
    # Task 1: Remove tiff directories
    if not remove_tiff_directories(root_path, dry_run):
        success = False
    
    # Task 2: Remove gt directories
    if success and not remove_gt_directories(root_path, dry_run):
        success = False
    
    # Task 3: Rename gt_consolidated directories
    if success and not rename_gt_consolidated_directories(root_path, dry_run):
        success = False
    
    # Task 4: Cleanup ply files in xyz folders
    if success and not cleanup_ply_files_in_xyz_folders(root_path, dry_run):
        success = False
    
    # Final summary
    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    
    if success:
        if dry_run:
            print("✅ All tasks completed successfully (dry run)")
            print("💡 Run with --no-dry-run to actually perform the cleanup")
        else:
            print("🎉 All cleanup tasks completed successfully!")
    else:
        print("❌ Some tasks failed - check the output above for details")
    
    return success

def main():
    parser = argparse.ArgumentParser(
        description="Cleanup dataset by removing specific directories and files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tasks performed in order:
1. Remove all 'tiff' endpoint directories
2. Remove all 'gt' directories  
3. Rename all 'gt_consolidated' directories to 'ground_truth'
4. Remove .ply files from 'xyz' folders (only if corresponding .tiff exists)

Examples:
  # Dry run (safe preview)
  python cleanup_dataset.py /path/to/dataset
  
  # Actually perform cleanup
  python cleanup_dataset.py /path/to/dataset --no-dry-run
        """
    )
    
    parser.add_argument(
        'path',
        help='Path to the dataset root directory'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Perform a dry run without actually making changes (default)'
    )
    
    parser.add_argument(
        '--no-dry-run',
        action='store_false',
        dest='dry_run',
        help='Actually perform the cleanup operations (use with caution)'
    )
    
    args = parser.parse_args()
    
    success = cleanup_dataset(args.path, args.dry_run)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())