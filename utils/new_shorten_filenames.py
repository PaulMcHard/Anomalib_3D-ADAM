#!/usr/bin/env python3
"""
Script to shorten dataset filenames from long format to just image numbers.
Converts: 
  - PartName_InstanceLetter_DefectType__CameraName_ImageNumber_mask.png → ImageNumber.png (or ImageNumber_InstanceLetter.png if clash)
  - PartName_InstanceLetter_DefectType__CameraName_ImageNumber_image.png → ImageNumber.png (or ImageNumber_InstanceLetter.png if clash)
  - PartName_InstanceLetter_DefectType__CameraName_ImageNumber_under_extrusion_mask.png → ImageNumber.png (handles extra text before mask)
  - PartName_InstanceLetter_DefectType__CameraName_ImageNumber_xyz.tiff → ImageNumber.tiff (or ImageNumber_InstanceLetter.tiff if clash)
  
Special handling for duplicate masks:
  - When multiple mask files exist with the same number and instance (variations with "_image" or spaces),
    only the shortest filename is renamed, others are left unchanged.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict

def extract_components(filename):
    """
    Extract the image number and instance letter from a filename.
    Expected formats:
    - PartName_InstanceLetter_DefectType__CameraName_ImageNumber_mask.png (for defects)
    - PartName_InstanceLetter_DefectType__CameraName_ImageNumber_image.png
    - PartName_InstanceLetter_DefectType__CameraName_ImageNumber_*_mask.png (extra text before mask)
    - PartName_InstanceLetter_DefectType__CameraName_ImageNumber_*_image.png (extra text before image)
    - PartName_InstanceLetter_DefectType__CameraName_ImageNumber_xyz.tiff
    Returns tuple (image_number, instance_letter) or (None, None) if pattern doesn't match.
    """
    # Pattern to match PNG files with _mask or _image suffix (may have extra text before)
    # Captures instance letter and image number
    # Handles cases like _under_extrusion_mask.png or _mask.png
    pattern_png = r"^.*?_([A-Z])_.*?__.*?_(\d+)_.*?(?:mask|image)\.png$"
    
    # Pattern to match TIFF files with _xyz suffix
    pattern_tiff = r"^.*?_([A-Z])_.*?__.*?_(\d+)_xyz\.tiff$"
    
    # Try PNG pattern first
    match = re.match(pattern_png, filename)
    if match:
        return match.group(2), match.group(1)  # image_number, instance_letter
    
    # Try TIFF pattern
    match = re.match(pattern_tiff, filename)
    if match:
        return match.group(2), match.group(1)  # image_number, instance_letter
    
    # Try alternative patterns without double underscore
    # Also handle extra text before mask/image
    pattern_alt_png = r"^.*?_([A-Z])_.*?_(\d+)_.*?(?:mask|image)\.png$"
    pattern_alt_tiff = r"^.*?_([A-Z])_.*?_(\d+)_xyz\.tiff$"
    
    match = re.match(pattern_alt_png, filename)
    if match:
        return match.group(2), match.group(1)
    
    match = re.match(pattern_alt_tiff, filename)
    if match:
        return match.group(2), match.group(1)
    
    # Fallback: Try to extract just the image number without instance letter
    # Also handle extra text before mask/image
    pattern_fallback_png = r"^.*?_(\d+)_.*?(?:mask|image)\.png$"
    pattern_fallback_tiff = r"^.*?_(\d+)_cloud\.tiff$"
    
    match = re.match(pattern_fallback_png, filename)
    if match:
        return match.group(1), None
    
    match = re.match(pattern_fallback_tiff, filename)
    if match:
        return match.group(1), None
    
    return None, None

def determine_rename_strategy(directory_path):
    """
    Determine the best renaming strategy for files in a directory.
    Returns a dictionary mapping old filenames to new filenames.
    """
    # First pass: collect all files and their components
    file_info = {}
    image_number_groups = defaultdict(list)
    
    for filename in os.listdir(directory_path):
        # Check for both .png and .tiff files
        if not (filename.endswith('.png') or filename.endswith('.tiff')):
            continue
            
        filepath = os.path.join(directory_path, filename)
        if not os.path.isfile(filepath):
            continue
        
        image_number, instance_letter = extract_components(filename)
        if image_number is None:
            print(f"  Warning: Cannot parse: {filename}")
            continue
        
        extension = '.tiff' if filename.endswith('.tiff') else '.png'
        file_info[filename] = {
            'image_number': image_number,
            'instance_letter': instance_letter,
            'extension': extension
        }
        
        # Group files by image number and extension
        key = (image_number, extension)
        image_number_groups[key].append(filename)
    
    # Second pass: determine naming strategy
    renames = {}
    potential_clashes = []
    files_to_skip = set()  # Files we'll leave alone due to duplicate mask situation
    
    for (image_number, extension), filenames in image_number_groups.items():
        if len(filenames) == 1:
            # No clash, use simple naming
            filename = filenames[0]
            new_name = f"{image_number}{extension}"
            renames[filename] = new_name
        else:
            # Multiple files with same number - check the nature of the clash
            
            # First, check if this is a duplicate mask situation
            # (same instance letter, variations in the suffix after the number)
            instance_groups = defaultdict(list)
            for filename in filenames:
                info = file_info[filename]
                instance_groups[info['instance_letter']].append(filename)
            
            # Check each instance group
            resolved_any = False
            for instance_letter, instance_files in instance_groups.items():
                if len(instance_files) > 1:
                    # Multiple files with same instance letter - likely duplicate masks
                    # Check if they're mask files with variations
                    all_masks = all('mask' in f.lower() for f in instance_files)
                    
                    if all_masks:
                        # These are duplicate mask files - take the shortest one
                        shortest = min(instance_files, key=len)
                        
                        # Check if the variations involve "_image" or whitespace issues
                        has_image_variation = any('_image' in f for f in instance_files)
                        has_space_variation = any(' ' in f for f in instance_files)
                        
                        if has_image_variation or has_space_variation:
                            # This is the specific case mentioned - rename shortest, skip others
                            if instance_letter:
                                new_name = f"{image_number}_{instance_letter}{extension}"
                            else:
                                new_name = f"{image_number}{extension}"
                            
                            renames[shortest] = new_name
                            resolved_any = True
                            
                            # Mark other files to be skipped
                            for f in instance_files:
                                if f != shortest:
                                    files_to_skip.add(f)
                                    print(f"   - Will skip (duplicate mask): {f}")
                        else:
                            # Other type of duplicate - add to potential clashes
                            potential_clashes.extend(instance_files)
                    else:
                        # Not all masks - regular clash
                        potential_clashes.extend(instance_files)
                elif len(instance_files) == 1:
                    # Single file for this instance letter
                    filename = instance_files[0]
                    if not resolved_any and len(filenames) > 1:
                        # Need to use instance letter to disambiguate
                        if instance_letter:
                            new_name = f"{image_number}_{instance_letter}{extension}"
                            renames[filename] = new_name
                            resolved_any = True
                        else:
                            potential_clashes.append(filename)
                    elif not resolved_any:
                        # Single file, no clash
                        new_name = f"{image_number}{extension}"
                        renames[filename] = new_name
            
            # If we didn't resolve anything above, try the original logic
            if not resolved_any and not potential_clashes:
                # Check if we can disambiguate with instance letters
                files_with_letters = []
                files_without_letters = []
                
                for filename in filenames:
                    if filename in files_to_skip:
                        continue
                    info = file_info[filename]
                    if info['instance_letter']:
                        files_with_letters.append(filename)
                    else:
                        files_without_letters.append(filename)
                
                # Check if instance letters are unique
                instance_letters = set()
                all_have_letters = len(files_without_letters) == 0
                
                for filename in files_with_letters:
                    letter = file_info[filename]['instance_letter']
                    if letter in instance_letters:
                        # Duplicate instance letters - this is a real clash
                        potential_clashes.extend([f for f in filenames if f not in files_to_skip])
                        break
                    instance_letters.add(letter)
                else:
                    # All instance letters are unique (or don't exist)
                    if all_have_letters:
                        # All files have unique instance letters, use them
                        for filename in filenames:
                            if filename not in files_to_skip:
                                info = file_info[filename]
                                new_name = f"{image_number}_{info['instance_letter']}{extension}"
                                renames[filename] = new_name
                    else:
                        # Mixed or no instance letters - this is a clash
                        potential_clashes.extend([f for f in filenames if f not in files_to_skip])
    
    # Identify true clashes that couldn't be resolved
    clashes = defaultdict(list)
    for filename in potential_clashes:
        if filename not in renames and filename not in files_to_skip:
            info = file_info.get(filename, {})
            if info:
                # These files couldn't be renamed due to conflicts
                key = f"{info['image_number']}{info['extension']}"
                clashes[key].append(filename)
    
    return renames, clashes

def process_directory(root_path, dry_run=True):
    """
    Process all subdirectories and files in the given root path.
    """
    root_path = Path(root_path)
    
    if not root_path.exists():
        print(f"Error: Path '{root_path}' does not exist.")
        return False
    
    total_files = 0
    total_renames = 0
    total_unresolved_clashes = 0
    directories_with_clashes = []
    
    print(f"{'DRY RUN: ' if dry_run else ''}Processing directory: {root_path}")
    print("=" * 60)
    
    # Process root directory and all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter for both PNG and TIFF files
        image_files = [f for f in filenames if f.endswith('.png') or f.endswith('.tiff')]
        
        if not image_files:
            continue
        
        rel_path = os.path.relpath(dirpath, root_path)
        if rel_path == '.':
            rel_path = 'Root directory'
        
        print(f"\n[Directory] {rel_path}")
        print(f"   Found {len(image_files)} image files (PNG/TIFF)")
        
        renames, clashes = determine_rename_strategy(dirpath)
        
        if clashes:
            print(f"   UNRESOLVED CLASHES in this directory!")
            directories_with_clashes.append(rel_path)
            for clash_key, old_names in clashes.items():
                print(f"      Cannot resolve clash for base name '{clash_key}':")
                for old_name in old_names:
                    print(f"        - {old_name}")
                total_unresolved_clashes += len(old_names)
        
        # Process renames (including those with instance letters)
        rename_count = 0
        for old_name, new_name in renames.items():
            if old_name != new_name:
                old_path = os.path.join(dirpath, old_name)
                new_path = os.path.join(dirpath, new_name)
                
                if not dry_run:
                    try:
                        os.rename(old_path, new_path)
                        print(f"   [OK] Renamed: {old_name} -> {new_name}")
                    except Exception as e:
                        print(f"   [ERROR] Failed to rename {old_name}: {e}")
                else:
                    print(f"   - Would rename: {old_name} -> {new_name}")
                
                rename_count += 1
                total_renames += 1
            else:
                print(f"   - No change needed: {old_name}")
        
        # Show summary for this directory if there were instance letter resolutions
        instance_letter_renames = [n for n in renames.values() if '_' in n and not n.startswith('_')]
        if instance_letter_renames:
            print(f"   Note: {len(instance_letter_renames)} files will include instance letters to avoid clashes")
        
        total_files += len(image_files)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total image files found: {total_files}")
    print(f"Files to be renamed: {total_renames}")
    
    if total_unresolved_clashes > 0:
        print(f"\nWARNING: {total_unresolved_clashes} files have UNRESOLVED naming clashes!")
        print(f"Directories with unresolved clashes ({len(directories_with_clashes)}):")
        for dir_name in directories_with_clashes:
            print(f"  - {dir_name}")
        print("\nThese files have conflicts that cannot be resolved automatically.")
        print("Manual intervention required for these files.")
        return False
    elif total_renames > 0:
        if dry_run:
            print(f"\nNo unresolved clashes detected. Safe to run without --dry-run")
            print("Files with the same image number will include instance letters (A/B/C/D) automatically.")
        else:
            print(f"\nSuccessfully renamed {total_renames} files")
    else:
        print("\nNo files need renaming")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Shorten dataset filenames to just image numbers (with instance letters when needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (default) to check for clashes
  python rename_dataset.py /path/to/dataset
  
  # Actually rename the files
  python rename_dataset.py /path/to/dataset --no-dry-run
  
  # Specify custom path with dry run
  python rename_dataset.py /path/to/dataset --dry-run
  
Notes:
  - Files with unique image numbers: 000.png
  - Files with same image number but different instance letters: 000_A.png, 000_B.png, etc.
  - Unresolved clashes will be reported and not renamed
        """
    )
    
    parser.add_argument(
        'path',
        help='Path to the dataset directory'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Perform a dry run without actually renaming files (default)'
    )
    
    parser.add_argument(
        '--no-dry-run',
        action='store_false',
        dest='dry_run',
        help='Actually rename the files (use with caution)'
    )
    
    args = parser.parse_args()
    
    success = process_directory(args.path, args.dry_run)
    
    if not args.dry_run and success:
        print("\nRenaming complete!")
    elif args.dry_run and success:
        print("\nTip: Run with --no-dry-run to actually rename the files")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())