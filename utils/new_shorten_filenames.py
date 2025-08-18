#!/usr/bin/env python3
"""
Script to shorten dataset filenames from long format to just image numbers.
Converts: 
  - Any file following PartName_InstanceLetter_...ImageNumber... pattern
  - To: ImageNumber.ext (when no clash) or ImageNumber_InstanceLetter.ext (when clash exists)
  
Uses the same naming convention logic established for mask consolidation.
"""

import os
import re
import argparse
from pathlib import Path
from collections import defaultdict

def extract_grouping_key(filename):
    """
    Extract the grouping key (PartName_InstanceLetter and ImageNumber) from the filename.
    Reuses the same logic from the mask consolidation script.
    
    Examples:
    - Tapa2M1_C_Cut_MechMind-Nano_Tapa2M1_C_Cut_Nano_009_image_cut_defect_2.png -> ('Tapa2M1_C', '009')
    - SpurGear_D_Warped_MechMind-Nano_SpurGear_D_Warped_Nano_006_warped_defect_1.png -> ('SpurGear_D', '006')
    - Gripper_Closed_C_Hole_MechMind-Nano_Gripper_Closed_C_Hole_Nano_000_image_hole_defect_1.png -> ('Gripper_Closed_C', '000')
    """
    # Get file extension
    if '.' in filename:
        name_without_ext = filename.rsplit('.', 1)[0]
        extension = '.' + filename.rsplit('.', 1)[1]
    else:
        name_without_ext = filename
        extension = ''
    
    # Look for all 3-digit numbers (potential image numbers)
    image_number_pattern = re.compile(r'\d{3}')
    image_number_matches = image_number_pattern.findall(name_without_ext)
    
    if not image_number_matches:
        return None
    
    # Use the last 3-digit number as the image number (most reliable)
    image_number = image_number_matches[-1]
    
    # Split by underscores
    parts = name_without_ext.split('_')
    
    if len(parts) < 3:
        return None
    
    # Strategy: Look for single uppercase letter that appears to be an instance letter
    # The instance letter should be a single uppercase letter (A, B, C, D, etc.)
    # and it should appear early in the filename structure
    
    for i in range(1, min(len(parts), 5)):  # Check first few positions for instance letter
        if len(parts[i]) == 1 and parts[i].isupper() and parts[i].isalpha():
            # Found potential instance letter, construct part name
            part_name_components = parts[:i+1]  # Include everything up to and including the instance letter
            part_name_instance = '_'.join(part_name_components)
            return (part_name_instance, image_number, extension)
    
    # Fallback: if no single letter found in expected positions, 
    # try the original logic (for backwards compatibility)
    if len(parts) >= 2 and len(parts[1]) == 1 and parts[1].isupper() and parts[1].isalpha():
        part_name_instance = f"{parts[0]}_{parts[1]}"
        return (part_name_instance, image_number, extension)
    
    return None

def determine_rename_strategy(directory_path):
    """
    Determine the best renaming strategy for files in a directory.
    Returns a dictionary mapping old filenames to new filenames.
    """
    # First pass: collect all files and their components
    file_info = {}
    image_number_groups = defaultdict(list)
    
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if not os.path.isfile(filepath):
            continue
        
        grouping_result = extract_grouping_key(filename)
        if grouping_result is None:
            print(f"  Warning: Cannot parse filename pattern: {filename}")
            continue
        
        part_name_instance, image_number, extension = grouping_result
        instance_letter = part_name_instance.split('_')[-1]  # Last part should be instance letter
        
        file_info[filename] = {
            'part_name_instance': part_name_instance,
            'image_number': image_number,
            'instance_letter': instance_letter,
            'extension': extension
        }
        
        # Group files by image number and extension for clash detection
        key = (image_number, extension)
        image_number_groups[key].append(filename)
    
    # Second pass: determine naming strategy
    renames = {}
    unresolved_clashes = []
    
    for (image_number, extension), filenames in image_number_groups.items():
        if len(filenames) == 1:
            # No clash, use simple naming
            filename = filenames[0]
            new_name = f"{image_number}{extension}"
            renames[filename] = new_name
        else:
            # Multiple files with same image number and extension - need to check instance letters
            instance_letters = []
            files_by_instance = {}
            
            for filename in filenames:
                info = file_info[filename]
                instance_letter = info['instance_letter']
                instance_letters.append(instance_letter)
                
                if instance_letter in files_by_instance:
                    # Duplicate instance letter - this is a true clash we can't resolve
                    unresolved_clashes.extend(filenames)
                    break
                files_by_instance[instance_letter] = filename
            else:
                # All instance letters are unique, we can resolve this clash
                for filename in filenames:
                    info = file_info[filename]
                    new_name = f"{info['image_number']}_{info['instance_letter']}{extension}"
                    renames[filename] = new_name
    
    # Remove any files that had unresolved clashes from the rename list
    for filename in unresolved_clashes:
        if filename in renames:
            del renames[filename]
    
    return renames, unresolved_clashes

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
    
    print(f"{'DRY RUN: ' if dry_run else ''}Processing directory tree: {root_path}")
    print("=" * 80)
    
    # Process root directory and all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter for files that aren't already in short format
        candidate_files = []
        for filename in filenames:
            # Skip files that already look like they're in short format (just numbers)
            if re.match(r'^\d{3}(_[A-Z])?\.[^.]+$', filename):
                continue
            candidate_files.append(filename)
        
        if not candidate_files:
            continue
        
        rel_path = os.path.relpath(dirpath, root_path)
        if rel_path == '.':
            rel_path = 'Root directory'
        
        print(f"\n[Directory] {rel_path}")
        print(f"   Found {len(candidate_files)} files to potentially rename")
        
        renames, unresolved_clashes = determine_rename_strategy(dirpath)
        
        if unresolved_clashes:
            print(f"   ⚠️  UNRESOLVED CLASHES in this directory!")
            directories_with_clashes.append(rel_path)
            clash_groups = defaultdict(list)
            
            # Group clashes by what they would conflict with
            for filename in unresolved_clashes:
                if filename in renames:
                    continue  # This shouldn't happen, but just in case
                
                grouping_result = extract_grouping_key(filename)
                if grouping_result:
                    _, image_number, extension = grouping_result
                    clash_key = f"{image_number}{extension}"
                    clash_groups[clash_key].append(filename)
            
            for clash_key, clash_files in clash_groups.items():
                print(f"      Cannot resolve clash for base name '{clash_key}':")
                for clash_file in clash_files:
                    print(f"        - {clash_file}")
                total_unresolved_clashes += len(clash_files)
        
        # Process renames
        simple_renames = 0
        instance_renames = 0
        
        for old_name, new_name in renames.items():
            if old_name == new_name:
                continue
                
            old_path = os.path.join(dirpath, old_name)
            new_path = os.path.join(dirpath, new_name)
            
            # Check if target file already exists
            if os.path.exists(new_path):
                print(f"   ⚠️  Cannot rename {old_name} -> {new_name} (target exists)")
                continue
            
            if not dry_run:
                try:
                    os.rename(old_path, new_path)
                    print(f"   ✅ Renamed: {old_name} -> {new_name}")
                except Exception as e:
                    print(f"   ❌ Failed to rename {old_name}: {e}")
                    continue
            else:
                print(f"   📝 Would rename: {old_name} -> {new_name}")
            
            if '_' in new_name and not new_name.startswith('_'):
                instance_renames += 1
            else:
                simple_renames += 1
            
            total_renames += 1
        
        # Show summary for this directory
        if simple_renames > 0:
            print(f"   📊 {simple_renames} files renamed to simple format (XXX.ext)")
        if instance_renames > 0:
            print(f"   📊 {instance_renames} files renamed with instance letters (XXX_Y.ext)")
        
        total_files += len(candidate_files)
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total candidate files found: {total_files}")
    print(f"Files {'to be renamed' if dry_run else 'successfully renamed'}: {total_renames}")
    
    if total_unresolved_clashes > 0:
        print(f"\n⚠️  WARNING: {total_unresolved_clashes} files have UNRESOLVED naming clashes!")
        print(f"Directories with unresolved clashes ({len(directories_with_clashes)}):")
        for dir_name in directories_with_clashes:
            print(f"  - {dir_name}")
        print("\n💡 These files have the same image number AND instance letter.")
        print("   Manual intervention required - consider using different instance letters.")
        return False
    elif total_renames > 0:
        if dry_run:
            print(f"\n✅ No unresolved clashes detected. Safe to run without --dry-run")
            print("💡 Files with the same image number will automatically include instance letters.")
        else:
            print(f"\n🎉 Successfully renamed {total_renames} files")
    else:
        print("\n📝 No files need renaming (all files already in correct format or no parseable files found)")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Shorten dataset filenames to just image numbers (with instance letters when needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (default) to check what would be renamed
  python shorten_filenames.py /path/to/dataset
  
  # Actually rename the files
  python shorten_filenames.py /path/to/dataset --no-dry-run
  
Renaming Logic:
  1. Extract image number and instance letter from complex filenames
  2. If no clash: rename to just ImageNumber.ext (e.g., 009.png)
  3. If clash detected: rename to ImageNumber_InstanceLetter.ext (e.g., 009_A.png, 009_B.png)
  4. Files already in short format are skipped
  5. Files with unresolvable clashes (same number + same instance) are reported but not renamed

Supported Filename Patterns:
  - PartName_InstanceLetter_...ImageNumber... (any extension)
  - Gripper_Closed_C_...000... -> 000_C.ext (if clash) or 000.ext (if no clash)
  - Tapa2M1_A_...005... -> 005_A.ext (if clash) or 005.ext (if no clash)
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
        print("\n🎉 Renaming complete!")
    elif args.dry_run and success:
        print("\n💡 Tip: Run with --no-dry-run to actually rename the files")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())