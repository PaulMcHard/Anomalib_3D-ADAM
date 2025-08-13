#!/usr/bin/env python3
"""
Dataset Reorganization Script

Reorganizes a dataset from train/test/validation split structure to a flattened structure
where 'good' samples and defect types are at the same level under each category.
"""

import os
import shutil
from pathlib import Path
import argparse
from typing import List, Tuple


def find_categories(source_dir: Path) -> List[Path]:
    """Find all category directories in the source directory."""
    categories = []
    for item in source_dir.iterdir():
        if item.is_dir():
            # Check if it has train/test/validation subdirectories
            if any((item / split).exists() for split in ['train', 'test', 'validation']):
                categories.append(item)
    return categories


def reorganize_category(source_category: Path, target_category: Path, verbose: bool = True) -> Tuple[int, int]:
    """
    Reorganize a single category from source to target.
    
    Returns:
        Tuple of (files_copied, directories_created)
    """
    files_copied = 0
    dirs_created = 0
    
    if verbose:
        print(f"\nProcessing category: {source_category.name}")
    
    # Create target category directory
    target_category.mkdir(parents=True, exist_ok=True)
    dirs_created += 1
    
    # Process train folder - move its contents to a new 'good' folder
    train_path = source_category / 'train'
    if train_path.exists():
        target_good_path = target_category / 'good'
        if verbose:
            print(f"  Copying train/ -> good/")
        fc, dc = copy_directory_contents(train_path, target_good_path, verbose=False)
        files_copied += fc
        dirs_created += dc
    
    # Process test folder (contains defect types)
    test_path = source_category / 'test'
    if test_path.exists():
        for defect_dir in test_path.iterdir():
            if defect_dir.is_dir():
                target_defect_path = target_category / defect_dir.name
                if verbose:
                    print(f"  Copying test/{defect_dir.name} -> {defect_dir.name}/")
                fc, dc = copy_directory_contents(defect_dir, target_defect_path, verbose=False)
                files_copied += fc
                dirs_created += dc
    
    # Process validation folder (contains defect types)
    validation_path = source_category / 'validation'
    if validation_path.exists():
        for defect_dir in validation_path.iterdir():
            if defect_dir.is_dir():
                target_defect_path = target_category / defect_dir.name
                if verbose:
                    print(f"  Copying validation/{defect_dir.name} -> {defect_dir.name}/")
                fc, dc = copy_directory_contents(defect_dir, target_defect_path, verbose=False)
                files_copied += fc
                dirs_created += dc
    
    return files_copied, dirs_created


def copy_directory_contents(source: Path, target: Path, verbose: bool = True) -> Tuple[int, int]:
    """
    Copy all contents from source directory to target directory, preserving structure.
    
    Returns:
        Tuple of (files_copied, directories_created)
    """
    files_copied = 0
    dirs_created = 0
    
    # Create target directory if it doesn't exist
    target.mkdir(parents=True, exist_ok=True)
    dirs_created += 1
    
    # Walk through source directory
    for root, dirs, files in os.walk(source):
        # Convert to Path objects
        root_path = Path(root)
        
        # Calculate relative path from source
        rel_path = root_path.relative_to(source)
        
        # Create corresponding directory in target
        target_dir = target / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)
        if rel_path != Path('.'):
            dirs_created += 1
        
        # Copy all files
        for file in files:
            source_file = root_path / file
            target_file = target_dir / file
            
            if not target_file.exists():
                shutil.copy2(source_file, target_file)
                files_copied += 1
                if verbose:
                    print(f"    Copied: {source_file.relative_to(source.parent)}")
            elif verbose:
                print(f"    Skipped (exists): {target_file.relative_to(target.parent)}")
    
    return files_copied, dirs_created


def show_structure_preview(source_dir: Path, categories: List[Path]):
    """Show a preview of the current structure and what will be created."""
    print("\n" + "="*60)
    print("STRUCTURE PREVIEW")
    print("="*60)
    
    for category in categories[:1]:  # Show first category as example
        print(f"\nExample for category: {category.name}")
        print("\nCurrent structure:")
        
        # Show train structure
        train_path = category / 'train'
        if train_path.exists():
            print(f"  {category.name}/train/")
            for item in list(train_path.iterdir())[:3]:
                if item.is_dir():
                    print(f"    ├── {item.name}/")
            if len(list(train_path.iterdir())) > 3:
                print(f"    └── ...")
        
        # Show test structure
        test_path = category / 'test'
        if test_path.exists():
            print(f"  {category.name}/test/")
            for item in list(test_path.iterdir())[:3]:
                if item.is_dir():
                    print(f"    ├── {item.name}/")
                    # Show subdirs
                    subdirs = [d.name for d in item.iterdir() if d.is_dir()][:3]
                    for subdir in subdirs:
                        print(f"        ├── {subdir}/")
            if len(list(test_path.iterdir())) > 3:
                print(f"    └── ...")
        
        print("\nWill be reorganized to:")
        print(f"  {category.name}/")
        print(f"    ├── good/  (from train/)")
        
        # Show what subdirs will be in good
        if train_path.exists():
            subdirs = [d.name for d in train_path.iterdir() if d.is_dir()][:3]
            for subdir in subdirs:
                print(f"        ├── {subdir}/")
            if len(subdirs) > 3:
                print(f"        └── ...")
        
        # Show defect types
        if test_path.exists():
            for item in list(test_path.iterdir())[:3]:
                if item.is_dir():
                    print(f"    ├── {item.name}/  (from test/{item.name}/)")
        
        validation_path = category / 'validation'
        if validation_path.exists():
            for item in list(validation_path.iterdir())[:2]:
                if item.is_dir():
                    print(f"    ├── {item.name}/  (from validation/{item.name}/)")
        
        print(f"    └── ...")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Reorganize dataset from train/test/validation split to flattened structure'
    )
    parser.add_argument(
        'source',
        type=str,
        help='Source directory containing the dataset'
    )
    parser.add_argument(
        'target',
        type=str,
        help='Target directory where reorganized dataset will be created'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress information'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='Skip the structure preview'
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source).resolve()
    target_dir = Path(args.target).resolve()
    
    # Validate source directory
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist")
        return 1
    
    if not source_dir.is_dir():
        print(f"Error: Source path '{source_dir}' is not a directory")
        return 1
    
    # Check if target exists and warn
    if target_dir.exists() and not args.dry_run:
        response = input(f"Target directory '{target_dir}' already exists. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
    
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    
    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be copied ***")
    
    # Find all categories
    categories = find_categories(source_dir)
    
    if not categories:
        print("No categories with train/test/validation structure found.")
        return 1
    
    print(f"\nFound {len(categories)} categories to process:")
    for cat in categories:
        print(f"  - {cat.name}")
    
    # Show structure preview unless disabled
    if not args.no_preview:
        show_structure_preview(source_dir, categories)
    
    if not args.dry_run:
        response = input("\nProceed with reorganization? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
    
    total_files = 0
    total_dirs = 0
    
    # Process each category
    for category in categories:
        target_category = target_dir / category.name
        
        if not args.dry_run:
            files, dirs = reorganize_category(category, target_category, verbose=args.verbose)
            total_files += files
            total_dirs += dirs
        else:
            print(f"\nWould process category: {category.name}")
            # Show what would be done
            train_path = category / 'train'
            if train_path.exists():
                print(f"  Would copy: train/ -> good/")
                subdirs = [d.name for d in train_path.iterdir() if d.is_dir()]
                if subdirs:
                    print(f"    (contains: {', '.join(subdirs[:5])}{', ...' if len(subdirs) > 5 else ''})")
            
            test_path = category / 'test'
            if test_path.exists():
                for defect in test_path.iterdir():
                    if defect.is_dir():
                        print(f"  Would copy: test/{defect.name}/ -> {defect.name}/")
            
            val_path = category / 'validation'
            if val_path.exists():
                for defect in val_path.iterdir():
                    if defect.is_dir():
                        print(f"  Would copy: validation/{defect.name}/ -> {defect.name}/")
    
    if not args.dry_run:
        print(f"\n{'='*50}")
        print(f"Reorganization complete!")
        print(f"  Total directories created: {total_dirs}")
        print(f"  Total files copied: {total_files}")
    else:
        print(f"\n{'='*50}")
        print("Dry run complete. No files were copied.")
    
    return 0


if __name__ == '__main__':
    exit(main())