#!/usr/bin/env python3
"""
Script to split the adam3d__masked_blob dataset into train/test sets following MVTec structure.
Takes the source dataset and creates a properly split version in the datasets/ directory.

Two modes available:
1. SUPERVISED (default): Splits by ANOMALY TYPE to prevent contamination and test generalization
2. UNSUPERVISED (--unsupervised): Train on normal samples only, test on everything

Both modes follow MVTec-AD structure with separate ground_truth directory.

Usage:
    # Supervised mode (anomaly-type-level separation)
    python split_adam3d_dataset.py --source "D:\Data\3d-adam-tests\adam3d__masked_blob" --output "datasets\adam3d_split"
    
    # Unsupervised mode (normal samples only in train)
    python split_adam3d_dataset.py --source "D:\Data\3d-adam-tests\adam3d__masked_blob" --output "datasets\adam3d_unsupervised" --unsupervised
"""

import os
import shutil
import argparse
from pathlib import Path
import random
from typing import List, Tuple, Dict
import json

def get_anomaly_types_in_category(category_path: Path) -> List[str]:
    """Get all anomaly types (directories) within a category."""
    if not category_path.exists():
        return []
    
    return [d.name for d in category_path.iterdir() 
            if d.is_dir() and (d / "rgb").exists()]

def copy_anomaly_type_unsupervised(source_anomaly_path: Path, dest_category_path: Path, 
                                  anomaly_type: str, split_name: str, train_good_ratio: float = 0.8, 
                                  seed: int = 42) -> int:
    """Copy anomaly type for unsupervised mode: ALL good samples in train, defects in test only."""
    
    if anomaly_type.lower() == "good":
        # For good samples in unsupervised mode: ALL go to train only
        if split_name != "train":
            return 0  # No good samples in test for unsupervised mode
            
        source_rgb = source_anomaly_path / "rgb"
        if not source_rgb.exists():
            return 0
        
        # Copy ALL good samples to train
        dest_split_path = dest_category_path / split_name / "good"
        dest_split_path.mkdir(parents=True, exist_ok=True)
        
        file_count = 0
        for rgb_file in source_rgb.iterdir():
            if rgb_file.is_file() and rgb_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                shutil.copy2(rgb_file, dest_split_path / rgb_file.name)
                file_count += 1
        
        print(f"    {anomaly_type} -> {split_name}: {file_count} files (ALL good samples)")
        return file_count
    
    else:
        # For defect types in unsupervised mode: all go to test only
        if split_name != "test":
            return 0  # Defects only in test
        
        dest_split_path = dest_category_path / split_name / anomaly_type
        dest_split_path.mkdir(parents=True, exist_ok=True)
        
        # Copy RGB images
        source_rgb = source_anomaly_path / "rgb"
        file_count = 0
        
        if source_rgb.exists():
            for rgb_file in source_rgb.iterdir():
                if rgb_file.is_file() and rgb_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    shutil.copy2(rgb_file, dest_split_path / rgb_file.name)
                    file_count += 1
        
        # Copy corresponding binary masks to ground_truth directory
        source_masks = source_anomaly_path / "ground_truth"
        dest_gt_path = dest_category_path / "ground_truth" / anomaly_type
        dest_gt_path.mkdir(parents=True, exist_ok=True)
        
        gt_count = 0
        if source_masks.exists():
            for mask_file in source_masks.iterdir():
                if mask_file.is_file() and mask_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    shutil.copy2(mask_file, dest_gt_path / mask_file.name)
                    gt_count += 1
        
        print(f"    {anomaly_type} -> {split_name}: {file_count} files + {gt_count} ground truth masks")
        return file_count

def copy_anomaly_type(source_anomaly_path: Path, dest_category_path: Path, anomaly_type: str, split_name: str) -> int:
    """Copy anomaly type following MVTec structure: separate ground_truth directory at category level."""
    
    # For 'good' samples: copy to train/test split directory
    if anomaly_type.lower() == "good":
        dest_split_path = dest_category_path / split_name / "good"
        dest_split_path.mkdir(parents=True, exist_ok=True)
        
        # Copy RGB images
        source_rgb = source_anomaly_path / "rgb"
        file_count = 0
        
        if source_rgb.exists():
            for rgb_file in source_rgb.iterdir():
                if rgb_file.is_file() and rgb_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    shutil.copy2(rgb_file, dest_split_path / rgb_file.name)
                    file_count += 1
        
        print(f"    {anomaly_type} -> {split_name}: {file_count} files")
        return file_count
    
    # For defect types: copy to test split directory + create ground_truth
    else:
        dest_split_path = dest_category_path / split_name / anomaly_type
        dest_split_path.mkdir(parents=True, exist_ok=True)
        
        # Copy RGB images
        source_rgb = source_anomaly_path / "rgb"
        file_count = 0
        
        if source_rgb.exists():
            for rgb_file in source_rgb.iterdir():
                if rgb_file.is_file() and rgb_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    shutil.copy2(rgb_file, dest_split_path / rgb_file.name)
                    file_count += 1
        
        # Copy corresponding binary masks to ground_truth directory
        source_masks = source_anomaly_path / "ground_truth"
        dest_gt_path = dest_category_path / "ground_truth" / anomaly_type
        dest_gt_path.mkdir(parents=True, exist_ok=True)
        
        gt_count = 0
        if source_masks.exists():
            for mask_file in source_masks.iterdir():
                if mask_file.is_file() and mask_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    shutil.copy2(mask_file, dest_gt_path / mask_file.name)
                    gt_count += 1
        
        print(f"    {anomaly_type} -> {split_name}: {file_count} files + {gt_count} ground truth masks")
        return file_count

def split_anomaly_types_unsupervised(anomaly_types: List[str], train_good_ratio: float = 0.8, 
                                    seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split for unsupervised anomaly detection: train on normal samples only.
    - Train gets ALL 'good' samples only
    - Test gets ALL defect types only (no good samples)
    """
    random.seed(seed)
    
    good_types = [t for t in anomaly_types if t.lower() == "good"]
    defect_types = [t for t in anomaly_types if t.lower() != "good"]
    
    # For unsupervised: ALL good samples in train, ONLY defects in test
    train_types = good_types if good_types else []
    test_types = defect_types  # Only defect types in test
    
    return train_types, test_types

def split_anomaly_types_mvtec_style(anomaly_types: List[str], train_ratio: float = 0.8, 
                                   seed: int = 42) -> Tuple[List[str], List[str]]:
    """Split anomaly types into train/test sets following MVTec structure.
    - Maintains anomaly-type-level separation to prevent contamination
    - No validation split (just train/test like MVTec)
    """
    random.seed(seed)
    
    # Special handling for categories with limited anomaly types
    if len(anomaly_types) == 1:
        # Only one anomaly type (e.g., only 'good'), put it in train
        return anomaly_types, []
    elif len(anomaly_types) == 2:
        # Two anomaly types: one to train, one to test
        shuffled = anomaly_types.copy()
        random.shuffle(shuffled)
        return [shuffled[0]], [shuffled[1]]
    else:
        # Three or more anomaly types: split according to ratio
        shuffled = anomaly_types.copy()
        random.shuffle(shuffled)
        
        total_types = len(shuffled)
        train_end = max(1, int(total_types * train_ratio))
        
        train_types = shuffled[:train_end]
        test_types = shuffled[train_end:]
        
        # Ensure test set has at least one type if there are enough types
        if len(test_types) == 0 and total_types >= 2:
            test_types = [train_types.pop()]
        
        return train_types, test_types

def process_category(source_category_path: Path, output_category_path: Path, 
                    category_name: str, train_ratio: float = 0.8, 
                    seed: int = 42, unsupervised: bool = False) -> dict:
    """Process a single category and split anomaly types into train/test."""
    print(f"\n=== Processing category: {category_name} ===")
    
    stats = {
        "category": category_name,
        "splits": {"train": {}, "test": {}},
        "anomaly_type_assignments": {"train": [], "test": []}
    }
    
    # Get all anomaly types (directories) in this category
    anomaly_types = get_anomaly_types_in_category(source_category_path)
    
    if not anomaly_types:
        print(f"  No anomaly types found in {source_category_path}")
        return stats
    
    print(f"  Found anomaly types: {anomaly_types}")
    
    # Split anomaly types based on mode
    if unsupervised:
        # Unsupervised: train on good only, test on everything
        train_types, test_types = split_anomaly_types_unsupervised(
            anomaly_types, train_ratio, seed
        )
        print(f"  Unsupervised split assignment:")
    else:
        # Supervised: split anomaly types into train/test (no validation like MVTec)
        train_types, test_types = split_anomaly_types_mvtec_style(
            anomaly_types, train_ratio, seed
        )
        print(f"  Supervised split assignment:")
    
    print(f"    Train: {train_types}")
    print(f"    Test: {test_types}")
    
    # Store anomaly type assignments
    stats["anomaly_type_assignments"]["train"] = train_types
    stats["anomaly_type_assignments"]["test"] = test_types
    
    # Copy anomaly types to their assigned splits
    for split_name, assigned_types in [("train", train_types), ("test", test_types)]:
        for anomaly_type in assigned_types:
            source_anomaly_path = source_category_path / anomaly_type
            
            if unsupervised:
                file_count = copy_anomaly_type_unsupervised(
                    source_anomaly_path, output_category_path, anomaly_type, split_name, train_ratio, seed
                )
            else:
                file_count = copy_anomaly_type(
                    source_anomaly_path, output_category_path, anomaly_type, split_name
                )
            
            stats["splits"][split_name][anomaly_type] = file_count
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Split adam3d dataset by anomaly types following MVTec structure")
    parser.add_argument("--source", type=str, required=True, 
                       help="Path to source dataset directory")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to output directory (relative to current directory)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio of anomaly types for training set (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")
    parser.add_argument("--unsupervised", action="store_true",
                       help="Structure for unsupervised learning: ALL normal samples in train, defects only in test")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually copying files")
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    output_path = Path(args.output)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_path}' does not exist")
        return
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be copied")
        print(f"Would split dataset from: {source_path}")
        print(f"Would save split dataset to: {output_path.absolute()}")
        if args.unsupervised:
            print(f"Mode: UNSUPERVISED (train on ALL normal samples, test on defects only)")
            print(f"Train: ALL good samples, Test: defect types only")
        else:
            print(f"Mode: SUPERVISED (anomaly-type-level separation)")
            print(f"Train ratio: {args.train_ratio} (test ratio: {1-args.train_ratio})")
        print(f"Random seed: {args.seed}")
        print("Structure: MVTec-like with separate ground_truth directory")
        return
    
    print(f"Splitting dataset from: {source_path}")
    print(f"Saving split dataset to: {output_path.absolute()}")
    if args.unsupervised:
        print(f"Mode: UNSUPERVISED (train on ALL normal samples, test on defects only)")
        print(f"Train: ALL good samples, Test: defect types only")
    else:
        print(f"Mode: SUPERVISED (anomaly-type-level separation)")
        print(f"Train ratio: {args.train_ratio} (test ratio: {1-args.train_ratio})")
    print(f"Random seed: {args.seed}")
    print("Structure: MVTec-like with separate ground_truth directory")
    print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each category
    all_stats = []
    categories = [d for d in source_path.iterdir() if d.is_dir()]
    
    for category_dir in categories:
        category_name = category_dir.name
        output_category_path = output_path / category_name
        
        try:
            stats = process_category(
                category_dir, output_category_path, category_name,
                args.train_ratio, args.seed, args.unsupervised
            )
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing category {category_name}: {e}")
    
    # Save statistics
    stats_file = output_path / "split_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "split_parameters": {
                "train_ratio": args.train_ratio,
                "test_ratio": 1 - args.train_ratio,
                "seed": args.seed,
                "unsupervised": args.unsupervised,
                "source_path": str(source_path),
                "output_path": str(output_path.absolute()),
                "split_method": "unsupervised_mode" if args.unsupervised else "anomaly_type_based_mvtec_structure"
            },
            "category_stats": all_stats
        }, f, indent=2)
    
    print(f"\n✓ Dataset splitting complete!")
    print(f"✓ Statistics saved to: {stats_file}")
    print(f"✓ Split dataset available at: {output_path.absolute()}")
    
    # Print summary
    total_files = 0
    total_categories = len([s for s in all_stats if s["splits"]["train"] or s["splits"]["test"]])
    for stats in all_stats:
        for split in stats["splits"].values():
            for count in split.values():
                total_files += count
    
    print(f"\nSummary:")
    print(f"  Total categories processed: {total_categories}")
    print(f"  Total files copied: {total_files}")
    print(f"  Mode: {'Unsupervised' if args.unsupervised else 'Supervised'}")
    print(f"  Structure: MVTec-like with separate ground_truth directory")
    if args.unsupervised:
        print(f"  Training: ALL normal samples only")
        print(f"  Testing: Defect types only (no normal samples)")
    else:
        print(f"  No contamination: Each anomaly type appears in only one split")

if __name__ == "__main__":
    main()