#!/usr/bin/env python3
"""
Generate JSON representation of directory structure with files in endpoint directories.
"""

import os
import json
import argparse
from pathlib import Path


def get_directory_structure(path, max_depth=None, current_depth=0):
    """
    Recursively build directory structure as nested dictionary.
    Files are only included in directories that have no subdirectories (endpoints).
    
    Args:
        path: Directory path to scan
        max_depth: Maximum depth to traverse (None for unlimited)
        current_depth: Current recursion depth
    
    Returns:
        Dictionary representing directory structure
    """
    path = Path(path)
    
    if not path.is_dir():
        return None
    
    # Stop if we've reached max depth
    if max_depth is not None and current_depth >= max_depth:
        return {"name": path.name, "children": [], "files": []}
    
    children = []
    files = []
    
    try:
        # First, collect all items
        items = list(path.iterdir())
        
        # Get subdirectories
        subdirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
        
        # Recursively process subdirectories
        for subdir in sorted(subdirs):
            child_structure = get_directory_structure(
                subdir, max_depth, current_depth + 1
            )
            if child_structure:
                children.append(child_structure)
        
        # If this is an endpoint directory (no subdirs), collect files
        if not children:
            files = [
                {
                    "name": item.name,
                    "size": item.stat().st_size if item.is_file() else None
                }
                for item in sorted(items) 
                if item.is_file() and not item.name.startswith('.')
            ]
    
    except PermissionError:
        # Skip directories we can't read
        pass
    
    result = {
        "name": path.name,
        "path": str(path),
        "children": children
    }
    
    # Only include files array if this is an endpoint directory
    if not children and files:
        result["files"] = files
    
    return result


def get_directory_structure_all_files(path, max_depth=None, current_depth=0):
    """
    Alternative version that includes files in ALL directories, not just endpoints.
    """
    path = Path(path)
    
    if not path.is_dir():
        return None
    
    if max_depth is not None and current_depth >= max_depth:
        return {"name": path.name, "children": [], "files": []}
    
    children = []
    files = []
    
    try:
        items = list(path.iterdir())
        
        # Get subdirectories
        for item in sorted(items):
            if item.is_dir() and not item.name.startswith('.'):
                child_structure = get_directory_structure_all_files(
                    item, max_depth, current_depth + 1
                )
                if child_structure:
                    children.append(child_structure)
        
        # Get files in this directory
        files = [
            {
                "name": item.name,
                "size": item.stat().st_size
            }
            for item in sorted(items) 
            if item.is_file() and not item.name.startswith('.')
        ]
    
    except PermissionError:
        pass
    
    result = {
        "name": path.name,
        "path": str(path),
        "children": children
    }
    
    if files:
        result["files"] = files
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON directory structure with files"
    )
    parser.add_argument(
        "directory", 
        nargs="?", 
        default=".", 
        help="Directory to scan (default: current directory)"
    )
    parser.add_argument(
        "-d", "--max-depth", 
        type=int, 
        help="Maximum depth to traverse"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--indent", 
        type=int, 
        default=2, 
        help="JSON indentation (default: 2)"
    )
    parser.add_argument(
        "--include-hidden", 
        action="store_true", 
        help="Include hidden directories and files (starting with .)"
    )
    parser.add_argument(
        "--all-files", 
        action="store_true", 
        help="Include files in ALL directories, not just endpoint directories"
    )
    
    args = parser.parse_args()
    
    # Choose the appropriate function based on --all-files flag
    if args.all_files:
        structure = get_directory_structure_all_files(args.directory, args.max_depth)
    else:
        structure = get_directory_structure(args.directory, args.max_depth)
    
    if structure is None:
        print(f"Error: '{args.directory}' is not a valid directory")
        return 1
    
    # Convert to JSON
    json_output = json.dumps(structure, indent=args.indent, ensure_ascii=False)
    
    # Output to file or stdout
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"Directory structure saved to {args.output}")
    else:
        print(json_output)
    
    return 0


if __name__ == "__main__":
    exit(main())