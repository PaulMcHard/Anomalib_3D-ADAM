#!/usr/bin/env python3
"""
Generate JSON representation of directory structure (directories only, no files).
"""

import os
import json
import argparse
from pathlib import Path


def get_directory_structure(path, max_depth=None, current_depth=0):
    """
    Recursively build directory structure as nested dictionary.
    
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
        return {"name": path.name, "children": []}
    
    children = []
    
    try:
        # Get only subdirectories, ignore files
        for item in sorted(path.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):  # Skip hidden dirs
                child_structure = get_directory_structure(
                    item, max_depth, current_depth + 1
                )
                if child_structure:
                    children.append(child_structure)
    except PermissionError:
        # Skip directories we can't read
        pass
    
    return {
        "name": path.name,
        "path": str(path),
        "children": children
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON directory structure (directories only)"
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
        help="Include hidden directories (starting with .)"
    )
    
    args = parser.parse_args()
    
    # Get directory structure
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