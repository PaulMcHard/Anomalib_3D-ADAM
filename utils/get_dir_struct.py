#!/usr/bin/env python3
import os
import argparse
import json

def get_dir_structure(root_dir):
    """
    Recursively generates a dictionary representing the directory structure,
    ignoring files ending in .metadata and .cache directories.

    Args:
        root_dir (str): The path to the root directory.

    Returns:
        dict: A dictionary representing the file and directory structure.
    """
    dir_structure = {}
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            # Ignore .cache directories
            if item != '.cache':
                dir_structure[item] = get_dir_structure(item_path)
        else:
            # Ignore files ending with .metadata
            if not item.endswith('.metadata'):
                if item not in dir_structure:
                    dir_structure[item] = None
    return dir_structure

def main():
    """
    Main function to parse command-line arguments and generate the JSON output.
    """
    parser = argparse.ArgumentParser(description="Generate a JSON representation of a directory structure.")
    parser.add_argument("source_dir", type=str, help="Path to the directory to scan.")
    parser.add_argument("output_file", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()

    # Check if the source directory exists
    if not os.path.isdir(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' not found.")
        return

    print(f"Generating directory structure for '{args.source_dir}'...")

    # Get the directory structure
    structure = {
        os.path.basename(args.source_dir): get_dir_structure(args.source_dir)
    }

    # Write the structure to a JSON file
    with open(args.output_file, 'w') as f:
        json.dump(structure, f, indent=4)

    print(f"Directory structure successfully written to '{args.output_file}'.")

if __name__ == "__main__":
    main()