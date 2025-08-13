#!/usr/bin/env python3
"""
JSON Compression Tool

A standalone utility for compressing and decompressing JSON files
using gzip compression with base64 encoding for safe text storage.
"""

import json
import argparse
import sys
import os
import gzip
import base64
from pathlib import Path
from typing import Dict, Union

def compress_json(data: Union[Dict, str], compression_level: int = 6) -> str:
    """
    Compress JSON data using gzip and encode as base64.
    
    Args:
        data: Dictionary or JSON string to compress
        compression_level: Gzip compression level (1-9)
    
    Returns:
        Base64 encoded compressed data
    """
    if isinstance(data, dict):
        json_str = json.dumps(data, separators=(',', ':'))
    else:
        json_str = str(data)
    
    compressed = gzip.compress(json_str.encode('utf-8'), compresslevel=compression_level)
    return base64.b64encode(compressed).decode('ascii')

def decompress_json(compressed_data: str) -> Dict:
    """
    Decompress base64 encoded gzip JSON data.
    
    Args:
        compressed_data: Base64 encoded compressed JSON string
    
    Returns:
        Decompressed JSON data as dictionary
    """
    compressed = base64.b64decode(compressed_data.encode('ascii'))
    json_str = gzip.decompress(compressed).decode('utf-8')
    return json.loads(json_str)

def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def validate_json_file(file_path: str) -> bool:
    """Validate that a file contains valid JSON."""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError, IOError):
        return False

def compress_file(input_file: str, output_file: str = None, compression_level: int = 6, 
                 show_stats: bool = False) -> str:
    """
    Compress a JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output file (optional)
        compression_level: Gzip compression level (1-9)
        show_stats: Whether to print compression statistics
    
    Returns:
        Path to output file
    """
    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not validate_json_file(input_file):
        raise ValueError(f"Invalid JSON file: {input_file}")
    
    # Read and compress
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    compressed = compress_json(data, compression_level)
    
    # Determine output file
    if output_file is None:
        output_file = input_file + '.compressed'
    
    # Write compressed data
    with open(output_file, 'w') as f:
        f.write(compressed)
    
    # Show statistics if requested
    if show_stats:
        original_size = os.path.getsize(input_file)
        compressed_size = len(compressed.encode('utf-8'))
        ratio = (1 - compressed_size / original_size) * 100
        
        print(f"Input file: {input_file}", file=sys.stderr)
        print(f"Output file: {output_file}", file=sys.stderr)
        print(f"Original size: {format_size(original_size)}", file=sys.stderr)
        print(f"Compressed size: {format_size(compressed_size)}", file=sys.stderr)
        print(f"Compression ratio: {ratio:.1f}%", file=sys.stderr)
        print(f"Space saved: {format_size(original_size - compressed_size)}", file=sys.stderr)
    
    return output_file

def decompress_file(input_file: str, output_file: str = None, pretty: bool = False) -> str:
    """
    Decompress a compressed JSON file.
    
    Args:
        input_file: Path to compressed file
        output_file: Path to output file (optional, uses stdout if None)
        pretty: Whether to pretty-print the JSON
    
    Returns:
        Path to output file or None if stdout
    """
    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read and decompress
    with open(input_file, 'r') as f:
        compressed_data = f.read().strip()
    
    try:
        decompressed = decompress_json(compressed_data)
    except Exception as e:
        raise ValueError(f"Failed to decompress file (not a valid compressed JSON?): {e}")
    
    # Format output
    if pretty:
        output = json.dumps(decompressed, indent=2, ensure_ascii=False)
    else:
        output = json.dumps(decompressed, separators=(',', ':'), ensure_ascii=False)
    
    # Write output
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        return output_file
    else:
        print(output)
        return None

def batch_compress(input_pattern: str, compression_level: int = 6, 
                  output_suffix: str = '.compressed', show_stats: bool = False):
    """
    Compress multiple JSON files matching a pattern.
    
    Args:
        input_pattern: File pattern (e.g., "*.json")
        compression_level: Gzip compression level
        output_suffix: Suffix to add to compressed files
        show_stats: Whether to show statistics for each file
    """
    from glob import glob
    
    files = glob(input_pattern)
    if not files:
        print(f"No files found matching pattern: {input_pattern}", file=sys.stderr)
        return
    
    total_original = 0
    total_compressed = 0
    successful = 0
    
    for file_path in files:
        try:
            if validate_json_file(file_path):
                output_file = file_path + output_suffix
                compress_file(file_path, output_file, compression_level, show_stats)
                
                if show_stats:
                    original_size = os.path.getsize(file_path)
                    compressed_size = os.path.getsize(output_file)
                    total_original += original_size
                    total_compressed += compressed_size
                    print(f"✓ Compressed: {file_path}", file=sys.stderr)
                
                successful += 1
            else:
                print(f"✗ Skipped (invalid JSON): {file_path}", file=sys.stderr)
        except Exception as e:
            print(f"✗ Error compressing {file_path}: {e}", file=sys.stderr)
    
    if show_stats and successful > 0:
        total_ratio = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0
        print(f"\nBatch compression summary:", file=sys.stderr)
        print(f"Files processed: {successful}/{len(files)}", file=sys.stderr)
        print(f"Total original size: {format_size(total_original)}", file=sys.stderr)
        print(f"Total compressed size: {format_size(total_compressed)}", file=sys.stderr)
        print(f"Overall compression ratio: {total_ratio:.1f}%", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description='Compress and decompress JSON files using gzip + base64 encoding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress a single file
  %(prog)s compress data.json

  # Compress with custom output and compression level
  %(prog)s compress data.json -o compressed_data.txt --level 9

  # Decompress to stdout with pretty formatting
  %(prog)s decompress compressed_data.txt --pretty

  # Decompress to file
  %(prog)s decompress compressed_data.txt -o restored.json

  # Batch compress all JSON files
  %(prog)s batch "*.json" --stats

  # Check if a file is compressed JSON
  %(prog)s validate compressed_data.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Compress command
    compress_parser = subparsers.add_parser('compress', help='Compress a JSON file')
    compress_parser.add_argument('input', help='Input JSON file to compress')
    compress_parser.add_argument('-o', '--output', help='Output file (default: input + .compressed)')
    compress_parser.add_argument('--level', type=int, default=6, choices=range(1, 10),
                               help='Compression level 1-9 (default: 6, higher = better compression)')
    compress_parser.add_argument('--stats', action='store_true',
                               help='Show compression statistics')
    
    # Decompress command
    decompress_parser = subparsers.add_parser('decompress', help='Decompress a compressed JSON file')
    decompress_parser.add_argument('input', help='Compressed file to decompress')
    decompress_parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    decompress_parser.add_argument('--pretty', action='store_true',
                                 help='Pretty print the output JSON')
    
    # Batch compress command
    batch_parser = subparsers.add_parser('batch', help='Compress multiple JSON files')
    batch_parser.add_argument('pattern', help='File pattern (e.g., "*.json", "data/*.json")')
    batch_parser.add_argument('--level', type=int, default=6, choices=range(1, 10),
                            help='Compression level 1-9 (default: 6)')
    batch_parser.add_argument('--suffix', default='.compressed',
                            help='Suffix for compressed files (default: .compressed)')
    batch_parser.add_argument('--stats', action='store_true',
                            help='Show compression statistics')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Check if a file is valid compressed JSON')
    validate_parser.add_argument('input', help='File to validate')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'compress':
            output_file = compress_file(
                args.input, 
                args.output, 
                args.level, 
                args.stats
            )
            if not args.stats:
                print(f"Compressed: {args.input} → {output_file}")
        
        elif args.command == 'decompress':
            output_file = decompress_file(args.input, args.output, args.pretty)
            if output_file:
                print(f"Decompressed: {args.input} → {output_file}")
        
        elif args.command == 'batch':
            batch_compress(args.pattern, args.level, args.suffix, args.stats)
        
        elif args.command == 'validate':
            try:
                with open(args.input, 'r') as f:
                    compressed_data = f.read().strip()
                decompress_json(compressed_data)
                print(f"✓ Valid compressed JSON: {args.input}")
            except Exception as e:
                print(f"✗ Invalid compressed JSON: {args.input} ({e})")
                sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()