"""
Universal data preparation script for multiple formats (TXT, CSV, JSON)
Supports different text extraction patterns for various use cases
"""

import os
import sys
import pickle
import json
import csv
import argparse
import numpy as np

def load_txt(file_path):
    """Load plain text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_csv(file_path, text_column=None):
    """
    Load CSV file and extract text from specified column
    If text_column is None, concatenate all columns
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if text_column:
                if text_column in row:
                    texts.append(row[text_column])
                else:
                    raise ValueError(f"Column '{text_column}' not found in CSV")
            else:
                # Concatenate all columns
                texts.append(' '.join(row.values()))
    
    return '\n'.join(texts)

def load_json(file_path, text_field=None):
    """
    Load JSON file and extract text from specified field
    Supports both:
    - List of objects: [{"text": "..."}, {"text": "..."}]
    - Single object: {"text": "..."}
    - Nested structures
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    
    # Handle list of objects
    if isinstance(data, list):
        for item in data:
            if text_field:
                # Extract specific field
                if text_field in item:
                    texts.append(str(item[text_field]))
                else:
                    raise ValueError(f"Field '{text_field}' not found in JSON object")
            else:
                # Concatenate all string values
                texts.append(extract_text_from_dict(item))
    
    # Handle single object
    elif isinstance(data, dict):
        if text_field:
            texts.append(str(data[text_field]))
        else:
            texts.append(extract_text_from_dict(data))
    
    return '\n'.join(texts)

def extract_text_from_dict(obj):
    """Recursively extract all text from a dictionary"""
    texts = []
    for key, value in obj.items():
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            texts.append(extract_text_from_dict(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict):
                    texts.append(extract_text_from_dict(item))
    return ' '.join(texts)

def prepare_dataset(input_path, output_dir, file_type='auto', text_column=None, text_field=None):
    """
    Main function to prepare dataset from various formats
    
    Args:
        input_path: Path to input file or directory
        output_dir: Directory to save train.bin, val.bin, and meta.pkl
        file_type: 'txt', 'csv', 'json', or 'auto' (detect from extension)
        text_column: For CSV - which column contains text
        text_field: For JSON - which field contains text
    """
    
    # Auto-detect file type
    if file_type == 'auto':
        _, ext = os.path.splitext(input_path)
        file_type = ext[1:].lower()  # Remove the dot
    
    print(f"Processing {file_type.upper()} file: {input_path}")
    
    # Load data based on format
    if file_type == 'txt':
        data = load_txt(input_path)
    elif file_type == 'csv':
        data = load_csv(input_path, text_column)
    elif file_type == 'json':
        data = load_json(input_path, text_field)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    print(f"Length of dataset in characters: {len(data):,}")
    
    # Get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("All unique characters:", ''.join(chars))
    print(f"Vocab size: {vocab_size:,}")
    
    # Create character-to-index and index-to-character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    # Create train and validation splits (90% train, 10% validation)
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]
    
    # Encode to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")
    
    # Export to binary files
    os.makedirs(output_dir, exist_ok=True)
    
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    
    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"\n✓ Dataset prepared successfully!")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Files created: train.bin, val.bin, meta.pkl")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare dataset from multiple formats')
    parser.add_argument('input', help='Path to input file (TXT, CSV, or JSON)')
    parser.add_argument('--output', default=None, help='Output directory (default: same as input file directory)')
    parser.add_argument('--type', default='auto', choices=['auto', 'txt', 'csv', 'json'], 
                        help='File type (auto-detect by default)')
    parser.add_argument('--text-column', default=None, 
                        help='For CSV: column name containing text')
    parser.add_argument('--text-field', default=None, 
                        help='For JSON: field name containing text')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output is None:
        args.output = os.path.dirname(os.path.abspath(args.input))
    
    prepare_dataset(
        input_path=args.input,
        output_dir=args.output,
        file_type=args.type,
        text_column=args.text_column,
        text_field=args.text_field
    )