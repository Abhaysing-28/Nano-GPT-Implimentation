"""
Batch process multiple data files
"""
import os
import glob
from prepare_universal import prepare_dataset

def batch_prepare(input_pattern, output_base_dir, file_type='auto', text_column=None, text_field=None):
    """
    Process multiple files matching a pattern
    
    Example:
        batch_prepare('data/raw/*.txt', 'data/processed')
        batch_prepare('data/raw/*.csv', 'data/processed', text_column='review')
    """
    files = glob.glob(input_pattern)
    
    print(f"Found {len(files)} files matching pattern: {input_pattern}")
    
    for file_path in files:
        # Create output directory based on filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(output_base_dir, base_name)
        
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print(f"Output to: {output_dir}")
        print(f"{'='*60}")
        
        try:
            prepare_dataset(
                input_path=file_path,
                output_dir=output_dir,
                file_type=file_type,
                text_column=text_column,
                text_field=text_field
            )
        except Exception as e:
            print(f"ERROR processing {file_path}: {e}")
            continue
    
    print(f"\n✓ Batch processing complete! Processed {len(files)} files.")

if __name__ == '__main__':
    # Example usage
    batch_prepare('data/raw/*.txt', 'data/processed')