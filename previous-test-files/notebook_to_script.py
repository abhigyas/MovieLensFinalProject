import json
import sys
from pathlib import Path

def notebook_to_script(notebook_path, output_path=None):
    # Read notebook file
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Initialize output content
    script_content = []
    
    # Process each cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Add code cells directly
            script_content.extend(cell['source'])
            script_content.append('\n')
        elif cell['cell_type'] == 'markdown':
            # Convert markdown to Python comments
            markdown = ''.join(cell['source']).split('\n')
            script_content.extend([f'# {line}\n' if line else '#\n' for line in markdown])
            script_content.append('\n')
    
    # Set output path
    if output_path is None:
        output_path = Path(notebook_path).with_suffix('.py')
    
    # Write combined script
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(script_content)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        notebook_to_script(notebook_path, output_path)
    else:
        print("Usage: python notebook_to_script.py <notebook_path> [output_path]")