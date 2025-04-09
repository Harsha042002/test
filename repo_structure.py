import os
import sys

def print_repo_structure(start_path='.', indent='', exclude_dirs=None, exclude_patterns=None):
    """
    Print the structure of the repository, excluding specified directories and patterns.
    """
    if exclude_dirs is None:
        exclude_dirs = ['.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env']
    
    if exclude_patterns is None:
        exclude_patterns = ['.pyc', '.pyo', '.pyd', '.so', '.dll', '.class']
    
    # Get all items in the current directory
    try:
        items = sorted(os.listdir(start_path))
    except PermissionError:
        print(f"{indent}├── [Permission denied]")
        return
    
    # Process each item
    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        
        # Skip excluded directories
        if os.path.isdir(path) and item in exclude_dirs:
            continue
            
        # Skip files matching excluded patterns
        if any(item.endswith(pattern) for pattern in exclude_patterns):
            continue
        
        # Determine if this is the last item
        is_last = i == len(items) - 1
        
        # Print the current item
        if is_last:
            print(f"{indent}└── {item}")
            next_indent = indent + "    "
        else:
            print(f"{indent}├── {item}")
            next_indent = indent + "│   "
        
        # Recursively process directories
        if os.path.isdir(path):
            print_repo_structure(path, next_indent, exclude_dirs, exclude_patterns)

if __name__ == "__main__":
    start_path = '.'
    if len(sys.argv) > 1:
        start_path = sys.argv[1]
    
    print(f"Repository structure for: {os.path.abspath(start_path)}")
    print_repo_structure(start_path)
