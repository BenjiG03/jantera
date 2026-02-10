
import os
import shutil

REPLACEMENTS = {
    "jantera": "canterax",
    "Jantera": "Canterax",
    "JANTERA": "CANTERAX"
}

ROOT_DIR = "."
EXCLUDES = {".git", ".venv", "__pycache__", "jantera.egg-info", "canterax.egg-info", "dist", "build"}

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping binary file: {filepath}")
        return
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    new_content = content
    for old, new in REPLACEMENTS.items():
        new_content = new_content.replace(old, new)
    
    if new_content != content:
        print(f"Updating content: {filepath}")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
        except Exception as e:
            print(f"Error writing {filepath}: {e}")

def rename_item(path):
    dirname, basename = os.path.split(path)
    new_basename = basename
    for old, new in REPLACEMENTS.items():
        if old in new_basename:
            new_basename = new_basename.replace(old, new)
    
    if new_basename != basename:
        new_path = os.path.join(dirname, new_basename)
        print(f"Renaming: {path} -> {new_path}")
        try:
            shutil.move(path, new_path)
        except Exception as e:
            print(f"Error renaming {path}: {e}")
        return new_path
    return path

def process_directory(directory):
    # Pass 1: Replace content in files
    for root, dirs, files in os.walk(directory, topdown=True):
        # Filter directories in place to skip recursion
        dirs[:] = [d for d in dirs if d not in EXCLUDES]
        
        for file in files:
            filepath = os.path.join(root, file)
            # Skip this script itself
            if "rename_project_v2.py" in filepath: continue
            replace_in_file(filepath)

    # Pass 2: Rename files (bottom-up)
    for root, dirs, files in os.walk(directory, topdown=False):
        if any(ex in root.split(os.sep) for ex in EXCLUDES): continue
        for file in files:
            filepath = os.path.join(root, file)
            rename_item(filepath)

    # Pass 3: Rename directories (bottom-up)
    for root, dirs, files in os.walk(directory, topdown=False):
        if any(ex in root.split(os.sep) for ex in EXCLUDES): continue
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            rename_item(dir_path)

if __name__ == "__main__":
    process_directory(ROOT_DIR)
