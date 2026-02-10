
import os
import shutil

REPLACEMENTS = {
    "canterax": "canterax",
    "Canterax": "Canterax",
    "CANTERAX": "CANTERAX"
}

ROOT_DIR = "."
EXCLUDES = [".git", ".venv", "__pycache__", "canterax.egg-info", "canterax.egg-info"]

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping binary file: {filepath}")
        return

    new_content = content
    for old, new in REPLACEMENTS.items():
        new_content = new_content.replace(old, new)
    
    if new_content != content:
        print(f"Updating content: {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

def rename_file_or_dir(path):
    dirname, basename = os.path.split(path)
    new_basename = basename
    for old, new in REPLACEMENTS.items():
        if old in new_basename:
            new_basename = new_basename.replace(old, new)
    
    if new_basename != basename:
        new_path = os.path.join(dirname, new_basename)
        print(f"Renaming: {path} -> {new_path}")
        shutil.move(path, new_path)
        return new_path
    return path

def process_directory(directory):
    for root, dirs, files in os.walk(directory, topdown=False): # Bottom-up to handle directory renames safely
        dirs[:] = [d for d in dirs if d not in EXCLUDES]
        
        # Process files
        for file in files:
            filepath = os.path.join(root, file)
            replace_in_file(filepath)
            rename_file_or_dir(filepath)
            
        # Process directories (renaming them if needed)
        # Since os.walk topdown=False, we process child dirs first, then rename parent
        # But os.walk yields the original directory name in 'root', renaming it might affect the loop?
        # Actually topdown=False yields children of root before root.
        # But we manipulate 'dirs' in-place only if topdown=True.
        # Let's do a second pass for directory renaming to be safe or handle in loop carefully.
        
    # Second pass for directory renaming (since walk yields current root, renaming it while inside might be tricky)
    # Actually just renaming files first is safer. Let's do files first.

    # Now rename directories bottom-up
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            if dir_name in EXCLUDES: continue
            dir_path = os.path.join(root, dir_name)
            rename_file_or_dir(dir_path)

if __name__ == "__main__":
    process_directory(ROOT_DIR)
