import os
import shutil

def create_dataset_structure():
    # Base directory structure
    structure = {
        'dataset': {
            'train': {
                'silent': {'GRID': {}},
                'speaking': {'GRID': {}}
            },
            'val': {
                'silent': {'GRID': {}},
                'speaking': {'GRID': {}}
            },
            'test': {
                'silent': {'GRID': {}},
                'speaking': {'GRID': {}}
            },
            'models': {},
            'tensorboard': {}
        }
    }

    # Create the directory structure
    for parent_dir, children in structure.items():
        create_nested_dirs('', parent_dir, children)

    print("Dataset directory structure created!")
    print("\nStructure created:")
    print("dataset/")
    print("├── train/")
    print("│   ├── silent/")
    print("│   │   └── GRID/")
    print("│   └── speaking/")
    print("│       └── GRID/")
    print("├── val/")
    print("│   ├── silent/")
    print("│   │   └── GRID/")
    print("│   └── speaking/")
    print("│       └── GRID/")
    print("├── test/")
    print("│   ├── silent/")
    print("│   │   └── GRID/")
    print("│   └── speaking/")
    print("│       └── GRID/")
    print("├── models/")
    print("└── tensorboard/")

def create_nested_dirs(parent, dir_name, children):
    """Recursively create nested directory structure"""
    current_path = os.path.join(parent, dir_name)
    if not os.path.exists(current_path):
        os.makedirs(current_path)
    
    if isinstance(children, dict):
        for child_name, child_children in children.items():
            create_nested_dirs(current_path, child_name, child_children)

if __name__ == "__main__":
    # Ask if user wants to delete existing dataset directory
    if os.path.exists('dataset'):
        response = input("dataset/ directory already exists. Delete and recreate? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree('dataset')
            create_dataset_structure()
        else:
            print("Aborted. Please backup or delete the existing dataset/ directory manually.")
    else:
        create_dataset_structure()
