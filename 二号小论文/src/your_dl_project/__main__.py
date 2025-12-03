# src/filereader_tool/__main__.py

import os
from .core import read_file_content

def main():
    """
    Main entry point for the application script.
    """
    # Construct the path to the file.
    # This assumes the script is run from the project's root directory.
    # A more robust solution might use importlib.resources for packaged data.
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_to_read = os.path.join(project_root, 'data', 'your_file.txt')
    
    # Call the function and get the content
    content = read_file_content(file_to_read)
    
    # Print the returned content
    print(content)

if __name__ == "__main__":
    main()
