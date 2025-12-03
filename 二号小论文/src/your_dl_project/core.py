# src/filereader_tool/core.py

def read_file_content(filepath):
    """
    Reads the content of a specified file and returns it.
    If the file is not found or another error occurs, it returns an error message.

    :param filepath: The path to the file to be read.
    :return: The file's content as a string, or an error message string.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return f"Error: The file '{filepath}' was not found."
    except Exception as e:
        return f"An error occurred while reading the file: {e}"
