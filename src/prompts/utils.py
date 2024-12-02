import ast


def extract_class_code(file_path: str, class_name: str) -> str:
    """
    Extracts the code for a specified class from a Python file.

    Args:
        file_path (str): Path to the Python file.
        class_name (str): Name of the class to extract.

    Returns:
        str: The code of the specified class as a string, or None if not found.
    """
    try:
        with open(file_path, "r") as file:
            file_content = file.read()

        # Parse the file content into an AST
        tree = ast.parse(file_content)

        # Locate the specified class in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Extract the source code for the class
                start_line = node.lineno - 1
                end_line = max(
                    [n.lineno for n in ast.walk(node) if hasattr(n, "lineno")]
                )
                return "\n".join(file_content.splitlines()[start_line:end_line])

        return None  # Class not found

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_function_code(file_path: str, function_name: str) -> str:
    """
    Extracts the code for a specified function from a Python file, including any decorators.

    Args:
        file_path (str): Path to the Python file.
        function_name (str): Name of the function to extract.

    Returns:
        str: The code of the specified function (including decorators) as a string, or None if not found.
    """
    try:
        with open(file_path, "r") as file:
            file_content = file.read()

        # Parse the file content into an AST
        tree = ast.parse(file_content)

        # Locate the specified function in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Determine the start and end lines, including decorators
                decorators = node.decorator_list
                start_line = (
                    min(
                        (decorator.lineno for decorator in decorators),
                        default=node.lineno,
                    )
                    - 1
                )
                end_line = max(
                    [n.lineno for n in ast.walk(node) if hasattr(n, "lineno")]
                )

                # Return the source code for the function with decorators
                return "\n".join(file_content.splitlines()[start_line:end_line])

        return None  # Function not found

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
