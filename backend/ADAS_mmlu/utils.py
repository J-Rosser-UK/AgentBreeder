import random
import string
from collections import namedtuple
import ast
import numpy as np
import logging

Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])


def bootstrap_confidence_interval(data, num_bootstrap_samples=100000, confidence_level=0.95):
    """
    Calculate the bootstrap confidence interval for the mean of 1D accuracy data.
    Also returns the median of the bootstrap means.
    
    Args:
    - data (list or array of float): 1D list or array of data points.
    - num_bootstrap_samples (int): Number of bootstrap samples.
    - confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).
    
    Returns:
    - str: Formatted string with 95% confidence interval and median as percentages with one decimal place.
    """
    # Convert data to a numpy array for easier manipulation
    data = np.array(data)

    # List to store the means of bootstrap samples
    bootstrap_means = []

    # Generate bootstrap samples and compute the mean for each sample
    for _ in range(num_bootstrap_samples):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Compute the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # Convert bootstrap_means to a numpy array for percentile calculation
    bootstrap_means = np.array(bootstrap_means)

    # Compute the lower and upper percentiles for the confidence interval
    lower_percentile = (1.0 - confidence_level) / 2.0
    upper_percentile = 1.0 - lower_percentile
    ci_lower = np.percentile(bootstrap_means, lower_percentile * 100)
    ci_upper = np.percentile(bootstrap_means, upper_percentile * 100)

    # Compute the median of the bootstrap means
    median = np.median(bootstrap_means)

    # Convert to percentages and format to one decimal place
    ci_lower_percent = ci_lower * 100
    ci_upper_percent = ci_upper * 100
    median_percent = median * 100

    # Return the formatted string with confidence interval and median
    confidence_interval_string = f"95% Bootstrap Confidence Interval: ({ci_lower_percent:.1f}%, {ci_upper_percent:.1f}%), Median: {median_percent:.1f}%"
    return confidence_interval_string, ci_lower, ci_upper, median


def extract_class_code(file_path:str, class_name:str) -> str:
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
                end_line = max([n.lineno for n in ast.walk(node) if hasattr(n, "lineno")])
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
                start_line = min((decorator.lineno for decorator in decorators), default=node.lineno) - 1
                end_line = max([n.lineno for n in ast.walk(node) if hasattr(n, "lineno")])
                
                # Return the source code for the function with decorators
                return "\n".join(file_content.splitlines()[start_line:end_line])
        
        return None  # Function not found
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None