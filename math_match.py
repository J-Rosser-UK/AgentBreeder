from sympy import Eq, sympify, simplify
from sympy.parsing.latex import parse_latex


class LatexMatcher:
    @staticmethod
    def parse_single_expr(s: str):
        """
        Parse a single LaTeX or plain mathematical expression into a simplified SymPy object.

        Args:
            s (str): The expression to parse.

        Returns:
            SymPy expression.
        """
        if not s:
            return None  # Handle empty strings gracefully

        try:
            expr = parse_latex(s)
        except Exception:
            try:
                expr = sympify(s)
            except Exception as e:
                raise ValueError(f"Failed to parse expression '{s}': {e}")

        return simplify(expr)

    @staticmethod
    def strip_dollar(s: str) -> str:
        """
        Strip surrounding dollar signs from a LaTeX expression if present.

        Args:
            s (str): The expression string.

        Returns:
            str: The expression without surrounding dollar signs.
        """
        s = s.strip()
        if s.startswith("$") and s.endswith("$"):
            return s[1:-1]
        return s

    @staticmethod
    def parse_expression(s: str):
        """
        Parse an expression which can be a single expression or a comma-separated list.

        Args:
            s (str): The expression string.

        Returns:
            SymPy expression or tuple of SymPy expressions.
        """
        s = s.strip()
        if not s:
            return tuple()  # Return an empty tuple for empty strings

        if "," in s:
            # Split by commas and parse each element
            elements = [LatexMatcher.parse_single_expr(e.strip()) for e in s.split(",")]
            return tuple(e for e in elements if e is not None)
        else:
            return LatexMatcher.parse_single_expr(s)

    @staticmethod
    def compare_expression_lists(list1: tuple, list2: tuple) -> bool:
        """
        Compare two lists of SymPy expressions for equivalence, ignoring order.

        Args:
            list1 (tuple): First list of SymPy expressions.
            list2 (tuple): Second list of SymPy expressions.

        Returns:
            bool: True if lists are equivalent as multisets, False otherwise.
        """
        # Create a mutable copy of list2's expressions to track matches
        unmatched = list(list2)

        for expr1 in list1:
            match_found = False
            for idx, expr2 in enumerate(unmatched):
                if expr1.equals(expr2):
                    # Match found; remove from unmatched and break
                    del unmatched[idx]
                    match_found = True
                    break
            if not match_found:
                # No matching expression found for expr1
                return False

        # All expressions in list1 have been matched; ensure list2 has no extra elements
        return len(unmatched) == 0

    @staticmethod
    def match_latex(expr1: str, expr2: str) -> bool:
        """
        Compare two LaTeX expressions or lists for mathematical equivalence.

        Args:
            expr1 (str): First LaTeX expression or list.
            expr2 (str): Second LaTeX expression or list.

        Returns:
            bool: True if the expressions or lists are equivalent, False otherwise.
        """
        try:
            # Strip dollar signs if present
            expr1 = LatexMatcher.strip_dollar(expr1)
            expr2 = LatexMatcher.strip_dollar(expr2)

            # Parse both expressions
            sympy_expr1 = LatexMatcher.parse_expression(expr1)
            sympy_expr2 = LatexMatcher.parse_expression(expr2)

            # Compare the two parsed expressions
            if isinstance(sympy_expr1, tuple) and isinstance(sympy_expr2, tuple):
                if len(sympy_expr1) != len(sympy_expr2):
                    return False
                return LatexMatcher.compare_expression_lists(sympy_expr1, sympy_expr2)
            elif isinstance(sympy_expr1, tuple) or isinstance(sympy_expr2, tuple):
                # One is a list and the other is a single expression
                return False
            else:
                # Both are single expressions
                return sympy_expr1.equals(sympy_expr2)

        except Exception as e:
            raise ValueError(f"Error parsing LaTeX expressions: {e}")


if __name__ == "__main__":
    # Define test cases as tuples of (expr1, expr2)
    test_cases = [
        (r"\frac{1}{2} + \frac{1}{3}", r"1"),  # Test Case 1: Not Equivalent
        (r"$\frac{1}{2} + \frac{1}{2}$", r"1"),  # Test Case 2: Equivalent
        ("2, 3, 4", "2, 5"),  # Test Case 3: Not Equivalent
        ("2, 3, 4", "3, 2, 4"),  # Test Case 4: Equivalent (Permutation)
        (r"\sin^2{x} + \cos^2{x}", "1"),  # Test Case 5: Equivalent
        (r"$x^2 - y^2$", r"(x - y)(x + y)"),  # Test Case 6: Equivalent
        ("a, b, c", "a, b, c"),  # Test Case 7: Equivalent
        ("a, b, c", "a, b"),  # Test Case 8: Not Equivalent
        (r"\sqrt{4}", "2"),  # Test Case 9: Equivalent
        (r"$\sqrt{4}$", "2"),  # Test Case 10: Equivalent
        ("2, 2, 3", "2, 3, 2"),  # Test Case 11: Equivalent with duplicates
        ("2, 2, 3", "2, 3, 3"),  # Test Case 12: Not Equivalent with duplicates
        (r"e^{i\pi} + 1", "0"),  # Test Case 13: Equivalent (Euler's identity)
        ("x + y, y + x", "2*x + 2*y"),  # Test Case 14: Not Equivalent
        ("x + y, y + x", "x + y, x + y"),  # Test Case 15: Equivalent with duplicates
        ("", ""),  # Test Case 16: Both empty (Equivalent)
        ("", "0"),  # Test Case 17: One empty, one not (Not Equivalent)
    ]

    matcher = LatexMatcher()

    for idx, (expr1, expr2) in enumerate(test_cases, start=1):
        try:
            is_equal = matcher.match_latex(expr1, expr2)
            result = "Equivalent" if is_equal else "Not Equivalent"
        except ValueError as ve:
            result = f"Error: {ve}"
        print(f"Test Case {idx}: {result}", f"([{expr1}], [{expr2}])")
