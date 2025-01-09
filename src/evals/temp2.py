def is_subsequence(sub, main):
    """
    Check if 'sub' is a subsequence of 'main'.
    """
    it = iter(main)
    return all(elem in it for elem in sub)


def longest_consecutive_subsequence(sub, main):
    """
    Find the longest consecutive subsequence in 'sub' that is a subsequence of 'main'.

    Parameters:
    - sub: List of elements representing the subsequence to check.
    - main: List of elements representing the main sequence.

    Returns:
    - The longest consecutive subsequence from 'sub' that is a subsequence of 'main'.
      Returns an empty list if no such subsequence exists.
    """
    max_len = 0
    longest_sub = []

    # Iterate over all possible start indices in 'sub'
    for start in range(len(sub)):
        # Iterate over possible end indices, starting from the end of 'sub' down to 'start'
        for end in range(len(sub), start, -1):
            # Current candidate subsequence
            candidate = sub[start:end]

            # If the candidate length is less than or equal to current max, no need to check further
            if len(candidate) <= max_len:
                break  # Optimization: no longer subsequences possible from this start

            # Check if the candidate is a subsequence of 'main'
            if is_subsequence(candidate, main):
                # Update max_len and longest_sub if a longer subsequence is found
                if len(candidate) > max_len:
                    max_len = len(candidate)
                    longest_sub = candidate
                break  # No need to check shorter subsequences from this start

    return longest_sub


# Example Usage:
answer_trace = ["A", "B", "C", "D", "E", "F", "G"]
target_trace = ["X", "A", "Y", "B", "C", "Z", "D", "E", "F", "W", "G"]

longest_subseq = longest_consecutive_subsequence(answer_trace, target_trace)
print("Longest Consecutive Subsequence:", longest_subseq)
