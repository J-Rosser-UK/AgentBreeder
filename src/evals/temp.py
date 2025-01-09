def quicksort_trace(arr):
    """
    Perform quicksort on the input array while tracing the state of the array after each partition.

    Parameters:
    arr (list of float): The list of numerical keys to sort.

    Returns:
    tuple: A tuple containing:
        - trace (list of lists): The states of the array after each partitioning step.
        - pred (list of float): The final sorted array.
    """
    trace = []

    def quicksort(arr, low, high):
        if low < high:
            p = partition(arr, low, high)
            quicksort(arr, low, p - 1)
            quicksort(arr, p + 1, high)

    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        # Place the pivot in the correct position
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        # Append a copy of the current state to trace
        trace.append(arr.copy())
        return i + 1

    # Append the initial state of the array
    trace.append(arr.copy())
    # Perform quicksort
    quicksort(arr, 0, len(arr) - 1)
    # Return the trace and the sorted array
    return (trace, arr)


# Example Usage
if __name__ == "__main__":
    keys = [0.075, 0.82, 0.256, 0.682, 0.304]
    trace, pred = quicksort_trace(keys)

    print(trace, pred)

    # # Formatting the output to match the example
    # formatted_trace = ", ".join([f"[{', '.join(map(str, state))}]" for state in trace])
    # formatted_pred = f"[{', '.join(map(str, pred))}]"

    # print(f"trace | pred:\n{formatted_trace} | {formatted_pred}")
