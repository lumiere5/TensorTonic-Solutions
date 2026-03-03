import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    r, c = A.shape
    res = np.zeros((c, r), dtype="float")

    for row in range(0, r):
        for col in range(0, c):
            res[col][row] = A[row][col]

    return res
