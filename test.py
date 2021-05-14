import numpy as np

if __name__ == "__main__":
    a = [[1], [2], [3], [4]]
    b = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    a = np.array(a)
    b = np.array(b)
    print(np.sum(a*b, axis=0))