import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

if __name__ == "__main__":
    f = open('iris.csv')
    D = np.array([[], [], [], []])
    L = np.array([])
    label = -1
    for line in f:
        s = np.array([line.split(',')[0], line.split(',')[1], line.split(',')[2], line.split(',')[3]], dtype=float)

        if line.split(',')[4].rstrip() == 'Iris-setosa':
            label = 0
        elif line.split(',')[4].rstrip() == 'Iris-versicolor':
            label = 1
        elif line.split(',')[4].rstrip() == 'Iris-virginica':
            label = 2
        col = np.shape(D)[1]
        D = np.insert(D, col, s, axis=1)
        L = np.insert(L, col, label)

    mu = D.mean(1)
    mu = mu.reshape((mu.size, 1))
    DC = D - mu

    C = (DC @ DC.T) / np.shape(DC)[1]

    # eig for non symmetric matrix, does not sort eigenvalues/vectors
    # eigh for symmetric matrix, sorts ascending eigenvalues/vectors
    s, U = np.linalg.eigh(C)
    # print(np.shape(U))
    P = U[:, ::-1][:, 0:2]

    y = P.T @ DC

    x = np.zeros((2, 150))

    plt.scatter(y[0, L == 0], y[1, L == 0])
    plt.scatter(y[0, L == 1], y[1, L == 1])
    plt.scatter(y[0, L == 2], y[1, L == 2])

    Sw = np.zeros((D.shape[0], D.shape[0]))  # (4, 4)
    Sb = np.zeros((D.shape[0], D.shape[0]))  # (4, 4)

    for i in range(3):
        mu2 = D[:, L == i].mean(1)
        mu2 = mu2.reshape((mu2.size, 1))
        DC = D[:, L == i] - mu2
        Sw += (DC @ DC.T)
        Sb += (mu2 - mu) @ (mu2 - mu).T * np.shape(D[:, L == i])[1]
    Sw = Sw / np.shape(D)[1]
    Sb = Sb / np.shape(D)[1]

    s, U = scipy.linalg.eigh(Sb, Sw)

    W = U[:, ::-1][:, 0:2]

    # Notice that the columns of W are not necessarily orthogonal.
    # If we want, we can find a basis U for the
    # subspace spanned by W using the singular value decomposition of W
    # UW, _, _ = np.linalg.svd(W)
    # U = UW[:, 0:2]

    y = W.T @ D
    plt.figure()

    plt.scatter(y[0, L == 0], y[1, L == 0])
    plt.scatter(y[0, L == 1], y[1, L == 1])
    plt.scatter(y[0, L == 2], y[1, L == 2])

    plt.show()
