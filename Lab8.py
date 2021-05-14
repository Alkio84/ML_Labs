import numpy as np
import sklearn
import Lab5
from scipy import optimize

def f(x): # x = numpy attay of shape (2, )
    y = x[0]
    z = x[1]
    return (y + 3)**2 + np.sin(y) + (z + 1)**2

def fgrad(x): # x = numpy attay of shape (2, )
    y = x[0]
    z = x[1]
    gradx = 2 * (y + 3) + np.cos(y)
    grady = 2 * (z + 1)
    grad = np.array([gradx, grady])
    return ((y + 3)**2 + np.sin(y) + (z + 1)**2, grad)

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L != 0] # We remove setosa from L
    L[L == 2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def logreg_obj_wrap(DTR, LTR, l):
    def logreg_obj(v):
        w = v[0:-1]
        b = v[-1]
        n = DTR.shape[1]
        z = np.where(LTR == 1, 1, -1)
        # formula 2
        # J1 = l / 2 * np.power(np.linalg.norm(w.T, 2), 2)
        # s = w.T @ DTR + b
        # J2 = LTR.T * np.log1p(np.exp(-s)) + (1 - LTR.T) * np.log1p(np.exp(s))
        # formula 3
        J1 = l * (np.linalg.norm(w, 2) ** 2) / 2
        J2 = np.log1p(np.exp(-z * (w.T @ DTR + b)))
        J = J1 + np.sum(J2) / n

        return J
    return logreg_obj

def logreg_obj_wrap_mc(DTR, LTR, l):
    def logreg_obj_mc(v):
        w = v[0:-1]
        b = v[-1]
        n = DTR.shape[1]
        z = np.where(LTR == 1, 1, -1)
        # formula 2
        # J1 = l / 2 * np.power(np.linalg.norm(w.T, 2), 2)
        # s = w.T @ DTR + b
        # J2 = LTR.T * np.log1p(np.exp(-s)) + (1 - LTR.T) * np.log1p(np.exp(s))
        # formula 3
        J1 = l * (np.linalg.norm(w, 2) ** 2) / 2
        J2 = np.log1p(np.exp(-z * (w.T @ DTR + b)))
        J = J1 + np.sum(J2) / n

        return J
    return logreg_obj

if __name__ == "__main__":
    # (x, f, d) = optimize.fmin_l_bfgs_b(f, [0, 0], approx_grad=True)
    # print(x, f, d)
    # (x, f, d) = optimize.fmin_l_bfgs_b(fgrad, [0, 0])
    # print(x, f, d)

    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = Lab5.split_db_2to1(D, L)

    l = 10e-6
    x0 = np.zeros(DTR.shape[0] + 1)
    logreg_obj = logreg_obj_wrap(DTR, LTR, l)
    logreg_obj_mc = logreg_obj_wrap_mc(DTR, LTR, l)

    (x, f, d) = optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)
    w, b = x[0:-1], x[-1]

    S = w.T @ DTE + b
    LP = np.where(S > 0, 1, 0)

    acc = sum(np.where(LP == LTE, 1, 0)) / LTE.shape[0]
    print(1-acc)


