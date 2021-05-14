import numpy as np
import sklearn
import Lab5
from scipy import optimize

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L != 0] # We remove setosa from L
    L[L == 2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def SVM_dual_obj_wrap(DTR, LTR, K):
    DTRc = np.row_stack((DTR, np.ones(DTR.shape[1])))
    DTRc[DTR.shape[0]] = DTRc[DTR.shape[0]] * K
    x = DTRc
    z = np.where(LTR == 1, 1, -1)
    def SVM_dual_obj(v):
        a = v
        # for i in range(x.shape[1]):
        #     for j in range(x.shape[1]):
        #         H[i, j] = z[i] * z[j] * (x[:, i].T @ x[:, j] + K)
        G = np.dot(x.T, x) + 1
        H = G * z.reshape(z.shape[0], 1)
        H = H * z
        J1 = a.T @ H @ a / 2
        J2 = - a.T @ np.ones(a.shape)
        grad = H @ a - np.ones(a.shape)
        return (J1 + J2, grad.reshape(x.shape[1]))
    return SVM_dual_obj





if __name__ == "__main__":
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = Lab5.split_db_2to1(D, L)


    C = 10
    x0 = np.ones(DTR.shape[1]) * C
    SVM_dual_obj = SVM_dual_obj_wrap(DTR, LTR, 1)

    l = []
    for i in range(DTR.shape[1]):
        l.append((0, C))
    (x, f, d) = optimize.fmin_l_bfgs_b(SVM_dual_obj, x0, bounds=l, factr=1)
    print("f: ", -f)



