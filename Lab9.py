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

def SVM_dual_obj_wrap(DTR, z, K):
    x = DTR
    def SVM_dual_obj(v):
        a = v
        # H = np.zeros((x.shape[1], x.shape[1]))
        # for i in range(x.shape[1]):
        #    for j in range(x.shape[1]):
        #        H[i, j] = z[i] * z[j] * (x[:, i].T @ x[:, j] + K)
        G = x.T @ x + K
        H = G * z.reshape(z.shape[0], 1)
        H = H * z
        J1 = a.T @ H @ a / 2
        J2 = - a.T @ np.ones(a.shape)
        grad = H @ a - np.ones(a.shape)
        return J1 + J2, grad.reshape(x.shape[1])
    return SVM_dual_obj

def SVM_evaluation(DTE, LTE, wc, K):
    w = wc[0:-1]
    b = wc[-1] * K

    eval_mat = w.T @ DTE + b
    estimated_label = np.where(eval_mat > 0, 1, 0)

    correct = np.where(estimated_label == LTE, 1, 0)
    return sum(correct) / (len(correct))

if __name__ == "__main__":
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = Lab5.split_db_2to1(D, L)
    #definizione costanti C e K
    C = 0.1
    K = 10
    # aggiunta di una riga inizializzata a K ai dati oer creare x cappello
    DTRc = np.row_stack((DTR, np.ones(DTR.shape[1])))
    DTRc[DTR.shape[0]] = DTRc[DTR.shape[0]] * K
    #creazione di z
    z = np.where(LTR == 1, 1, -1)

    x0 = np.zeros(DTR.shape[1])
    SVM_dual_obj = SVM_dual_obj_wrap(DTRc, z, K)

    l = []
    for i in range(DTR.shape[1]):
        l.append((0, C))
    (x, f, d) = optimize.fmin_l_bfgs_b(SVM_dual_obj, x0, bounds=l, factr=1)
    print(-f)
    #calcolo w usato poi per etichettare
    wc = np.sum(z * x * DTRc, axis=1)

    #metodo 1 - moltiplico tutta la matrice dati con riga aggiuntiva K per w e classifico per < o > di 0
    DTEc = np.row_stack((DTE, np.ones(DTE.shape[1])))
    DTEc[DTE.shape[0]] = DTEc[DTR.shape[0]] * K

    eval_mat = wc.T @ DTEc
    label = np.where(eval_mat > 0, 1, 0)

    prec = np.where(label == LTE, 1, 0)
    s = sum(prec) / (len(prec))
    print(s)

    #metodo 2 - trovo w(senza cappello) e b e trovo eval_mat come w*x+b*k, poi classifico per < o > di 0
    #migliore perchè non devi modificare i dati
    print(1 - SVM_evaluation(DTE, LTE, wc, K))




