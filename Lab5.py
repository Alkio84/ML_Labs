import scipy
import sklearn.datasets
import numpy as np
import Lab4

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def media(D):
    mu = D.mean(1).reshape((D.shape[0], 1))
    return mu

def cov(D, mu): #dati e mu già relativi alla classe
    y = (D - mu) @ (D - mu).T
    y = y / D.shape[1]
    return y


# 0 = Multivariate Gaussian Classifier   1 = Naive Bayes   2 = Tied Covariance Gaussian Classifier
def Lab5class(var, DTR, LTR, DTE, LTE, nclass):

    S = np.zeros((nclass, DTE.shape[1]))
    SJoint = np.zeros((1, DTE.shape[1]))
    sigma = np.zeros((DTR.shape[0], DTR.shape[0]))
    if(var == 0): #Multivariate Gaussian Classifier
        for i in range(nclass):
            mu = media(DTR[:, LTR == i]) # media per classe
            sigma = cov(DTR[:, LTR == i], mu) # matrice di covarianza per classe
            S[i] = Lab4.logpdf_GAU_ND2(DTE, mu, sigma)
    elif(var == 1): #Naive Bayes -> covariance(sigma) is only the diag, we assume the statistic independence
        for i in range(nclass):
            mu = media(DTR[:, LTR == i])  # media per classe
            sigma = np.diag(np.diag(cov(DTR[:, LTR == i], mu)))  # diag della matrice di covarianza per classe
            S[i] = Lab4.logpdf_GAU_ND2(DTE, mu, sigma)
    elif(var == 2):  # Tied Covariance Gaussian Classifier
        for i in range(nclass):
            mu = media(DTR[:, LTR == i])  # media per classe
            sigmatmp = cov(DTR[:, LTR == i], mu)  # matrice di covarianza per classe
            # matrice delle covarianze unica, ottenuta come somma PESATA delle singole matrici delle covarianze
            sigma += sigmatmp * DTR[:, LTR == i].shape[1]
        sigma = sigma / DTR.shape[1]
        # Dopo aver ottenuto la matrice delle covarianze tra classi si può costruire S, usando per ogni classe la propria media
        for i in range(nclass):
            mu = media(DTR[:, LTR == i])
            S[i] = Lab4.logpdf_GAU_ND2(DTE, mu, sigma)
    # usando i dati in log ci sono problemi numerici di segno (alcune percentuali sono negative)
    # usando un esponenziale normale ci sono problemi nel calcolarlo
    #si può usare un trick "log-sum-exp"
    # moltiplicazione x 1/3 in log diventa somma
    SJoint = scipy.special.logsumexp(S, 0)
    SPost = np.zeros((nclass, DTE.shape[1]))

    for i in range(S.shape[1]):
        SPost[0, i] = S[0, i] - SJoint[i]
        SPost[1, i] = S[1, i] - SJoint[i]
        SPost[2, i] = S[2, i] - SJoint[i]
        # print("{:.3f}".format(np.exp(SPost[0, i])), "{:.3f}".format(np.exp(SPost[1, i])), "{:.3f}".format(np.exp(SPost[2, i])))

    #assegnazione etichetta(quella con valore più alto) e calcolo accuracy
    FinalL = S.argmax(0)
    acc=0
    for i in range(SPost.shape[1]):
        if(FinalL[i] == LTE[i]):
            acc += 1
    acc = acc/LTE.size
    # print('Accuracy: ', acc*100, '%')
    return acc

if __name__ == "__main__":
    D, L = load_iris()
    print('Fixed Split (2/3):')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    acc = Lab5class(2, DTR, LTR, DTE, LTE, 3)
    print("{:.3f}".format(acc))
    print('Cross Validation: ')
    acc = 0
    for i in range(D.shape[1]):
        acc += Lab5class(2, D[:, np.arange(D.shape[1]) != i], L[np.arange(D.shape[1]) != i], D[:, i].reshape((D.shape[0], 1)), L[i].reshape((1, 1)), 3)
    acc = acc / D.shape[1]
    print("{:.3f}".format(acc))









