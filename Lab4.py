import numpy as np
import matplotlib.pyplot as plt


def GAU_pdf(x, mu, var):
    y=np.exp(-(x-mu)**2/2/var)/np.sqrt(2*np.pi*var)
    return y

def GAU_logpdf(x, mu, var):
    y=-0.5*np.log(2*np.pi)-0.5*np.log(var)-((x-mu)**2)/(2*var)
    return y

def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    N = x.shape[1]
    y = np.zeros((N, ))
    k = - 0.5 * M * np.log(2*np.pi) - 0.5 * np.linalg.slogdet(C)[1]
    for i in range(N):
        m = x[:, i].reshape((M, 1)) - mu
        y[i] = k - 0.5 * m.T @ np.linalg.inv(C) @ m
    return y

def logpdf_GAU_ND2(x, mu, C):
    M = x.shape[0]
    N = x.shape[1]
    y = np.zeros((N, N))
    k = - 0.5 * M * np.log(2*np.pi) - 0.5 * np.linalg.slogdet(C)[1]
    m = x - mu
    y = k - 0.5 * m.T @ np.linalg.inv(C) @ m
    return np.diag(y)

XGAU = np.load('XGau.npy')

ll_samples = GAU_pdf(XGAU, 1.0, 2.0) #mu e var a caso
likelihood = ll_samples.prod() #prodotto tra tutti i valori del vettore per trovare la likelihood
#viene sempre 0, i valori sono troppo piccoli moltiplicati tra loro

#usiamo il log per non ottenere 0 e poter calcolare la loglikelihood
lllog_samples = GAU_logpdf(XGAU, 1.0, 2.0) #valori a caso
loglikelihood = np.sum(lllog_samples) #somma di tutti i valori del vettore per trovare likelihood del log, loglikelihood
print(loglikelihood)

#trovo media e varianza e ricalcolo(dimostrato che la gaussiano che meglio approssima è quella
#che ha mu = media e var = varianza). Ottenere questi valori è come calcolare ogni mu e var e cercare il max
mu = XGAU.mean()
var = XGAU.var()


ll = GAU_logpdf(XGAU, mu, var) #funzione di gauss log applicata al vettore
maxll = np.sum(ll) #somma dei valori per vedere la likelihood max
#Non serve sommarli dato che sappiamo già che è la max, è solo una dimostrazione ulteriore
print(maxll)

#scoperti mu e var abbiamo la gaussiana ma
#essendo il log della likelihood mettiamo tutto su un esponenziale
plt.figure()
plt.hist(XGAU, bins=50, density=True)
XPlot = np.linspace(-8, 12, 1000)
plt.plot(XPlot, np.exp(GAU_logpdf(XPlot, mu, var)))#usiamo linspace per disegnare la gaussiana
plt.show()

XND = np.load('Solutions/XND.npy')
mu = np.load('Solutions/muND.npy')
C = np.load('Solutions/CND.npy')
pdfSol = np.load('Solutions/llND.npy')
pdfGau = logpdf_GAU_ND2(XND, mu, C)
print(np.abs(pdfSol - pdfGau).mean())
