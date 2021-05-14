import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
def getConfusionMatrix(predictions, labels, nclass):
    conf = np.zeros((nclass, nclass), dtype=int)
    for i in range(predictions.shape[0]):
        conf[predictions[i], labels[i]] += 1
    return conf

def optimalBayesBinary(llrs, Cfn=10, Cfp=1, pi1=0.5):
    return np.where(llrs > -np.log(pi1*Cfn/((1-pi1)*Cfp)), 1, 0)

def bayesRisk(conf, Cfn=10, Cfp=1, pi1=0.5):
    fnr = conf[0, 1] / (conf[0, 1]+conf[1, 1])
    fpr = conf[1, 0] / (conf[1, 0]+conf[0, 0])
    return pi1*Cfn*fnr+(1-pi1)*Cfp*fpr

def normalizedBayesRisk(conf, Cfn=10, Cfp=1, pi1=0.5):
    B = bayesRisk(conf, Cfn, Cfp, pi1)
    Bdummy = min(pi1*Cfn, (1-pi1)*Cfp)
    return B/Bdummy

def minDetectionCost(llrs, lab):
    min_dcf = float('inf')
    for i in np.arange(min(llrs), max(llrs), 0.1):
        pred = np.where(llrs > i, 1, 0)
        conf = getConfusionMatrix(pred, lab, 2)
        r = normalizedBayesRisk(conf)
        if min_dcf > r:
            min_dcf = r
    return min_dcf

def plotROC(llrs, lab): #riceve le loglikelihoods, calcola per ogni threshold e plotta TPR over FPR
    TPR = []
    FPR = []
    index = 0
    for i in np.arange(min(llrs), max(llrs), 0.1):
        pred = np.where(llrs > i, 1, 0)
        conf = getConfusionMatrix(pred, lab, 2)
        TPR.insert(index,conf[1, 1] / (conf[0, 1] + conf[1, 1]))
        FPR.insert(index, conf[1, 0] / (conf[0, 0] + conf[1, 0]))
        index += 1
    plt.grid()
    plt.plot(np.array(FPR), np.array(TPR))
    plt.show()
    return




if __name__ == "__main__":
    commedia_predictions = np.load('Lab6Sol/commedia_ll.npy').argmax(0)
    commedia_labels = np.load('Lab6Sol/commedia_labels.npy')
    commedia_infpar_llr = np.load('Lab6Sol/commedia_llr_infpar.npy')
    commedia_infpar_labels = np.load('Lab6Sol/commedia_labels_infpar.npy')
    print('Confusion matrix frequencist')
    print(getConfusionMatrix(commedia_predictions, commedia_labels, 3))
    bayes_predictions = optimalBayesBinary(commedia_infpar_llr)
    print('Confusion matrix optimal bayes')
    print(getConfusionMatrix(bayes_predictions, commedia_infpar_labels, 2))
    print('Risk: ', bayesRisk(getConfusionMatrix(bayes_predictions, commedia_infpar_labels, 2)))
    print('Risk normalized: ', normalizedBayesRisk(getConfusionMatrix(bayes_predictions, commedia_infpar_labels, 2)))
    print(minDetectionCost(commedia_infpar_llr, commedia_infpar_labels))
    plotROC(commedia_infpar_llr, commedia_infpar_labels)

