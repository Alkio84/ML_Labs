import sys
import numpy as np
from scipy import special


def load_data():
    lInf = []

    if sys.version_info.major == 3:  # Check if Python version is Python 3 or Python 2
        f = open('Divina Commedia/inferno.txt', encoding="ISO-8859-1")
    else:
        f = open('Divina Commedia/inferno.txt')

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    if sys.version_info.major == 3:  # Check if Python version is Python 3 or Python 2
        f = open('Divina Commedia/purgatorio.txt', encoding="ISO-8859-1")
    else:
        f = open('Divina Commedia/purgatorio.txt')

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    if sys.version_info.major == 3:  # Check if Python version is Python 3 or Python 2
        f = open('Divina Commedia/paradiso.txt', encoding="ISO-8859-1")
    else:
        f = open('Divina Commedia/paradiso.txt')
    for line in f:
        lPar.append(line.strip())
    f.close()
    np.array(lInf)
    return np.array(lInf), np.array(lPur), np.array(lPar)


def split_data(l, n):
    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])

    return lTrain, lTest

def textSplitter(text):
    splitted = []
    for row in text:
        l = row.split(" ")
        for i in range(len(l)):
            l[i]=l[i].lower()
        splitted.extend(l)
    return splitted

def dictionaryBuilder(texts): #riceve lista di testi
    dic = {}
    for text in texts:
        for word in text:
                dic[word] = 0
    return dic

def dictionaryFiller(text, dic):
    dic2 = dic.copy()
    for word in text:
        dic2[word] += 1
    return dic2

if __name__ == "__main__":
    #carico i dati
    inf, pur, par = load_data()
    #divido in training e label
    infT, infL = split_data(inf, 4)
    purT, purL = split_data(pur, 4)
    parT, parL = split_data(par, 4)
    #split di ogni riga in lista di parole
    infT = textSplitter(infT)
    purT = textSplitter(purT)
    parT = textSplitter(parT)

    #costruzione dizionario vuoto con tutte le parole di training
    parts = [infT, purT, parT]
    dict = dictionaryBuilder(parts)
    #riempimento dizionari
    dictInfT = dictionaryFiller(infT, dict)
    dictPurT = dictionaryFiller(purT, dict)
    dictParT = dictionaryFiller(parT, dict)
    eps = 0.001
    p = np.zeros((len(dict.keys()), 3))
    keys = list(dict.keys())
    indexDict = {}
    for i in range(len(keys)):
        indexDict[keys[i]] = i
    infTlen = len(infT)
    purTlen = len(purT)
    parTlen = len(parT)

    for word in keys:
        index = indexDict[word]
        p[index, 0] += np.log((dictInfT[word] + eps) / infTlen)
        p[index, 1] += np.log((dictPurT[word] + eps) / purTlen)
        p[index, 2] += np.log((dictParT[word] + eps) / parTlen)

    lh0, lh1, lh2 = 0, 0, 0
    cnt = 0
    for terzina in infL:
        for word in terzina.split(" "):
            word = word.lower()
            if word in keys:
                lh0 += p[indexDict[word], 0]
                lh1 += p[indexDict[word], 1]
                lh2 += p[indexDict[word], 2]
        if lh0 > lh1 and lh0 > lh2:
            cnt += 1
        lh0, lh1, lh2 = 0, 0, 0

    print("inferno : ", cnt / len(infL))
    cnt = 0

    for terzina in purL:
        for word in terzina.split(" "):
            word = word.lower()
            if word in keys:
                lh0 += p[indexDict[word], 0]
                lh1 += p[indexDict[word], 1]
                lh2 += p[indexDict[word], 2]
        if lh1 > lh0 and lh1 > lh2:
            cnt += 1
        lh0, lh1, lh2 = 0, 0, 0

    print("purgatorio : ", cnt / len(purL))
    cnt = 0

    for terzina in parL:
        for word in terzina.split(" "):
            word = word.lower()
            if word in keys:
                lh0 += p[indexDict[word], 0]
                lh1 += p[indexDict[word], 1]
                lh2 += p[indexDict[word], 2]
        if lh2 > lh0 and lh2 > lh1:
            cnt += 1
        lh0, lh1, lh2 = 0, 0, 0

    print("paradiso : ", cnt/len(parL))













