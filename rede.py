import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from functools import reduce


lista = list(range(150))
shuffle(lista)
samples = lista[:120]
tests = lista[120:]

mydata = np.genfromtxt('iris.txt', delimiter = ",")
X = mydata[:, :-1].reshape(-1, 4)
Y = mydata[:, -1].reshape(-1, 1)

hitRatesPlot = []
confusionMatrix = []

def train(Xtr, Ytr, learningRate, eras):
	totalError = 1
	hitRatesPlotN = []
	w = np.random.rand(1,5)
	for n in range(eras):
		if totalError == 0:
			break;
		totalError = 0
		shuffle(samples)
		for a in samples:
			x = np.append(-1, Xtr[a])
			d = Ytr[a]
			y = predict(x, w)
			error = d - y
			totalError += abs(error)
			w = w + learningRate*error*x
		hitRatesPlotN.append(test(Xtr, Ytr, w))
	hitRatesPlot.append(hitRatesPlotN)
	return w


def predict(x, w):
	y = 1 if np.dot(w, x) >= 0 else 0
	return y

def test(Xte, Yte, w):
	yhat = [predict(np.append(-1, x), w) for x in Xte]
	TP = sum(list(map(lambda x: 1 if ((yhat[x] == Yte[x]) and (Yte[x] == 1)) else 0, tests)))
	TN = sum(list(map(lambda x: 1 if ((yhat[x] == Yte[x]) and (Yte[x] != 1)) else 0, tests)))
	FP = sum(list(map(lambda x: 1 if ((yhat[x] != Yte[x]) and (Yte[x] != 1)) else 0, tests)))
	FN = sum(list(map(lambda x: 1 if ((yhat[x] != Yte[x]) and (Yte[x] == 1)) else 0, tests)))
	confusionMatrixN = np.array([[TP,FP],[FN,TN]])
	confusionMatrix.append(confusionMatrixN)
	hitRate = (TP + TN)/(TP + TN + FP + FN)
	return hitRate

def accuracyAndStandardDeviation(Xte, Yte, realizations):
	hitRates = []
	for a in range(realizations):
		w = train(Xte, Yte, 0.1, 100)
		hitRate = test(Xte, Yte, w)
		hitRates.append(hitRate)
	accuracy = np.mean(hitRates) 
	standardDeviation = np.std(hitRates)
	informations = {"accuracy": accuracy, "standardDeviation": float(standardDeviation)}
	return informations

w = train(X, Y, 0.1, 100)
informations = accuracyAndStandardDeviation(X, Y, 20)

q = reduce(lambda a, b: a if ((abs((confusionMatrix[a][0][0] + confusionMatrix[a][1][1])/len(tests) - informations["accuracy"])) < (abs((confusionMatrix[b][0][0] + confusionMatrix[b][1][1])/len(tests) - informations["accuracy"]))) else b, range(len(confusionMatrix)))
greatConfusionMatrix = confusionMatrix[q]

print("vector w: {}".format(w))
print("greatConfusionMatrix: \n{}".format(greatConfusionMatrix))
print("accuracy: {:.4f}".format(informations["accuracy"]))
print("standardDeviation: {:.4f}".format(informations["standardDeviation"]))

num = reduce(lambda a, b: a if (abs(hitRatesPlot[a][-1] - informations["accuracy"]) < abs(hitRatesPlot[b][-1] - informations["accuracy"]) ) else b, range(len(hitRatesPlot)))
plot = hitRatesPlot[num]
plt.plot(range(len(plot)), plot, 'r')
plt.xlabel("Época")
plt.ylabel("Taxa de acerto")
plt.show()
