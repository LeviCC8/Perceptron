import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from functools import reduce


mydata = np.genfromtxt('iris.txt', delimiter = ",")
X = mydata[:, :4].reshape(-1, 4)
Y = mydata[:, -1].reshape(-1, 1)

dataNumber = len(X)
dataNumber08 = int(0.8*dataNumber)
perceptronsNumber = len(np.unique(Y))

lista = np.random.permutation(dataNumber)
samples = lista[:dataNumber08]
tests = lista[dataNumber08:]

#hitRatesPlot = []
#confusionMatrix = []

def train(samples, tests, learningRate, eras):
	totalError = np.ones((perceptronsNumber, 1))
	#hitRatesPlotN = []
	W = np.random.rand((len(X[0]) + 1)*perceptronsNumber).reshape(perceptronsNumber, len(X[0]) + 1)
	for n in range(eras):
		if totalError.sum() == 0:
			break;
		totalError = np.zeros((perceptronsNumber, 1))
		shuffle(samples)
		for a in samples:
			x = np.append(-1, X[a]).reshape(1, -1)
			d = np.zeros((perceptronsNumber, 1))
			d[int(Y[a][0])] = 1
			y = np.array([predict(x[0], w) for w in W]).reshape(-1, 1)
			error = d - y
			totalError += [abs(z) for z in error]
			W = W + learningRate*error@x
		#hitRatesPlotN.append(test(w, tests))
	#hitRatesPlot.append(hitRatesPlotN)
	return W


def predict(x, w):
	y = 1 if np.dot(w, x) >= 0 else 0
	return y

def test(w, tests):
	hitRate = []
	for q in range(perceptronsNumber):
		yhat = [predict(np.append(-1, x), w[q]) for x in X]
		Yd = [1 if (z == q) else 0 for z in Y]
		TP = sum(list(map(lambda x: 1 if ((yhat[x] == Yd[x]) and (Yd[x] == 1)) else 0, tests)))
		TN = sum(list(map(lambda x: 1 if ((yhat[x] == Yd[x]) and (Yd[x] != 1)) else 0, tests)))
		FP = sum(list(map(lambda x: 1 if ((yhat[x] != Yd[x]) and (Yd[x] != 1)) else 0, tests)))
		FN = sum(list(map(lambda x: 1 if ((yhat[x] != Yd[x]) and (Yd[x] == 1)) else 0, tests)))
		confusionMatrixN = np.array([[TP,FP],[FN,TN]])
		#confusionMatrix.append(confusionMatrixN)
		hitRate.append((TP + TN)/(TP + TN + FP + FN))
	return hitRate


W = train(samples, tests, 0.1, 100)
hitRate = test(W, tests)
print(W)
print(hitRate)