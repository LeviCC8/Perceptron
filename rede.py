import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from functools import reduce

dataNumber = 150
dataNumber08 = int(0.8*dataNumber)

mydata = np.genfromtxt('iris.txt', delimiter = ",")
X = mydata[:, :2].reshape(-1, 2)
Y = mydata[:, -1].reshape(-1, 1)

hitRatesPlot = []
confusionMatrix = []

def train(samples, tests, learningRate, eras):
	totalError = 1
	hitRatesPlotN = []
	w = np.random.rand(len(X[0]) + 1)
	for n in range(eras):
		if totalError == 0:
			break;
		totalError = 0
		shuffle(samples)
		for a in samples:
			x = np.append(-1, X[a])
			d = Y[a]
			y = predict(x, w)
			error = d - y
			totalError += abs(error)
			w = w + learningRate*error*x
		hitRatesPlotN.append(test(w, tests))
	hitRatesPlot.append(hitRatesPlotN)
	return w


def predict(x, w):
	y = 1 if np.dot(w, x) >= 0 else 0
	return y

def test(w, tests):
	yhat = [predict(np.append(-1, x), w) for x in X]
	TP = sum(list(map(lambda x: 1 if ((yhat[x] == Y[x]) and (Y[x] == 1)) else 0, tests)))
	TN = sum(list(map(lambda x: 1 if ((yhat[x] == Y[x]) and (Y[x] != 1)) else 0, tests)))
	FP = sum(list(map(lambda x: 1 if ((yhat[x] != Y[x]) and (Y[x] != 1)) else 0, tests)))
	FN = sum(list(map(lambda x: 1 if ((yhat[x] != Y[x]) and (Y[x] == 1)) else 0, tests)))
	confusionMatrixN = np.array([[TP,FP],[FN,TN]])
	confusionMatrix.append(confusionMatrixN)
	hitRate = (TP + TN)/(TP + TN + FP + FN)
	return hitRate


def accuracyAndStandardDeviation(realizations):
	hitRates = []
	weightVectors = []
	for a in range(realizations):
		lista = np.random.permutation(dataNumber)
		samples = lista[:dataNumber08]
		tests = lista[dataNumber08:]
		w = train(samples, tests, 0.1, 100)
		hitRate = test(w, tests)
		hitRates.append(hitRate)
		weightVectors.append(w)
	accuracy = np.mean(hitRates) 
	standardDeviation = np.std(hitRates)
	bestWeightVector = reduce(lambda a, b: a if (abs(test(a, tests) - accuracy) < abs(test(b, tests) - accuracy)) else b,weightVectors)
	informations = {"accuracy": accuracy, "standardDeviation": float(standardDeviation), "w": bestWeightVector}
	return informations

def functionPlot(x1):
	w = informations['w']
	x2 = (-w[1]*x1 + w[0])/w[2]
	return x2

informations = accuracyAndStandardDeviation(20)
q = reduce(lambda a, b: a if ((abs((confusionMatrix[a][0][0] + confusionMatrix[a][1][1])/dataNumber*0.2 - informations["accuracy"])) < (abs((confusionMatrix[b][0][0] + confusionMatrix[b][1][1])/dataNumber*0.2 - informations["accuracy"]))) else b, range(len(confusionMatrix)))
greatConfusionMatrix = confusionMatrix[q]

print("greatConfusionMatrix: \n{}".format(greatConfusionMatrix))
print("accuracy: {:.4f}".format(informations["accuracy"]))
print("standardDeviation: {:.4f}".format(informations["standardDeviation"]))

#Plot hitRate X Eras
plt.figure(1)
num = reduce(lambda a, b: a if (abs(hitRatesPlot[a][-1] - informations["accuracy"]) < abs(hitRatesPlot[b][-1] - informations["accuracy"]) ) else b, range(len(hitRatesPlot)))
bestHitRate = hitRatesPlot[num]
plt.plot(range(len(bestHitRate)), bestHitRate, 'r')
plt.xlabel("Era")
plt.ylabel("Hit rate")

#Plot Hyperplane
plt.figure(2)
hyperplaneX = np.linspace(X[:, :1].min() - 1, X[:, :1].max() + 1, 1000)
hyperplaneY = [functionPlot(a) for a in hyperplaneX]
x1High = [(X[:, :1][a] if Y[a] == 1 else None) for a in range(len(Y))]
x2High = [(X[:, 1:2][a] if Y[a] == 1 else None) for a in range(len(Y))]
x1Low = [(X[:, :1][a] if Y[a] == 0 else None) for a in range(len(Y))]
x2Low = [(X[:, 1:2][a] if Y[a] == 0 else None) for a in range(len(Y))]
plt.plot(hyperplaneX, hyperplaneY, 'k', x1High, x2High, 'r^', x1Low, x2Low, 'bo')
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()