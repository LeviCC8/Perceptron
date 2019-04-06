import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from functools import reduce

dataNumber = 150
dataNumber08 = int(0.8*dataNumber)

mydata = np.genfromtxt('iris.txt', delimiter = ",")
X = mydata[:, :-1].reshape(-1, 4)
Y = mydata[:, -1].reshape(-1, 1)

hitRatesPlot = []
confusionMatrix = []

def train(samples, tests, learningRate, eras):
	totalError = 1
	hitRatesPlotN = []
	w = np.random.rand(1,5)
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
	for a in range(realizations):
		lista = np.random.permutation(dataNumber)
		samples = lista[:dataNumber08]
		tests = lista[dataNumber08:]
		w = train(samples, tests, 0.1, 100)
		hitRate = test(w, tests)
		hitRates.append(hitRate)
	accuracy = np.mean(hitRates) 
	standardDeviation = np.std(hitRates)
	informations = {"accuracy": accuracy, "standardDeviation": float(standardDeviation)}
	return informations

informations = accuracyAndStandardDeviation(20)

q = reduce(lambda a, b: a if ((abs((confusionMatrix[a][0][0] + confusionMatrix[a][1][1])/dataNumber*0.2 - informations["accuracy"])) < (abs((confusionMatrix[b][0][0] + confusionMatrix[b][1][1])/dataNumber*0.2 - informations["accuracy"]))) else b, range(len(confusionMatrix)))
greatConfusionMatrix = confusionMatrix[q]

print("greatConfusionMatrix: \n{}".format(greatConfusionMatrix))
print("accuracy: {:.4f}".format(informations["accuracy"]))
print("standardDeviation: {:.4f}".format(informations["standardDeviation"]))

num = reduce(lambda a, b: a if (abs(hitRatesPlot[a][-1] - informations["accuracy"]) < abs(hitRatesPlot[b][-1] - informations["accuracy"]) ) else b, range(len(hitRatesPlot)))
plot = hitRatesPlot[num]
plt.plot(range(len(plot)), plot, 'r')
plt.xlabel("Época")
plt.ylabel("Taxa de acerto")
plt.show()