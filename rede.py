import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

#Iris-setosa = 0
#Iris-versicolor = 1
#Iris-virginica = 2


lista = list(range(150))
shuffle(lista)
samples = lista[:120]
tests = lista[120:]

mydata = np.genfromtxt('iris.txt', delimiter = ",")
X = mydata[:, :-1].reshape(-1, 4)
Y = mydata[:, -1].reshape(-1, 1)

setosa = {"num": 0}
versicolor = {"num": 1}
virginica = {"num": 2}

def train(xtr, ytr, plant):
	totalError = 1
	learningRate = 0.1
	plant["hitRatesPlot"] = []
	plant["w"] = np.array([-0.521, 0.543, -0.510, -0.380, 0.390])
	for eras in range(100):
		if totalError == 0:
			break;
		totalError = 0
		shuffle(samples)
		for a in samples:
			x = np.append(-1, xtr[a])
			yd = actualOutput(ytr[a], plant["num"])
			y = predict(x, plant["w"])
			error = yd - y
			totalError += abs(error)
			plant["w"] = plant["w"] + learningRate*error*x
		plant["hitRatesPlot"].append(test(xtr, ytr, plant))


def predict(x, w):
	y = 1 if (w*x).sum() >= 0 else 0
	return y

def actualOutput(y, num):
	yd = 1 if y == num else 0
	return yd

def test(xte, yte, plant):
	TP = sum(list(map(lambda x: 1 if ((predict(np.append(-1, xte[x]), plant["w"]) == actualOutput(yte[x], plant["num"])) and (yte[x] == plant["num"])) else 0,tests)))
	TN = sum(list(map(lambda x: 1 if ((predict(np.append(-1, xte[x]), plant["w"]) == actualOutput(yte[x], plant["num"])) and (yte[x] != plant["num"])) else 0,tests)))
	FP = sum(list(map(lambda x: 1 if ((predict(np.append(-1, xte[x]), plant["w"]) != actualOutput(yte[x], plant["num"])) and (yte[x] != plant["num"])) else 0,tests)))
	FN = sum(list(map(lambda x: 1 if ((predict(np.append(-1, xte[x]), plant["w"]) != actualOutput(yte[x], plant["num"])) and (yte[x] == plant["num"])) else 0,tests)))
	confusionMatrix = np.array([[TP,FP],[FN,TN]])
	hitRate = (TP + TN)/30
	return hitRate

def accuracyAndStandardDeviation(xte, yte, plant):
	hitRates = []
	for a in range(100):
		train(xte, yte, plant)
		hitRate = test(xte, yte, plant)
		hitRates.append(hitRate)
	accuracy = (sum(hitRates))/len(hitRates) 
	standardDeviation = ((sum(map(lambda a: (a - accuracy)**2, hitRates)))/len(hitRates))**(1/2) 
	plant["accuracy"] = accuracy
	plant["standardDeviation"] = float(standardDeviation)

train(X, Y, setosa)
train(X, Y, versicolor)
train(X, Y, virginica)

accuracyAndStandardDeviation(X, Y, setosa)
accuracyAndStandardDeviation(X, Y, versicolor)
accuracyAndStandardDeviation(X, Y, virginica)

print("vetor w0: {}".format(setosa["w"]))
print("accuracy: {:.2f}".format(setosa["accuracy"]))
print("standardDeviation: {:.4f}".format(setosa["standardDeviation"]))
print("vetor w1: {}".format(versicolor["w"]))
print("accuracy: {:.2f}".format(versicolor["accuracy"]))
print("standardDeviation: {:.4f}".format(versicolor["standardDeviation"]))
print("vetor w2: {}".format(virginica["w"]))
print("accuracy: {:.2f}".format(virginica["accuracy"]))
print("standardDeviation: {:.4f}".format(virginica["standardDeviation"]))

plt.plot(range(len(setosa["hitRatesPlot"])), setosa["hitRatesPlot"], 'r')
plt.plot(range(len(versicolor["hitRatesPlot"])), versicolor["hitRatesPlot"], 'b')
plt.plot(range(len(virginica["hitRatesPlot"])), virginica["hitRatesPlot"], 'g')
plt.title("Taxa de acerto em função das épocas")
plt.xlabel("Época")
plt.ylabel("Taxa de acerto")
plt.show()