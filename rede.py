import numpy as np
from amostra import Amostra
from random import shuffle
import matplotlib.pyplot as plt

taxaDeAprendizado = 0.1

lista = list(range(150))
shuffle(lista)
amostras = lista[:120]
testes = lista[120:]

w0 = np.array([-0.521, 0.543, -0.510, -0.380, 0.390])
w1 = np.array([-0.521, 0.543, -0.510, -0.380, 0.390])
w2 = np.array([-0.521, 0.543, -0.510, -0.380, 0.390])
listaDeAcertos_w0 = []
listaDeAcertos_w1 = []
listaDeAcertos_w2 = []

listaDeAmostrasETestes = Amostra()
#Iris-setosa = 0
#Iris-versicolor = 1
#Iris-virginica = 2

def vetorDePesos(planta, w, listaDeAcertos):
	possiveisSaidas = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
	erroTotal = 1
	for epocas in range(100):
		if erroTotal == 0:
			break
		erroTotal = 0
		for a in amostras:
			saidaDaAmostra = possiveisSaidas.index(listaDeAmostrasETestes.saida(a))
			if saidaDaAmostra == planta:
				saidaDesejada = 1
			else:
				saidaDesejada = 0
			entradaDaAmostra = np.array([-1] + listaDeAmostrasETestes.entrada(a))
			saida = saidaAtual(w, entradaDaAmostra)
			erro = saidaDesejada - saida
			erroTotal += abs(erro)
			w = w + taxaDeAprendizado*erro*entradaDaAmostra
		listaDeAcertos.append(taxaDeAcerto(planta, w))
	return w

def taxaDeAcerto(planta, w):
	acertos = 0
	possiveisSaidas = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
	for n in testes:
		saidaDoTeste = possiveisSaidas.index(listaDeAmostrasETestes.saida(n))
		if saidaDoTeste == planta:
			saidaDesejada = 1
		else:
			saidaDesejada = 0
		entradaDoTeste = np.array([-1] + listaDeAmostrasETestes.entrada(n))
		saida = saidaAtual(w, entradaDoTeste)
		if saida == saidaDesejada:
			acertos += 1
	return (acertos/30)*100

def saidaAtual(w, x):
	produtoInterno = (w*x).sum()
	saidaAtual = 1 if produtoInterno >= 0 else 0
	return saidaAtual

#TREINAMENTO DA IRIS-SETOSA
w0 = vetorDePesos(0, w0, listaDeAcertos_w0,)

#TREINAMENTO DA IRIS-VERSICOLOR
w1 = vetorDePesos(1, w1, listaDeAcertos_w1)

#TREINAMENTO DA IRIS-VIRGINICA
w2 = vetorDePesos(2, w2, listaDeAcertos_w2)

print("Vetor w0: {}".format(w0))
print("Vetor w1: {}".format(w1))
print("Vetor w2: {}".format(w2))

print("Taxa de acerto da Setosa(w0): {} %".format(taxaDeAcerto(0, w0)))
print("Taxa de acerto da Versicolor(w1): {} %".format(taxaDeAcerto(1, w1)))
print("Taxa de acerto da Virginica(w2): {} %".format(taxaDeAcerto(2, w2)))

plt.plot(range(len(listaDeAcertos_w0)), listaDeAcertos_w0, 'r')
plt.plot(range(len(listaDeAcertos_w1)), listaDeAcertos_w1, 'b')
plt.plot(range(len(listaDeAcertos_w2)), listaDeAcertos_w2, 'g')
plt.title("Taxa de acerto em função das épocas")
plt.xlabel("Época")
plt.ylabel("Taxa de acerto")

plt.show()