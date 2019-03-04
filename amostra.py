class Amostra:
	
	def __init__(self):
		arquivo = open("iris.txt", 'r')
		self.__linhasDoIris = arquivo.readlines()
		arquivo.close()

	def entrada(self, numeroDaAmostra):
		entradas_saida = self.__linhasDoIris[numeroDaAmostra].split(',')
		self.__entrada = list(map(lambda x: float (x), entradas_saida[:4]))
		return self.__entrada

	def saida(self, numeroDaAmostra):
		entradas_saida = self.__linhasDoIris[numeroDaAmostra].split(',')
		self.__saida = entradas_saida[-1]
		return self.__saida.replace('\n', '')