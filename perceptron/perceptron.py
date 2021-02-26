#!/usr/bin/env python3
import numpy as np

#Define o número de épocas e o numero de amostras 
numEpocas = 100000
q = 6

#Atributos 
peso = np.array([113, 122, 107, 98, 115, 120])
ph = np.array([6.8, 4.7, 5.2, 3.6, 2.9, 4.2])

#Bias 
bias = 1 

#Entrada do percecptron
X = np.vstack((peso,ph))
Y = np.array([-1, 1, -1, -1, 1, 1])

#Taxa de aprendizado
eta = 0.1

#Define o vetor de pesos 
W = np.zeros([1,3])  #Duas entradas + o bias (vetor de 3 colunas inicializado com zeros)

#Array para armazenar os erros
e = np.zeros(6)

#Função de ativação: degrau bipolar
def funcaoAtivacao(valor):
    if valor < 0.0:
        return(-1)
    else:
        return(1)

#Início do programa principal
for j in range(numEpocas):
    for k in range(q):

        #Insere o bias no vetor de entrada
        Xb = np.hstack( (bias, X[:,k]) )

        #Calcula o campo induzido
        V = np.dot( W , Xb)

        #Calcula a saída do percectron
        Yr = funcaoAtivacao(V)

        #Calcula o erro
        e[k] = Y[k] - Yr

        #Treinamento do percpron
        W = W + eta*e[k] * Xb
    
print( "Vetor de erros (e) " + str(e))