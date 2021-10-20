import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score


previsores = pd.read_csv('entradas_breast.csv')
# faz a  leitura do arquivo csv com os dados de entrada (previsores)

classe = pd.read_csv('saidas_breast.csv')
# faz a leitura do arquivo csv com os dados de saída (respostas)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)
# faz a divisão dos dados para que seja possivel criar um DataFrame de treinamento e um de teste

classificador = Sequential()
# iniciando a rede neural

classificador.add(Dense(units=16,
                        activation='relu',
                        kernel_initializer='random_uniform',
                        input_dim=30))
classificador.add(Dense(units=16,
                        activation='relu',
                        kernel_initializer='random_uniform'))

classificador.add(Dense(units=1,
                        activation='sigmoid'))

# parametros para o Dense()
# units - quantidade de neuronios na camada oculta
# formula para qtd de neuronios = (n_de_entradas+n_de_saida) / 2
# no exemplo estudado (30+2)/2 ~ 16
# activation - função de ativação utilizada
# kernel_initializer - forma de inicialização dos pesos
# input_dim - quantidade de entradas na camada de entrada, somente na primeira camada oculta

otimizador = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0001)
# otimizador ADAM serve para realizar o aprendizado da rede neural
# lr - learning_rate é o valor da taxa de aprendizado (o padrão é 0.001)
# decay é o valor do decaimento da taxa de aprendizado

classificador.compile(optimizer=otimizador,
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
'''classificador.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])'''

# parametros para o compile()
# optimizer qual é a função utilizada para fazer o ajuste dos pesos
# loss função de perda, responsavel por realizar o calculo do erro
# metrics calcula a precisão da rede criada

classificador.fit(previsores_treinamento, 
                  classe_treinamento,
                  batch_size=10,
                  epochs=100)
# o fit serve para realizar o treinamento da rede

pesos0 = classificador.layers[0].get_weights()
# os valores obtidos aqui são os pesos efetivos gerados durante o treinamento 
# da rede, podemos especificar os pesos de qual camada queremos
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()


previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5) # atribui verdadeiro para valores maiores que 0,5
# a função predict utiliza o conjunto passado para realizar a operação de teste
# na rede neural criada, retorna valor de probabilidade (resultado da função sigmoide)

precisao = accuracy_score(classe_teste, previsoes)
# o ACCURACY_SCORE compara 2 vetores matriciais

matriz = confusion_matrix(classe_teste, previsoes)
# é possível gerar uma matrix de comparação entre os valores estimados e os
# resultados da rede neural, deve ser passado a classe de teste e as respostas

resultado = classificador.evaluate(previsores_teste, classe_teste)

# modificação para testar o upload para o github 15:36 20/10/2021
