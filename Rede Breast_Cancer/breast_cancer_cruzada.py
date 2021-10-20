import pandas as pd
# import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede():
    classificador = Sequential()
    # iniciando a rede neural

    classificador.add(Dense(units=16, 
                            activation='relu',
                            kernel_initializer='random_uniform',
                            input_dim=30))
    
    classificador.add(Dropout(0.2))
    # o unico parametro é a porcentagem que sera usada para o dropout
    # essa função serve para prevenir o overfitting da rede neural
    
    classificador.add(Dense(units=16,
                            activation='relu',
                            kernel_initializer='random_uniform'))
    
    classificador.add(Dropout(0.2))
    # outra camada de dropout para a segunda camada de neuronios
    
    classificador.add(Dense(units=1,
                            activation='sigmoid'))
    
    otimizador = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0001)
    
    classificador.compile(optimizer=otimizador,
                          loss='binary_crossentropy',
                          metrics=['binary_accuracy'])
    
    return classificador

classificador = KerasClassifier(build_fn=criarRede,
                                epochs=100,
                                batch_size=10)
# build_fn é a função que efetivamente fará a criação da rede neural, definidia logo a cima
# epochs é o número de epocas que seram utilizadas
# batch_size não lembro o que é

resultado = cross_val_score(estimator=classificador,
                            X=previsores, y=classe,
                            cv=10, scoring='accuracy') 

# estimator classificador
# X é o atributo previsor, y é a classe
# cv é quantas vezes o teste será realizado
# scoring a forma de retornar os resultados

media = resultado.mean()
desvio = resultado.std() # é possível observar o overfitting através do desvio padrao dos dados


    