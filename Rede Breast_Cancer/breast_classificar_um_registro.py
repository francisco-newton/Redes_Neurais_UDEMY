import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')


classificador = Sequential()
# iniciando a rede neural

classificador.add(Dense(units=8, 
                        activation="relu",
                        kernel_initializer="normal",
                        input_dim=30))

classificador.add(Dropout(0.2))

classificador.add(Dense(units=8,
                        activation="relu",
                        kernel_initializer="normal"))

classificador.add(Dropout(0.2))

classificador.add(Dense(units=1,
                        activation='sigmoid'))

classificador.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=['binary_accuracy'])

classificador.fit(previsores, classe, batch_size=10, epochs=100)

novo = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,11,1,1,1,1,1,1,1,1,1,1,1,1,1]])

previsao_nova = classificador.predict(novo)
previsao_nova = (previsao_nova > 0.9)