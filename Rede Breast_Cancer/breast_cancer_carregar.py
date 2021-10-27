import numpy as np
import pandas as pd
from keras.models import model_from_json

arquivo = open('classificador_breast.json', 'r')
# realiza a leitura do arquivo que contem a estrutura da rede neural

estrutura_rede = arquivo.read()
# pega os valores contidos no arquivo que foi aberto na variavel arquivo

arquivo.close() # fecha o arquivo que foi aberto para liberar memoria
                # realizar o fechamento somente apos a leitura pelo read()

classificador = model_from_json(estrutura_rede)
# cria a rede neural baseada nos parametros salvos no arquivo json
classificador.load_weights('classificador_breast.h5')
# carrega os pesos salvos no arquivo h5

novo = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,11,1,1,1,1,1,1,1,1,1,1,1,1,1]])

previsao_nova = classificador.predict(novo)
previsao_nova = (previsao_nova > 0.9)

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
resultado = classificador.evaluate(previsores, classe)