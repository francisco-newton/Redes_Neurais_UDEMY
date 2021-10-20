import numpy as np

def reluFunction(x):
    if x >= 0:
        return x
    else:
        return 0
    # usada em redes neurais convolucionais
    
def linearFunction(x):
    return x
    # usada para regressão linear, somente retorna o valor passado
    
def softmaxFunction(x):
    ex = np.exp(x)
    y = ex/ex.sum()
    return y
    # para problemas de classificação com mais de 2 classes
    
# teste = reluFunction()
# teste = linearFunction(0)
valores = [5.0, 2.0, 1.3]
print(softmaxFunction(valores))