import numpy as np
import func_ativacao_II as f2

def stepfunction(x):
    if x >= 1:
        return 1
    else:
        return 0
    # usada somente em problemas linearmente separaveis
    
def sigmoidfunction(x):
    y = 1 / (1 + np.exp(-x))
    return y
    # usada para classificação binaria (retorna valores entre 0 e 1)
    
def hyperbolic_tanget(x):
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return y
    # tambem serve para classificar valores (retorna valores entre -1 e 1)

print(sigmoidfunction(2.1),
      hyperbolic_tanget(2.1),
      f2.reluFunction(2.1),
      f2.linearFunction(2.1))