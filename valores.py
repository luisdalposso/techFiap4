import random
import json

# gera 60 numeros entre 100 e 300
valores = [round(random.uniform(100.0, 300.0), 2) for _ in range(60)]

# constrói a string
entrada_api = {"prices": valores}

# valida e exibe o json
print(json.dumps(entrada_api))