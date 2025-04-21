# src/predict.py
import argparse
import pandas as pd
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--temperatura", type=float, required=True, help="Temperatura do dia em graus Celsius")
args = parser.parse_args()

# Carregar modelo
model = joblib.load("modelo_sorvete.pkl")

# Fazer previsão
entrada = pd.DataFrame({"Temperatura": [args.temperatura]})
previsao = model.predict(entrada)
print(f"Previsão de vendas de sorvete para {args.temperatura}ºC: {previsao[0]:.2f} unidades")