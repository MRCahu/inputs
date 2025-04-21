# src/train.py
import pandas as pd
import joblib
import argparse
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="inputs/dados.csv", help="Caminho para o CSV de entrada")
args = parser.parse_args()

# Leitura dos dados
df = pd.read_csv(args.data)
X = df[["Temperatura"]]
y = df["Vendas"]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Habilitar MLflow
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    joblib.dump(model, "modelo_sorvete.pkl")
    mlflow.log_artifact("modelo_sorvete.pkl")