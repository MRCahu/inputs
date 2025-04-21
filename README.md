# Prevendo Vendas de Sorvete com Machine Learning 🍦📊

Este projeto aplica um modelo de regressão linear para prever a quantidade de sorvetes vendidos com base na temperatura do dia.

## 🌟 Objetivo
- Prever vendas com base na temperatura do dia.
- Utilizar o MLflow para rastrear experimentos, métricas e artefatos.
- Criar scripts de treinamento e previsão prontos para produção.
- Construir um portfólio de projetos aplicados em Machine Learning.

## 📁 Estrutura do Projeto
```
ice-cream-sales-prediction/
├── inputs/
│   └── dados.csv
├── src/
│   ├── train.py
│   └── predict.py
├── environment.yml
├── README.md
```

## ▶️ Como Executar

### 1. Criar ambiente:
```bash
conda env create -f environment.yml
conda activate icecream-env
```

### 2. Treinar o modelo:
```bash
python src/train.py --data inputs/dados.csv
```

### 3. Fazer previsão:
```bash
python src/predict.py --temperatura 30
```

## 📷 Prints do Projeto
| Gráfico de Dispersão | MLflow Tracking |
|----------------------|------------------|
| ![](https://via.placeholder.com/300x180?text=Grafico+de+Dispersao) | ![](https://via.placeholder.com/300x180?text=MLflow+Tracking) |

## 🔍 Insights
- A temperatura possui correlação direta com o volume de vendas.
- Com base no histórico, é possível planejar melhor a produção e reduzir desperdícios.
- MLflow facilita o rastreamento, comparação e reutilização de modelos.

## 🚀 Tecnologias Utilizadas
- Python 3.10
- Scikit-learn
- MLflow
- Pandas
- Joblib
- Conda

## 📅 Status
Projeto finalizado e pronto para uso no portfólio.

## ⚖️ Licença
Este projeto está licenciado sob os termos da licença MIT.
"""

# environment.yml
"""
name: icecream-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - scikit-learn
  - pandas
  - joblib
  - pip:
      - mlflow
