# Arquivo: src/train.py
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import mlflow

# Inicia o rastreamento MLflow
mlflow.autolog()

def treinar_modelo_vendas(data_path, learning_rate):
    # 1. Carregar Dados (Exemplo Simulado)
    # Na prática, você usaria um Dataset registrado no Azure ML
    data = pd.read_csv(data_path)
    X = data[['Temperatura']] # Feature de entrada
    y = data['Vendas']       # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Treinar o Modelo de Regressão
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    # 3. Avaliar
    previsoes = modelo.predict(X_test)
    rmse = mean_squared_error(y_test, previsoes, squared=False)

    # 4. Registrar com MLflow
    # mlflow.autolog() já registra métricas, mas vamos registrar manualmente a RMSE como exemplo
    mlflow.log_metric("rmse", rmse)
    mlflow.log_param("algoritmo", "LinearRegression")
    mlflow.log_param("learning_rate", learning_rate) # Parâmetro injetado via argparse

    print(f"Modelo treinado com RMSE: {rmse}")
    
    # 5. Registrar o Modelo (Artefato)
    mlflow.sklearn.log_model(modelo, "modelo_gelato")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Define a entrada de dados (passada pelo pipeline)
    parser.add_argument("--training_data", type=str, help="Caminho para o conjunto de dados de treinamento")
    # Define um hiperparâmetro de exemplo (embora não usado em LR, é fundamental para outros modelos)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    
    args = parser.parse_args()
    
    treinar_modelo_vendas(args.training_data, args.learning_rate)
