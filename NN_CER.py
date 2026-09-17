#-------------------------------------------------
# 1. Importação de bibliotecas
#-------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#---------------------------------------------------
# 2. Importação do dataset da UCI
#---------------------------------------------------

dataset = fetch_ucirepo(id=235)

# print(dataset.metadata)
# print(dataset.variables)

df = dataset.data.original.copy()
# print(df.head())
# print(df.shape)

# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

#---------------------------------------------------
# 3. Criando o timestamp
#---------------------------------------------------

df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format = "%d/%m/%Y %H:%M:%S"
)

#print(df[["Date","Time", "datetime"]].head())

#---------------------------------------------------
# 4. Organizando a série temporal
#---------------------------------------------------

df = df.set_index("datetime")
df = df.sort_index()

#---------------------------------------------------
# 5. Convertendo valores numéricos
#---------------------------------------------------

numeric_columns = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

#---------------------------------------------------
# 6. Interpolação
#---------------------------------------------------

df[numeric_columns] = (
    df[numeric_columns]
    .interpolate(method="time")
)

#print(df[numeric_columns].isnull().sum())

#---------------------------------------------------
# 7. COnvertendo dados por hora
#---------------------------------------------------

hourly = df["Global_active_power"].resample("1h").mean()
#print(hourly.head())

#---------------------------------------------------
# 8. Visualizando o consumo
#---------------------------------------------------

#plt.figure(figsize=(15, 5))

#plt.plot(hourly)

#plt.title("Consumo médio de energia elétrica por hora")
#plt.xlabel("Data")
#plt.ylabel("Global Active Power (kW)")

#plt.show()

#---------------------------------------------------
# 9. Criando as janelas de tempo
#---------------------------------------------------

def create_sequences(data, window_size):

    x = []
    y = []

    for i in range(window_size, len(data)):

        x.append(data[i-window_size:i])
        y.append(data[i])

    return np.array(x), np.array(y)

window_size = 24

x, y = create_sequences(
    hourly.values,
    window_size
)

#print(x.shape)
#print(y.shape)

#---------------------------------------------------
# Vizualizando uma amostra
#---------------------------------------------------

#print("Entrada:")
#print(x[0])

#print("\n Saída esperada:")
#print(y[0])

#-----------------------------------------------------
# 10. Dividindo os dados em treino, validação e teste
#-----------------------------------------------------

Train_size = int(len(x) * 0.7)
val_size = int(len(x) * 0.15)

x_train = x[:Train_size]
y_train = y[:Train_size]

x_val = x[Train_size:Train_size + val_size]
y_val = y[Train_size:Train_size + val_size]

x_test = x[Train_size + val_size:]
y_test = y[Train_size + val_size:]

#-----------------------------------------------------
# 11. Normalizando os dados
#-----------------------------------------------------

scaler = StandardScaler()
scaler.fit(x_train)

X_train_scaled = scaler.transform(x_train)
X_val_scaled = scaler.transform(x_val)
X_test_scaled = scaler.transform(x_test)

target_scaler = StandardScaler()

target_scaler.fit(y_train.reshape(-1, 1))

y_train_scaled = target_scaler.transform(
    y_train.reshape(-1, 1)
)

y_val_scaled = target_scaler.transform(
    y_val.reshape(-1, 1)
)

y_test_scaled = target_scaler.transform(
    y_test.reshape(-1, 1)
)

#-----------------------------------------------------
# 12. Hiperparametros
#-----------------------------------------------------

N_neurons_layer_1 = 96
N_neurons_layer_2 = 64

lr = 0.001

batch_size = 32

epochs = 30

#-----------------------------------------------------
# 13. Criando o modelo
#-----------------------------------------------------

model = keras.Sequential([

    layers.Dense(
        N_neurons_layer_1,
        activation="relu",
    ),

    layers.Dense(
        N_neurons_layer_2,
        activation="relu",
    ),

    layers.Dense(
        1,
        activation="linear",
    )
])

#-----------------------------------------------------
# 14. Compilando o modelo
#-----------------------------------------------------

optimizer = keras.optimizers.Adam(
    learning_rate=lr
)

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mae"]
)

#-----------------------------------------------------
# 15. Treinando o modelo
#-----------------------------------------------------

history = model.fit(
    X_train_scaled,
    y_train_scaled,

    validation_data=(
        X_val_scaled, 
        y_val_scaled),

    epochs=epochs,
    batch_size=batch_size,

    verbose=1
)

#-----------------------------------------------------
# 16. Fazendo previsões
#-----------------------------------------------------

y_pred_scaled = model.predict(
    X_test_scaled
)
# Voltando para a escala original:
y_pred = target_scaler.inverse_transform(
    y_pred_scaled
).flatten()

y_teste_original = y_test  

#-----------------------------------------------------
# 17. Avaliando o modelo
#-----------------------------------------------------

mae = mean_absolute_error(
    y_teste_original,
    y_pred
)

mse = mean_squared_error(
    y_teste_original,
    y_pred
)

rmse = np.sqrt(mse)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

#-----------------------------------------------------
# 18. Visualizando o histórico de treinamento
#-----------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Evolução da Loss")
plt.xlabel("Épocas")
plt.ylabel("MSE")

plt.legend(["Treino", "Validação"])

plt.show()

#-----------------------------------------------------
# 19. Visualizando as previsões vs valores reais
#-----------------------------------------------------

plt.figure(figsize=(15, 5))

plt.plot(y_teste_original[:300], 
         label="Valores Reais"
)

plt.plot(y_pred[:300], 
        label="Previsões"
)

plt.title("Consumo real vs Consumo previsto")

plt.xlabel("Hora")
plt.ylabel("Global Active Power (kW)")

plt.legend()

plt.show()