import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from data import generate_student_data

# Generar datos de ejemplo para 6964 alumnos
data = generate_student_data()

# Convertir variables categóricas a numéricas usando One-Hot Encoding
categorical_cols = ['proc_zone', 'family_support']

# Reemplazar 'Yes' y 'No' con 1 y 0 respectivamente en 'family_support'
data['family_support'] = data['family_support'].replace({'yes': 1, 'no': 0})

# Aplicar One-Hot Encoding solo a 'proc_zone'
data_encoded = pd.get_dummies(data, columns=['proc_zone'], drop_first=True)

# Preprocesamiento de datos
features = data_encoded.drop('dropout', axis=1)  # Todas las columnas menos 'dropout' son características
labels = data_encoded['dropout']  # 'dropout' es la etiqueta que queremos predecir

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Escalar las características para normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir el modelo de red neuronal
model = Sequential()

# Capa de entrada y primera capa oculta
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))

# Segunda capa oculta
model.add(Dense(32, activation='relu'))

# Capa de salida
model.add(Dense(1, activation='sigmoid'))  # 'sigmoid' para problemas de clasificación binaria

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callback para detener el entrenamiento si la pérdida en el conjunto de validación deja de disminuir
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Plot de curvas de aprendizaje
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot de curvas de precisión
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')

# Obtener las predicciones del modelo para el conjunto de prueba
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión como un gráfico
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Dropout', 'Dropout'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
