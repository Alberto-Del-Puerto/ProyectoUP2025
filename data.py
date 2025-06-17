import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Definir una función para generar datos de ejemplo
def generate_student_data():
    file = 'Fulldata.csv'
    gender = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=24)  # 0: mujer, 1: hombre
    adm_score = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=25)  # Calificación de examen de ingreso (50-100)
    enroll_age = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=26)  # Edad al ingresar (17-24)
    grade1 = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=27) # Calificación del parcial 1 (50-100)
    grade2 = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=28)  # Calificación del parcial 2 (50-100)
    grade3 = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=29)  # Calificación del parcial 3 (50-100)
    study_hours = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=30)  # Horas de estudio a la semana (0-30)
    stress_level = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=31)  # Nivel de estrés (1-5)
    extracurricular_activities = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=32)  # Número de actividades extracurriculares (0-10)
    num_courses = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=33)  # Número de cursos inscritos (0-10)
    avg_sleep = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=34)  # Horas promedio de sueño 
    meals_per_day = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=35)  # Número de comidas al día 
    exercise_hours = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=36)  # Horas de ejercicio al día 
    proc_zone = np.genfromtxt(file, delimiter=',', skip_header=1,dtype=str,usecols=37)  # Zona de procedencia
    family_support = np.genfromtxt(file, delimiter=',', skip_header=1,dtype=str,usecols=38)  # Recibe apoyo familiar
    internet_access = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=39)  # Acceso a internet fuera de la escuela (0: no, 1: sí)
    dropout = np.genfromtxt(file, delimiter=',', skip_header=1,usecols=1)  # Etiqueta de abandono (0: no, 1: sí)

    # Crear DataFrame con los datos generados
    data = pd.DataFrame({
        'gender': gender,
        'adm_score': adm_score,
        'enroll_age': enroll_age,
        'grade1': grade1,
        'grade2': grade2,
        'grade3': grade3,
        'study_hours': study_hours,
        'stress_level': stress_level,
        'extracurricular_activities': extracurricular_activities,
        'num_courses': num_courses,
        'avg_sleep': avg_sleep,
        'meals_per_day': meals_per_day,
        'exercise_hours': exercise_hours,
        'proc_zone': proc_zone,
        'family_support': family_support,
        'internet_access': internet_access,
        'dropout': dropout
    })

    return data