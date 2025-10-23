from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np

# предварительно переводим .txt в .csv
data_set = pd.read_csv('household_power_consumption.csv', sep=';', na_values=['?'])

# Удаляем нечисловые колонки
data_set = data_set.drop(columns=["Date", "Time"])

# Преобразуем все колонки в числовой формат
for column in data_set.columns:
    data_set[column] = pd.to_numeric(data_set[column], errors='coerce') # при ошибках преобразования, конвертируется в NaN

data_set = data_set.dropna() # удаляет строки, где любое значение является NaN

#-----------------
# №1
#-----------------
#Разделяем выборку на train_set и test_set
train_set = data_set.sample(frac=0.8, random_state=42)  #записываем 80% в train_set, с каждым запуском будут выбираться одни
                                                        # и те же строки
test_set = data_set.drop(train_set.index)   #исключаем строки, которые записаны в train_set


# В качестве целевой переменной будет использовано Global_active_power, а все остальное будет признаками

# Разделим нашу выборку на Y и X
y_train_set = train_set["Global_intensity"]
x_train_set = train_set.drop(columns=["Global_intensity"])

# тоже самое сделаем с test_set
y_test_set = test_set["Global_intensity"]
x_test_set = test_set.drop(columns=["Global_intensity"])

print('ПУНКТ 1:\t\tПРОЙДЕН')
#-----------------
# №2
#-----------------
#ЛР по сути предполагает что между признаками и целевой переменной есть линейная зависимость
#Наша задача найти такую прямую которая минимизирует разницу между предсказанным и реальными значениями
# С математической точки зрения y = w_0 + w_1*x_1 + w_2*x_2 + ... + w_n * x_n, где как раз нам нужно найти эти коэффициенты w
# В программном будет использоваться градиентный спуск


model = LinearRegression() # создаем объект класса LinearRegression
model.fit(x_train_set, y_train_set) # по сути запускается алгоритм градиентного спуска

#используем нашу обученную модель, чтобы предсказать какие-то значения
y_train_pred = model.predict(x_train_set)
y_test_pred = model.predict(x_test_set)

print('ПУНКТ 2:\t\tПРОЙДЕН')
#-----------------
# №3
#-----------------
#Точность проверим с помощью коэфициента детерминации - это доля дисперсии зависимой переменной, объясняемая рассматриваемой моделью

r2_train = r2_score(y_train_set, y_train_pred)
r2_test = r2_score(y_test_set, y_test_pred)

#Численная оценка качества решения
print(f'Численная оценка на тренировочном выборке: {r2_train}\nЧисленная оценка на тестовой выборке: {r2_test}')

print('ПУНКТ 3:\t\tПРОЙДЕН')
#-----------------
# №4
#-----------------
# Так как зависимость не всегда линейная, то на помощь приходит ЛР с полинимиальными признаками, которая помогает улавливать нелинейные зависимости
# Будем добавлять в нашу модель ряд признаков, которые являются степенями исходных признаков

degrees = [x for x in range(1,5)]

r2_train_list = [] #список коэффициентов детерминации для тренировочного множества
r2_test_list = [] #список коэффициентов детерминации для тестового множества

#использование линейной регрессии с полиномиальными признаками
for i in range(len(degrees)):
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline( #объект который автоматизирует процесс трансформации признаков в полиномиальные
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )

    pipeline.fit(x_train_set, y_train_set) #запуск градиентного спуска

    y_train_pred = pipeline.predict(x_train_set)
    y_test_pred = pipeline.predict(x_test_set)

    r2_train_list.append(r2_score(y_train_set, y_train_pred))
    r2_test_list.append(r2_score(y_test_set, y_test_pred))

plt.figure(figsize=(10, 6))
plt.plot(degrees, r2_train_list, label="Train R^2", marker='o')
plt.plot(degrees, r2_test_list, label="Test R^2", marker='o')
plt.ylabel("R^2")
plt.xlabel("Степени полиномиальной функции")
plt.legend()
plt.grid(True)
plt.show()

optimal_degree_index = np.argmax(r2_test_list)
optimal_degree = degrees[optimal_degree_index]

print('ПУНКТ 4:\t\tПРОЙДЕН')
#-----------------
# №5
#-----------------
# Чтобы избежать проблемы переобучения  и не дать модели стать слишком сложной
# Вариает решения: ухудшить модель на тренировочных тестах
# В обычной линейной регрессии мы хотели минимизировать сумму квадратов ошибок
# В ЛР с использованием регулирезации будем минимизировать сумму квадратов ошибок + добавлять штраф за большие коэффициенты
# Для чего это делается: когда появляется большой коэффицент, значит модель придает этому признаку слишком много значения, чтобы подстроится под шум
# Чтобы как-то отрегулировать эту важность мы и будем добавлять штраф

#Риски
# Если наше а будет равно 0, тогда мы запустим обычную ЛР и есть шанс переобучить модель
# Если наше а будет очень маленькое, тогда будут добавляться слишком маленькие штрафы, что по сути тоже самое что и выше
# Если наше а будет очень большим тогда мы можем недообучить модель

alpha = np.logspace(-4, 3, 10)

r2_train_list = []
r2_test_list = []

for a in alpha:
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=optimal_degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=a, max_iter=5000))
    ])

    pipeline.fit(x_train_set, y_train_set)

    y_train_pred = pipeline.predict(x_train_set)
    y_test_pred = pipeline.predict(x_test_set)

    r2_train_list.append(r2_score(y_train_set, y_train_pred))
    r2_test_list.append(r2_score(y_test_set, y_test_pred))

plt.figure(figsize=(10, 6))
plt.semilogx(alpha, r2_train_list, label="Train R^2", marker='o', markersize=3)
plt.semilogx(alpha, r2_test_list, label="Test R^2", marker='o', markersize=3)
plt.xlabel("Альфа")
plt.ylabel("R^2")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.show()

# Находим оптимальное alpha
optimal_alpha_index = np.argmax(r2_test_list)
optimal_alpha = alpha[optimal_alpha_index]
print(optimal_alpha)

print('ПУНКТ 5:\t\tПРОЙДЕН')


