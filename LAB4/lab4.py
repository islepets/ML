import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# Чтение CSV файла
data_set = pd.read_csv('final.csv')

#-----------------
# №1
#-----------------
#Разделяем выборку на train_set и test_set
train_set = data_set.sample(frac=0.8, random_state=42)  #записываем 80% в train_set, с каждым запуском будут выбираться одни
                                                        # и те же строки
test_set = data_set.drop(train_set.index)   #исключаем строки, которые записаны в train_set


# В качестве целевой переменной будет использовано target, а все остальное будет признаками

# Разделим нашу выборку на Y и X
y_train_set = train_set["target"]
x_train_set = train_set.drop(columns=["target"])

# тоже самое сделаем с test_set
y_test_set = test_set["target"]
x_test_set = test_set.drop(columns=["target"])

#-----------------
# №2
#-----------------
#Стандартизация данных, т.е. приведение к нудевому среднему и кдиничной дисперсии
# x' = (x - u) / σ
# σ - стандартное отклонние
# u - среднее значение признака
# Стандартизируем, потому что нейронные сети "чувствительны" к масщтабу данных

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_set) #fit_transform используется ТОЛЬКО для обучающих данных
x_test_scaled = scaler.transform(x_test_set) # transform - используется для тестовых

#-----------------
# №3
#-----------------
#Perception - однослойная нейроная сеть, реализует простейшую линейную функцию y = f(w*x+b), решает только линейные задачи
#Функция активации - ступенчатая
perceptron = Perceptron(max_iter=2000, random_state=42) # создаем объект простой линейной модели
perceptron.fit(x_train_scaled, y_train_set)
y_pred_perceptron = perceptron.predict(x_test_scaled)
accuracy_perceptron = accuracy_score(y_test_set, y_pred_perceptron)

#MLPClassifier - многослойная нейронная сеть, каждый нейрон это f(sum(w_i*x_i+b))
#Функции активации - RelU, Tanh, Logistic
mlp = MLPClassifier(max_iter=2000, random_state=42) # создаем объект многослойной нейроной сети
mlp.fit(x_train_scaled, y_train_set)
y_pred_mlp = mlp.predict(x_test_scaled)
accuracy_mlp = accuracy_score(y_test_set, y_pred_mlp)

#-----------------
# №4
#-----------------
print(f"Perceptron: {accuracy_perceptron:.3f}")
print(f"MLPClassifier: {accuracy_mlp:.3f}")

#-----------------
# №5
#-----------------
#Осуществялем подбор гиперпараметров

# [hidden_layer_sizes] - коли-во нейронов в слоях
# [activation] - функции активации
# [solver] - алгоритмы оптимизации
# [alpha] - параметр регулеризации
# [learning_rate_init] - начальная скорость обучения

#Алгоритмы активизации
# Стохастический - используется стохастический градиентный спуск, веса обновляюся после каждого примера
# Adam - адаптивная скорость обучения для каждого параметра, более стабильна и имеется быстрая скорость сходимости

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01], # L2-регулеризация
    'learning_rate_init': [0.001, 0.01]
}

#Использование кросс-валидации:каждая часть становится по переменно тестовой
# train-train-test
# train-test-train
# test-train-train

#Используем самую простую комбинацию параметров(144) и проверяется на 5 фолдах(часть в кросс-валидации)

                                                                           #Раняя остановка при переобучении
grid_search = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
                           param_grid,
                           scoring='accuracy',
                           cv=5, # кросс-валидация
                           n_jobs=-1, #использование всех ядер процессора
                           verbose=1) # вывод рогресса
grid_search.fit(x_train_scaled, y_train_set)

print("\nЛучшие параметры:", grid_search.best_params_)
print("Лучшая точность на валидационной выборке: {:.3f}".format(grid_search.best_score_))

# Оценка лучшей модели на тестовых данных
best_mlp = grid_search.best_estimator_
y_pred_best = best_mlp.predict(x_test_scaled)
accuracy_best = accuracy_score(y_test_set, y_pred_best)
print(f"Точность лучшей модели на тестовых данных: {accuracy_best:.3f}")

#-----------------
# №5 (Визуализация)
#-----------------
results = pd.DataFrame(grid_search.cv_results_)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(results['mean_test_score'], marker='o', alpha=0.7)
ax1.set_xlabel('Номер эксперимента')
ax1.set_ylabel('Средняя точность')
ax1.set_title('Результаты перебора гиперпараметров')
ax1.grid(True, alpha=0.3)

models = ['Perceptron', 'MLP (базовый)', 'MLP (лучший)']
accuracies = [accuracy_perceptron, accuracy_mlp, accuracy_best]
bars = ax2.bar(models, accuracies, color=['blue', 'red', 'green'])
ax2.set_ylabel('Точность')
ax2.set_title('Сравнение точности моделей')
ax2.set_ylim(0, 1.1)
for bar, accuracy in zip(bars, accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{accuracy:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


