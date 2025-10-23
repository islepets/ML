import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

def gen_csv():
    rows = 430
    x1 = np.linspace(0, 100, rows)
    x2 = np.linspace(-50, 50, rows)
    y = 0.01 * np.tan(x1) * np.power(x2, 2)

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })
    df.to_csv('param.csv', index=False, encoding='utf-8')

def y_x1(data):
    pl.figure(figsize=(10, 10))
    pl.subplot(2, 2, 1)
    #pl.plot(data['x1'], data['y'], 'b-', linewidth=1)
    pl.scatter(data['x1'], data['y'], color='red', s=5)
    pl.xlabel('x1')
    pl.ylabel('y')
    pl.grid(True, alpha=0.3)
    #pl.tight_layout()
    pl.show()

def y_x2(data):
    pl.figure(figsize=(10, 10))
    pl.subplot(2, 2, 1)
    #pl.plot(data['x2'], data['y'], 'r-', linewidth=1)
    pl.scatter(data['x2'], data['y'], color='blue', s=5)
    pl.xlabel('x2')
    pl.ylabel('y')
    pl.grid(True, alpha=0.3)
    pl.show()

def _3d(data):
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data['x1'], data['x2'], data['y'])
    pl.show()

def save_str(data):
    sr_x1 = data['x1'].mean()
    sr_x2 = data['x2'].mean()
    cond = data[(data['x1'] < sr_x1) | (data['x2'] < sr_x2)]
    cond.to_csv('new_csv.csv', index=False, encoding='utf-8')

def main():
    param = int(input('''Управление программой:
        1 - генерация таблицы
        2 - постоить графики
        3 - вывести среднее, максимальное и минимальное для каждого столбца
        4 - сохранить в новый csv файл те строки, для которых выполняется условие
        5 - построить 3D график
    '''))
    data = pd.read_csv('param.csv')

    if param == 1:
        gen_csv()
        print('Сгенерировано!')

    if param == 2:
        y_x1(data)
        y_x2(data)
    if param == 3:
        print(f'Среднее значение столбца x1: {data['x1'].mean()},\t'
                f'Максимальное значение столбца x1: {data['x1'].max()},\t'
                  f'Минимальное значение столбца x1: {data['x1'].min()},\t')
        print(f'Среднее значение столбца x2: {data['x2'].mean()},\t'
                  f'Максимальное значение столбца x2: {data['x2'].max()},\t'
                  f'Минимальное значение столбца x2: {data['x2'].min()},\t')
        print(f'Среднее значение столбца y: {data['y'].mean()},\t'
                  f'Максимальное значение столбца y: {data['y'].max()},\t'
                  f'Минимальное значение столбца y: {data['y'].min()},\t')
    if param == 4:
        save_str(data)

    if param == 5:
        _3d(data)

if __name__ == '__main__':
    main()