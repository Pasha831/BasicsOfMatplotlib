import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np

plt.style.use('seaborn-whitegrid')

''' Примитив
fig, ax = plt.subplots()  # 1x1 холст с 1 графиком ax

x = [-3, -2, -1, 0, 1, 2, 3]
y = [a**2 for a in x]
x_abs = [abs(a) for a in x]  # модуль х
x_neg = [-abs(a) for a in x]  # отрицательные значения

ax.plot(x, y)  # рисую на графике ax - просто y
ax.plot(x, x_abs)  # рисую на графике ax - модуль х
ax.plot(x, x_neg)  # рисую на графике ax - отрицательные значения

plt.show()  # вывожу холст fig с графиком ax
'''

''' Отображение графиков на матрице
fig, axes = plt.subplots(2, 2)  # 2x2 график с 4 графиками, axes - матрица из 4-х элементов

x = [-3, -2, -1, 0, 1, 2, 3]
y = [a**2 for a in x]
x_abs = [abs(a) for a in x]  # модуль х
x_neg = [-abs(a) for a in x]  # отрицательные значения

axes[0, 0].plot(x, y)  # первая ячейка - 0, 0 - график с y
axes[0, 1].plot(x, x_abs)  # вторая ячейка - 0, 1 - график модуля
axes[1, 0].plot(x, x_neg)  # третья ячейка - 1, 0 - график минус модуля
# 4-ый график в матрице специльно оставлю пустым

plt.show()
'''

''' Наводим красоту (*суету)
fig, ax = plt.subplots()

x = np.array([-3, -2, -1, 0, 1, 2, 3])
y = np.array([a**2 for a in x])

# дефолтная параболла
ax.plot(x, y)

# график sin(x), где color задаёт цвет, а linestyle - вид линии
ax.plot(x, np.sin(x), color=(1.0, 0.2, 0.3), linestyle='-', marker='*', linewidth=5)
ax.plot(x, x + 5, color='#FFFC0B', linestyle='-.')

plt.show()
'''

''' Лимиты по осям
fig, ax = plt.subplots()

x = np.array([-3, -2, -1, 0, 1, 2, 3])
y = np.array([a**2 for a in x])

ax.plot(x, y, color='red', linestyle=':', marker='o')
ax.set_xlim(0, 5)  # лимит по оси X: только от 0 и только до 5
ax.set_ylim(-5, 10)  # лимит по оси Y: Только от -5 и только до 10

plt.show()
'''

''' Подписи и лэйблы
fig, ax = plt.subplots()

x = np.array([-3, -2, -1, 0, 1, 2, 3])
y = np.array([a**2 for a in x])

# задаём лэйблы - названия - графикам: label=''
ax.plot(x, y, color='red', linestyle='-', marker='o', label='y = x ** 2')
ax.plot(x * 0.5, y, color='green', linestyle=':', marker='^', label='y = 0.5*x ** 2')

ax.set_title('Графики', fontsize=20)  # создаём заголовок графика

ax.set_xlabel('Ось X')  # подписываем ось X
ax.set_ylabel('Ось Y')  # подписываем ось Y

ax.legend(loc='lower left')  # создаём легенду (справ. информацию) о лэйблах в нижнем левом углу графика

plt.show()
'''

''' Баловство с сеткой графика
fig, ax = plt.subplots()

x = np.array([-3, -2, -1, 0, 1, 2, 3])
y = np.array([a**2 for a in x])

ax.plot(x, y, color='red', linestyle='-', marker='o', label='y = x ** 2')

ax.set_xticks(np.array([-3, 3, 0.5]))
ax.set_yticks(np.array([0, 10, 3]))
ax.grid(color='blue', linewidth=2, linestyle='--')

plt.show()
'''

''' Сохранение и открытие (открытие .img - очень кривое :D)
fig, ax = plt.subplots()

x = np.array([-3, -2, -1, 0, 1, 2, 3])
y = np.array([a**2 for a in x])

ax.plot(x, y, color='red', linestyle='-', marker='o', label='y = x ** 2')

# fig.savefig('figure.png')
# plt.show()

img = mpimage.imread('figure.png')
plt.imshow(img)
plt.show()
'''

''' Scatter plot: график рассеивания
fig, ax = plt.subplots()

N = 50

x = np.random.rand(N)
y = np.random.rand(N)

colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2

ax.scatter(x, y, s=area, c=colors, alpha=0.5, marker='o', cmap='viridis')

plt.show()
'''

''' 3D график рассеивания
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(111, projection='3d')

N = 50

x = np.random.rand(N)
y = np.random.rand(N)

colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2

ax.scatter(x, y, s=area, c=colors, alpha=0.5, marker='o', cmap='viridis')

plt.show()
'''

''' Stacked bar charts: наслаивающиеся диаграммы
fig, ax = plt.subplots()

x = np.array([1,2,3])
y = np.array([a + 3 for a in x])
print(y)

ax.bar(x, y + 3, color='blue', width=0.5, alpha=0.8)
ax.bar(x, y, color='red', width=0.5)

plt.show()
'''

''' Hist - гистограмма
fg, ax = plt.subplots()

np.random.seed(19680801)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

ax.hist(x, 50, density=True, facecolor='g', alpha=0.75)

plt.show()
'''

''' Box-plot - график-усики
user_1 = [10, 3, 15, 21, 17, 14]
user_2 = [5, 13, 10, 7, 9, 12]

data = [user_1, user_2]

fig = plt.figure(figsize=(8, 6))

ax = fig.add_axes([0, 0, 1, 1])

bp = ax.boxplot(data)

plt.show()
'''

''' Pie chart - пирог
fig, ax = plt.subplots(figsize=(8, 8))

labels = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'June')

user_1 = [10, 15, 15, 20, 25, 15]

ax.pie(user_1, labels=labels, explode=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1), autopct='%1.1f%%', startangle=75, shadow=True)

plt.show()
'''

''' Time-serires - временной график
import pandas as pd
plt.figure(figsize=(8, 6))
ts = pd.Series(np.random.randn(100), index=pd.date_range('1/1/2020', periods=100))

ts = ts.cumsum()
ts.plot()
plt.show()
'''
