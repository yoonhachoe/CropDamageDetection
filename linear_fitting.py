import csv
import numpy as np

f1 = open('Green_training.csv', 'r', encoding='utf-8')
r = csv.reader(f1)
data_path_class_list = sorted(line for line in r)
x = []
y = []

for i in range(8850):
    x.append(float(data_path_class_list[i][3]))
    y.append(float(data_path_class_list[i][1]))

linear_model = np.polyfit(x, y, 1)
a, b = np.polyfit(x, y, 1)
print(a, b)

f1.close()
