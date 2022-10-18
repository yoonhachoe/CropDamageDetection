import csv
import numpy as np

def test():
    f1 = open('Green_test.csv', 'r', encoding='utf-8')
    f2 = open('test_linear.csv', 'w', encoding='utf-8')
    r = csv.reader(f1)
    w = csv.writer(f2)
    w.writerow(['gt', 'pred', 'quality'])
    data_path_class_list = sorted(line for line in r)
    x = []
    y = []
    for i in range(935):
        x.append(float(data_path_class_list[i][3]))
        y.append(float(data_path_class_list[i][1]))
    a = -0.4882624606820061
    b = 83.22858573956604
    predicted_y = [value * a + b for value in x]
    sum_test_loss = 0.0
    for i in range(935):
        test_loss = (y[i] - predicted_y[i])**2
        sum_test_loss += test_loss
        w.writerow([y[i], predicted_y[i], data_path_class_list[i][2]])
    print('test loss(RMSE)', np.sqrt(sum_test_loss / 935))
    f1.close()
    f2.close()

if __name__ == "__main__":
    test()