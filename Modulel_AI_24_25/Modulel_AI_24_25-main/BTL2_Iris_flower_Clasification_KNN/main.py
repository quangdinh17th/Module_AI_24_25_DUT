'''
 chiều dài, chiều rộng của đài hoa
 chiều dài, chiều rộng của cánh hoa
 cột class

 Mau test
     1   5.1,3.5,1.4,0.2, Iris-setosa
     51  7.0,3.2,4.7,1.4, Iris-versicolor
     101 6.3,3.3,6.0,2.5, Iris-virginica
 Mau train
     iris-147.data
'''
import pandas as pd
import numpy as np
from collections import Counter

columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'Class']

data_iris = pd.read_csv("iris-147.data", header=None, names=columns)

data_iris = data_iris.copy()

k_value = int(input("Nhập giá trị K: "))
test_vector = np.array(list(map(float, input("Nhập vector test gồm 4 đặc trưng: ").split(','))))

def manhattan_distance(row):
    return np.sum(np.abs(test_vector - row.iloc[:4]))

def euclidean_distance(row):
    return round(np.sqrt(np.sum(np.square(test_vector - row.iloc[:4]))), 3)

# Tính khoảng cách Manhattan cho từng mẫu và thêm vào cột 'Manhattan_Distance'
data_iris['Manhattan_Distance'] = data_iris.apply(manhattan_distance, axis=1)

# Tính khoảng cách Euclidean cho từng mẫu và thêm vào cột 'Euclidean_Distance'
data_iris['Euclidean_Distance'] = data_iris.apply(euclidean_distance, axis=1)

# Sắp xếp theo khoảng cách Manhattan và lấy K hàng đầu tiên
nearest_manhattan = data_iris.nsmallest(k_value, 'Manhattan_Distance')

# Sắp xếp theo khoảng cách Euclidean và lấy K hàng đầu tiên
nearest_euclidean = data_iris.nsmallest(k_value, 'Euclidean_Distance')

print(f"\n{k_value} mẫu có khoảng cách Manhattan nhỏ nhất:")
print(nearest_manhattan[['Class', 'Manhattan_Distance']])

print(f"\n{k_value} mẫu có khoảng cách Euclidean nhỏ nhất:")
print(nearest_euclidean[['Class', 'Euclidean_Distance']])

# Xác định lớp chiếm ưu thế cho khoảng cách Manhattan
manhattan_classes = nearest_manhattan['Class'].tolist()
manhattan_most_common = Counter(manhattan_classes).most_common(1)[0]
print(f"\n(Mahattan distance) Vector test thuộc Class: {manhattan_most_common[0]} ({manhattan_most_common[1]} lần)")

# Xác định lớp chiếm ưu thế cho khoảng cách Euclidean
euclidean_classes = nearest_euclidean['Class'].tolist()
euclidean_most_common = Counter(euclidean_classes).most_common(1)[0]
print(f"(Euclidean distance) Vector test thuộc Class: {euclidean_most_common[0]} ({euclidean_most_common[1]} lần)")

# 1.0,4.2,5.7,1.9
