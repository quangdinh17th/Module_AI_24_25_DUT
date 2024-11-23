import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Hàm tính khoảng cách Euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Hàm KNN tự định nghĩa
def knn(X_train, y_train, X_test, k=5):
    y_pred = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        y_pred.append(most_common)
    return np.array(y_pred)


# Bước 1: Đọc file CSV chứa các đặc trưng Hu và nhãn

input_csv = 'Program/python/final_exam/KNN_HU/hu_features.csv'  # Đường dẫn tới tệp CSV

df = pd.read_csv(input_csv)

# Bước 2: Chuẩn bị dữ liệu

features = df.iloc[:, 1:8]  # Các cột đặc trưng Hu

labels = df.iloc[:, 8]  # Cột nhãn Hu

# Chuyển các đặc trưng và nhãn thành mảng numpy
X = features.values
y = labels.values

# Bước 3: Chia dữ liệu theo phương pháp K-Fold với k=5
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Đánh giá mô hình KNN với k=5
y_true_k5 = []
y_pred_k5 = []

# Lưu các chỉ số cho từng fold
accuracies = []
precisions = []
recalls = []
f1_scores = []
confusion_matrices = []

print("Đánh giá mô hình KNN với k=5:")
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Huấn luyện và đánh giá mô hình KNN
    y_pred = knn(X_train, y_train, X_test, k=3)    #thay đổi k tại đây

    y_true_k5.extend(y_test)
    y_pred_k5.extend(y_pred)

    # Tính toán các chỉ số đánh giá
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Thêm các chỉ số vào danh sách
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    confusion_matrices.append(conf_matrix)

    # In ra nhãn thực tế và nhãn dự đoán cho mỗi ảnh trong fold
    print("\nSo sánh chi tiết dự đoán:")
    print("Ground Truth:  ", '   '.join(map(str, y_test)))
    print("Predicted:     ", '   '.join(map(str, y_pred)))
    print("\n" + "-" * 50 + "\n")

    # In ra các chỉ số và ma trận nhầm lẫn
    print(f"Fold {fold + 1}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Ma trận nhầm lẫn:")
    print(conf_matrix)

# Tính toán kết quả trung bình
mean_accuracy = np.mean(accuracies)
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1_score = np.mean(f1_scores)

# Tính toán ma trận nhầm lẫn trung bình
total_confusion_matrix = np.sum(confusion_matrices, axis=0)

# In kết quả trung bình
print("\nKết quả trung bình sau 5 fold:\n")
print("Ma trận nhầm lẫn trung bình:")
print(total_confusion_matrix)
# Vẽ biểu đồ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(total_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, square=True)

# Cài đặt nhãn cho biểu đồ
plt.xlabel('Nhãn dự đoán')
plt.ylabel('Nhãn thực tế')
plt.title('Ma trận nhầm lẫn sau 5 Fold')
plt.xticks(ticks=np.arange(5) + 0.5, labels=['1', '2', '3', '4', '5'])
plt.yticks(ticks=np.arange(5) + 0.5, labels=['1', '2', '3', '4', '5'])
plt.show()
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F1 Score: {mean_f1_score:.4f}")
1