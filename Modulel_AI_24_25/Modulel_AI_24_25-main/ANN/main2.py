import numpy as np
import csv
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import accuracy_score
import pandas as pd

# Hàm đọc dữ liệu từ CSV
def load_data_from_csv(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Bỏ qua dòng tiêu đề
        for row in reader:
            try:
                data.append(list(map(float, row[1:8])))  # Đọc đặc trưng từ các cột 2 đến 8
                labels.append(int(row[8]))  # Đọc nhãn từ cột thứ 9
            except ValueError:
                print("Lỗi khi chuyển đổi giá trị sang float ở dòng:", row)
                continue
    data = np.array(data)
    labels = np.eye(5)[labels]  # Chuyển nhãn thành dạng one-hot encoding (5 lớp)
    return data, labels


# Hàm kích hoạt sigmoid và softmax
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# Khởi tạo các trọng số và độ chệch
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.random.uniform(-0.5, 0.5, (input_size, hidden_size))
    b1 = np.random.uniform(-0.5, 0.5, (1, hidden_size))
    W2 = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))
    b2 = np.random.uniform(-0.5, 0.5, (1, output_size))
    return W1, b1, W2, b2


# Lan truyền tiến
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# Lan truyền ngược và cập nhật trọng số
def backward(X, Y, Z1, A1, A2, W1, b1, W2, b2, eta):
    m = X.shape[0]
    dZ2 = A2 - Y  # Tính toán lỗi đầu ra
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Cập nhật trọng số
    W1 -= eta * dW1
    b1 -= eta * db1
    W2 -= eta * dW2
    b2 -= eta * db2
    return W1, b1, W2, b2


# Hàm tính lỗi (MSE)
def calculate_error(Y, A2):
    return np.mean((Y - A2) ** 2)


# Hàm dự đoán nhãn của một mẫu mới
def predict(X_new, W1, b1, W2, b2):
    _, _, _, A2 = forward(X_new, W1, b1, W2, b2)
    predicted_class = np.argmax(A2, axis=1)  # Lấy nhãn có xác suất cao nhất
    return predicted_class


# Hàm nhập dữ liệu từ người dùng
def input_features():
    print("Nhập 7 đặc trưng :")
    try:
        # Nhập một dòng và tách thành danh sách các số thực
        features = list(map(float, input().strip().split(',')))
        if len(features) != 7:
            raise ValueError("Vui lòng nhập đúng 7 giá trị.")
        return np.array([features])  # Định dạng thành mảng 2 chiều để đưa vào mô hình
    except ValueError as e:
        print(f"Lỗi: {e}")
        return input_features()  # Thử nhập lại nếu có lỗi


# Hàm tính độ chính xác
def accuracy(predictions, labels):
    preds = np.argmax(predictions, axis=1)
    actual = np.argmax(labels, axis=1)
    return np.mean(preds == actual)


# Huấn luyện ANN với backpropagation và điều kiện dừng sớm
def train(X, Y, input_size, hidden_size, output_size, eta, epochs, patience=100):
    W1, b1, W2, b2 = initialize_weights(input_size, hidden_size, output_size)
    best_weights = (W1, b1, W2, b2)
    best_accuracy = 0
    patience_counter = 0

    # Lưu trữ độ chính xác và lỗi
    accuracy_list = []
    error_list = []

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)

        # Tính toán lỗi
        error = calculate_error(Y, A2)

        W1, b1, W2, b2 = backward(X, Y, Z1, A1, A2, W1, b1, W2, b2, eta)

        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            acc = accuracy(A2, Y)
            accuracy_list.append(acc)
            error_list.append(error)

            # Kiểm tra xem mô hình có tốt hơn không
            if acc > best_accuracy:
                best_accuracy = acc
                best_weights = (W1, b1, W2, b2)
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            # Nếu không cải thiện trong một số epoch, dừng huấn luyện
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    return best_weights  # Trả về trọng số tốt nhất



# Giả sử bạn đang đọc dữ liệu từ một file CSV
data = pd.read_csv('Output/dulieu_hu.csv')  # Thay 'your_file.csv' bằng tên file của bạn


# Đường dẫn đến file CSV
file_path = 'Output/dulieu_hu.csv'  # Thay bằng đường dẫn thực tế

# Đọc dữ liệu từ file CSV
X, Y = load_data_from_csv(file_path)
input_size = X.shape[1]
output_size = 5

# Huấn luyện mô hình với lớp ẩn có 10 neuron
W1_10, b1_10, W2_10, b2_10 = train(X, Y, input_size, 10, output_size, eta=0.1, epochs=5000)

# Huấn luyện mô hình với lớp ẩn có 15 neuron
W1_15, b1_15, W2_15, b2_15 = train(X, Y, input_size, 15, output_size, eta=0.1, epochs=5000)

X_new = input_features()  # Nhập 7 đặc trưng từ người dùng

# Thực hiện dự đoán với một mẫu dữ liệu mới
print("\nDự đoán nhãn cho mẫu mới bằng mô hình với 10 neuron ở lớp ẩn:")
predicted_label_10 = predict(X_new, W1_10, b1_10, W2_10, b2_10)
print(f"Dự đoán của mô hình với 10 neuron: Lớp {predicted_label_10[0]}")

print("\nDự đoán nhãn cho mẫu mới bằng mô hình với 15 neuron ở lớp ẩn:")
predicted_label_15 = predict(X_new, W1_15, b1_15, W2_15, b2_15)
print(f"Dự đoán của mô hình với 15 neuron: Lớp {predicted_label_15[0]}")


def k_fold_cross_validation(X, Y, input_size, hidden_size, output_size, eta=0.1, epochs=5000):
    # Khởi tạo KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 0
    cm = np.zeros((5, 5))  # Ma trận nhầm lẫn cho 5 lớp
    accuracy_tb = 0
    precision_tb = np.zeros(5)
    recall_tb = np.zeros(5)
    f1_tb = np.zeros(5)

    # Vòng lặp qua từng fold
    for train_index, test_index in kf.split(X):
        fold += 1
        # Chia tập dữ liệu theo chỉ số được tạo bởi KFold
        X_train_k, X_test_k = X[train_index], X[test_index]
        Y_train_k, Y_test_k = Y[train_index], Y[test_index]

        # Huấn luyện mô hình trên tập huấn luyện
        W1, b1, W2, b2 = train(X_train_k, Y_train_k, input_size, hidden_size, output_size, eta=eta, epochs=epochs)

        # Dự đoán trên tập kiểm thử
        _, _, _, A2 = forward(X_test_k, W1, b1, W2, b2)  # Lan truyền tiến
        Y_pred = np.argmax(A2, axis=1)
        Y_true = np.argmax(Y_test_k, axis=1)

        # Hiển thị ma trận nhầm lẫn cho từng fold
        cm_fold = confusion_matrix(Y_true, Y_pred)
        cm += cm_fold

        # Tính toán các chỉ số cho từng fold
        accuracy = accuracy_score(Y_true, Y_pred)
        precision = precision_score(Y_true, Y_pred, average=None, zero_division=0)
        recall = recall_score(Y_true, Y_pred, average=None, zero_division=0)
        f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)

        # Tính tổng các chỉ số cho từng fold
        accuracy_tb += accuracy
        precision_tb += precision
        recall_tb += recall
        f1_tb += f1

    # Hiển thị ma trận nhầm lẫn cho toàn bộ dữ liệu
    print("Confusion matrix for all folds:")
    print(cm)

    # Tính các chỉ số trung bình
    accuracy_tb /= fold
    precision_tb /= fold
    recall_tb /= fold
    f1_tb /= fold

    print(f"Accuracy average: {accuracy_tb:.4f}")
    print(f"Precision average: {precision_tb}")
    print(f"Recall average: {recall_tb}")
    print(f"F1 score average: {f1_tb}")


# Đánh giá mô hình với 5-fold cross-validation cho lớp ẩn có 10 neuron
print("\nEvaluating model with 10 neurons in hidden layer using 5-fold cross-validation:")
average_accuracy_10 = k_fold_cross_valida3.17887,7.93251,11.09523,13.84563,-26.64358,-17.8416,-26.37035tion(X, Y, input_size, 10, output_size, eta=0.1, epochs=5000)

# Đánh giá mô hình với 5-fold cross-validation cho lớp ẩn có 15 neuron
print("\nEvaluating model with 15 neurons in hidden layer using 5-fold cross-validation:")
average_accuracy_15 = k_fold_cross_validation(X, Y, input_size, 15, output_size, eta=0.1, epochs=5000)
