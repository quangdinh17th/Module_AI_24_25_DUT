import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def initialize_parameters(input_size, hidden_size, output_size):
    # Khởi tạo trọng số ngẫu nhiên
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    # Net function là tuyến tính, và activation là sigmoid cho hidden layer
    z1 = np.dot(X, W1) + b1  # net function tuyến tính
    a1 = sigmoid(z1)  # hàm kích hoạt sigmoid

    # Net function và activation cho output layer
    z2 = np.dot(a1, W2) + b2  # net function tuyến tính
    a2 = sigmoid(z2)  # hàm kích hoạt sigmoid

    return z1, a1, z2, a2

def backward_propagation(X, y, a1, a2, W2):
    m = X.shape[0]
    # Tính đạo hàm
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def train_model(X, y, hidden_size=10, learning_rate=0.1, epochs=1000):
    input_size = X.shape[1]
    output_size = y.shape[1]

    # Khởi tạo tham số
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    # Training
    for epoch in range(epochs):
        # Forward propagation
        z1, a1, z2, a2 = forward_propagation(X, W1, b1, W2, b2)

        # Backward propagation
        dW1, db1, dW2, db2 = backward_propagation(X, y, a1, a2, W2)

        # Cập nhật tham số
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if (epoch + 1) % 100 == 0:
            loss = np.mean(np.square(y - a2))
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

    return W1, b1, W2, b2


def plot_confusion_matrix(conf_matrix, fold_num):
    plt.figure(figsize=(8, 6))

    # Tạo labels cho trục bắt đầu từ 1
    num_classes = conf_matrix.shape[0]
    labels = list(range(1, num_classes + 1))

    # Vẽ heatmap với labels mới
    sns.heatmap(conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)

    plt.title(f'Ma trận nhầm lẫn - Fold {fold_num}')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    plt.show()

def predict(X, W1, b1, W2, b2):
    # Forward propagation
    _, _, _, a2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(a2, axis=1)

def predict_and_evaluate(X_test, y_test, W1, b1, W2, b2, le):
    # Get predictions
    y_pred = predict(X_test, W1, b1, W2, b2)

    # Convert one-hot encoded test labels back to original labels
    y_test_orig = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test

    # Convert numeric labels back to original class labels
    y_test_labels = le.inverse_transform(y_test_orig)
    y_pred_labels = le.inverse_transform(y_pred)

    # Print detailed comparison horizontally
    print("\nSo sánh chi tiết dự đoán:")
    print("Ground Truth: ", end="")
    for true in y_test_labels:
        print(f"{true:3}", end=" ")
    print("\nPredicted:    ", end="")
    for pred in y_pred_labels:
        print(f"{pred:3}", end=" ")
    print("\n")

    # Print summary for each class
    counts = Counter(zip(y_test_labels, y_pred_labels))
    # for (true, pred), count in sorted(counts.items()):
    #     print(f"Ground Truth: {true:2}, Predicted: {pred:2}, Count: {count:3}")

    # Calculate metrics
    conf_matrix = confusion_matrix(y_test_orig, y_pred)
    accuracy = accuracy_score(y_test_orig, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_orig, y_pred, average='weighted')

    return conf_matrix, accuracy, precision, recall, f1

# Đọc và chuẩn bị dữ liệu
data = pd.read_csv("Program/python/final_exam/KNN_HU/hu_features.csv")
X = data.iloc[:, 1:8].values  # 7 Hu moments
y = data['Labels'].values

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Chuyển đổi labels thành one-hot encoding
le = LabelEncoder()
y = le.fit_transform(y)
y_onehot = np.eye(len(np.unique(y)))[y]

# Thiết lập k-fold cross validation
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf = KFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = []
total_conf_matrix = None
fold_num = 1

# Main training loop with k-fold cross validation
print("Bắt đầu huấn luyện và đánh giá mô hình...\n")
for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    print(f"{'=' * 50}")
    print(f"Fold {fold}")
    print(f"{'=' * 50}")

    # Split data for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_onehot[train_index], y_onehot[test_index]

    # Train model
    W1, b1, W2, b2 = train_model(X_train, y_train,
                                hidden_size=15,
                                learning_rate=0.05,
                                epochs=2000)

    # Evaluate and show detailed predictions
    conf_matrix, accuracy, precision, recall, f1 = predict_and_evaluate(
        X_test, y_test, W1, b1, W2, b2, le)

    if total_conf_matrix is None:
        total_conf_matrix = conf_matrix
    else:
        total_conf_matrix += conf_matrix

    # Save metrics
    all_metrics.append({
        'Fold': fold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    })

    # Print fold results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    plot_confusion_matrix(conf_matrix, fold_num)
    fold_num += 1

print(f"{'=' * 50}")
# Hiển thị confusion matrix gộp
print("Confusion Matrix after 5 fold:")
print(total_conf_matrix)

plt.figure(figsize=(8, 6))
# Tạo labels bắt đầu từ 1
num_classes = total_conf_matrix.shape[0]
labels = list(range(1, num_classes + 1))

# Thêm labels vào heatmap
sns.heatmap(total_conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,  # Thêm labels cho trục x
            yticklabels=labels)  # Thêm labels cho trục y

plt.title("Ma trận nhầm lẫn sau 5 Fold")
plt.ylabel('Nhãn thực tế')
plt.xlabel('Nhãn dự đoán')
plt.show()

# Tính toán và in metrics trung bình
metrics_df = pd.DataFrame(all_metrics)
print(f"\n{metrics_df.mean()[1:].round(4)}")
1