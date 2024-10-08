import numpy as np

I = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0]])

x_size, y_size = I.shape
y, x = np.indices(I.shape) + 1

# Tính m00 (tổng cường độ ảnh)
m00 = np.sum(I)
m10 = np.sum(x * I)
m01 = np.sum(y * I)

# Tính trọng tâm x_bar và y_bar
x_bar = m10 / m00
y_bar = m01 / m00

# Tính m02 và m20
m20 = np.sum(((x - x_bar) ** 2) * I)
m02 = np.sum(((y - y_bar) ** 2) * I)

# Tính các moment chuẩn hóa M02 và M20
M20 = m20 / (m00 ** (1 + (2 / 2)))
M02 = m02 / (m00 ** (1 + (2 / 2)))

# Đặc trưng S1
S1 = M20 + M02

# Hiển thị kết quả
print(f'm00 = {m00:.2f}')
print(f'x_bar = {x_bar:.2f}, y_bar = {y_bar:.2f}')
print(f'm20 = {m20:.2f}, m02 = {m02:.2f}')
print(f'M20 = {M20:.2f}, M02 = {M02:.2f}')
print(f'S1 = {S1:.2f}')

