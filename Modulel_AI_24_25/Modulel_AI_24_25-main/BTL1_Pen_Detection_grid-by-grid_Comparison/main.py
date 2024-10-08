'''
Đầu vào là Ảnh test và train ở dạng RGB
           Ảnh train đã resize
'''

import numpy as np
from PIL import Image
import cv2

def load_binary_image(image_path, threshold=246):
    img = Image.open(image_path)

    # Chuyển đổi sang ảnh grayscale
    img = img.convert('L')

    # Áp dụng ngưỡng (threshold) để chuyển sang ảnh nhị phân
    binary_img = img.point(lambda x: 255 if x > threshold else 0, 'L')

    # Chuyển đổi ảnh nhị phân thành mảng numpy
    binary_array = np.array(binary_img)
    return binary_array


# Lưu giá trị mảng của ảnh trong cửa sổ trượt, vị trí và kích thước khung trượt
def save_subarray(binary_array, x, y, width, height):
    return binary_array[y:y + height, x:x + width]

# Chạy cửa sổ trượt và Lưu giá trị mảng
def sliding_window_and_save_arrays(image_path, window_width, window_height, step_size):
    binary_image = load_binary_image(image_path)
    img_height, img_width = binary_image.shape

    # Danh sách lưu các mảng con của cửa sổ trượt
    subarrays = []

    # Trượt cửa sổ hình chữ nhật trên ảnh
    for y in range(0, img_height - window_height + 1, step_size):
        for x in range(0, img_width - window_width + 1, step_size):
            # Lấy mảng con tương ứng với cửa sổ trượt
            subarray = save_subarray(binary_image, x, y, window_width, window_height)
            subarrays.append((subarray, x, y))          # vị trí của khung trượt
    return subarrays

# So sánh ảnh train và ảnh của mỗi cửa sổ trượt bằng cách so sánh mỗi phần tử trong mảng
def compare_arrays(arr1, arr2):
    difference = np.sum(arr1 != arr2)
    return difference

# Vẽ hình chữ nhật trên ảnh
def draw_rectangle_and_show(image_path, x, y, width, height):
    img = cv2.imread(image_path)
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)  # Màu xanh lá cây, độ dày 2
    cv2.imshow('Image with Rectangle', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Sử dụng hàm
test_image_path = 'data/test/test1.jpg'
train_image_path = 'data/train/train2.jpg'
window_width = 50
window_height = 540
step_size = 20  # Kích thước bước trượt

# Lấy các mảng nhị phân từ khung trượt của ảnh test
subarrays = sliding_window_and_save_arrays(test_image_path, window_width, window_height, step_size)

arr_train = load_binary_image(train_image_path)

# Tìm khung trượt có sự khác biệt nhỏ nhất
min_difference = float('inf')   # Khởi tạo giá trị khác biệt nhỏ nhất là vô cùng lớn
min_index = -1                  # Chỉ số của mảng với khác biệt nhỏ nhất
min_x, min_y = 0, 0             # Vị trí của khung trượt có sự khác biệt nhỏ nhất

for i, (subarray, x, y) in enumerate(subarrays, start=1):
    difference = compare_arrays(subarray, arr_train)
    if difference is not None:
        if difference < min_difference:
            min_difference = difference
            min_index = i
            min_x, min_y = x, y

# In số điểm ảnh khác biệt nhỏ nhất và vẽ hình chữ nhật quanh khung trượt
if min_index != -1:
    print(f"Số điểm ảnh khác biệt nhỏ nhất là {min_difference} tại khung ảnh thứ {min_index}.")
    print(f"Vị trí của khung trượt là: ({min_x}, {min_y})")
    draw_rectangle_and_show(test_image_path, min_x, min_y, window_width, window_height)
