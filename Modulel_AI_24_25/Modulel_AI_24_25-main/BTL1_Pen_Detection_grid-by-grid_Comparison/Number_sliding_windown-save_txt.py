'''
Đầu vào là Ảnh test và train ở dạng RGB
           Ảnh train đã resize
'''

import numpy as np
from PIL import Image
import cv2


def load_binary_image(image_path, threshold=246):
    img = Image.open(image_path)

    # Chuyển đổi ảnh RGB sang ảnh grayscale
    img_gray = img.convert('L')

    # Áp dụng ngưỡng (threshold) để chuyển sang ảnh nhị phân
    binary_img = img_gray.point(lambda x: 255 if x > threshold else 0, '1')

    # Chuyển đổi ảnh nhị phân thành mảng numpy với các giá trị 0 và 1
    binary_array = np.array(binary_img)
    return binary_array


def save_subarray(binary_array, x, y, width, height):
    # Lấy mảng con tương ứng với vị trí và kích thước khung trượt
    return binary_array[y:y + height, x:x + width]


def sliding_window_and_save_arrays(image_path, window_width, window_height, step_size):
    # Load ảnh nhị phân từ ảnh RGB
    binary_image = load_binary_image(image_path)
    img_height, img_width = binary_image.shape

    # Danh sách lưu các mảng con của khung trượt
    subarrays = []

    # Đọc ảnh gốc để vẽ khung trượt
    img = cv2.imread(image_path)

    # Trượt khung hình chữ nhật trên ảnh
    for y in range(0, img_height - window_height + 1, step_size):
        for x in range(0, img_width - window_width + 1, step_size):
            # Lấy mảng con tương ứng với khung trượt
            subarray = save_subarray(binary_image, x, y, window_width, window_height)
            subarrays.append(subarray)

            # Vẽ khung chữ nhật lên ảnh
            img_with_rect = img.copy()
            cv2.rectangle(img_with_rect, (x, y), (x + window_width, y + window_height), (0, 255, 0), 2)

            # Hiển thị ảnh với khung trượt
            cv2.imshow('Sliding Window', img_with_rect)
            cv2.waitKey(100)  # Đợi 100ms để hiển thị quá trình trượt

    cv2.destroyAllWindows()  # Đóng cửa sổ hiển thị sau khi hoàn thành
    return subarrays


def save_arrays_to_txt(subarrays):
    # Lưu từng mảng con vào file txt
    for i, subarray in enumerate(subarrays):
        output_txt_path = f'subarray_{i + 1}.txt'
        # Chuyển đổi mảng thành chuỗi với 0 và 1 (thay True/False thành 0/1)
        binary_string = '\n'.join([''.join(map(str, row)) for row in subarray])
        binary_string = binary_string.replace('True', '1').replace('False', '0')  # Chuyển True/False thành 1/0
        with open(output_txt_path, 'w') as file:
            file.write(binary_string)
        print(f"Mảng {i + 1} đã được lưu vào file '{output_txt_path}'")


# Sử dụng hàm
image_path = r'E:\DUT 2020\Nam_5\2024-2025_1\Tri_Tue_Nhan_Tao\BTL1_Pen_Recognition\test\test_original.jpg'  # Thay đường dẫn đến ảnh của bạn
window_width = 50  # Chiều rộng khung hình chữ nhật
window_height = 540  # Chiều cao khung hình chữ nhật
step_size = 20  # Kích thước bước trượt (bạn có thể thay đổi giá trị này)

# Lấy các mảng nhị phân từ khung trượt
subarrays = sliding_window_and_save_arrays(image_path, window_width, window_height, step_size)

# Lưu các mảng vào file txt
save_arrays_to_txt(subarrays)
