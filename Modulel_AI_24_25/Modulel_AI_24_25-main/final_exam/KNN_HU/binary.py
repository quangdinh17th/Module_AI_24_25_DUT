import cv2
import os

# Đường dẫn đến thư mục chứa ảnh gốc và thư mục lưu ảnh nhị phân
input_folder = 'Program/python/final_exam/leaf_data_exam/Goc'
output_folder = 'Program/python/final_exam/leaf_data_exam/binary'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua từng file trong thư mục
for filename in os.listdir(input_folder):
    # Kiểm tra nếu file là ảnh (ví dụ: .jpg, .png)
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # Đọc ảnh RGB
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Chuyển ảnh RGB sang ảnh xám
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Áp dụng ngưỡng để chuyển sang ảnh nhị phân nền đen, lá cây trắng
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

        # Lưu ảnh nhị phân vào thư mục đích
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, binary_image)

print("Hoàn thành chuyển đổi tất cả ảnh trong thư mục sang ảnh nhị phân với nền đen và vật thể trắng.")
