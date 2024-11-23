import cv2
import os

# Đường dẫn đến thư mục chứa ảnh gốc và thư mục lưu ảnh xám
input_folder = 'Program/python/final_exam/data_250_leaf/data/Original'
output_folder = 'Program/python/final_exam/data_250_leaf/data/Gray'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua từng file trong thư mục
for filename in os.listdir(input_folder):
    # Kiểm tra nếu file là ảnh (ví dụ: .jpg, .png)
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith("jpeg"):
        # Đọc ảnh RGB
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Chuyển đổi sang ảnh xám (grayscale)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Lưu ảnh xám vào thư mục đích
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, gray_image)

print("Hoàn thành chuyển đổi tất cả ảnh trong thư mục sang ảnh xám.")
