'''
Điều chỉnh kích thước ảnh train và lưu ảnh vào thư mục đầu ra
Đường dẫn ảnh đầu vào và thư mục đầu ra
'''

import os
from PIL import Image

def resize_img_train(file_path, output_folder, size):
    img = Image.open(file_path)
    img = img.resize(size, Image.LANCZOS)

    # Tạo tên file đầu ra và lưu vào thư mục output
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    img.save(output_path)

    print(f"Đã chuẩn hóa và lưu vào: {output_path}")

image_file = r"C:\Users\LENOVO\Pictures\Lenovo\binary_cam1.png"
output_folder = r'data'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

resize_img_train(image_file, output_folder, size=(50, 540))
