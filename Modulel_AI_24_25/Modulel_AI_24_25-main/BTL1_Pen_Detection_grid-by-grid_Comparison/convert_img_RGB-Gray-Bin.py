'''
bit 0 chỉ màu đen và bit 1 chỉ màu trắng
0 chỉ màu đen, và 255 chỉ màu trắng, và 127 chỉ màu xám
'''

import os
from PIL import Image

def convert_images(input_directory, output_directory, threshold=225):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Lặp qua tất cả các tệp trong thư mục đầu vào
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_directory, filename)

            with Image.open(input_path) as img:
                # Chuyển đổi sang ảnh xám
                gray_img = img.convert('L')

                # Chuyển đổi sang ảnh nhị phân
                binary_img = gray_img.point(lambda x: 255 if x < threshold else 0, 'L')

                output_filename = f"binary_{os.path.splitext(filename)[0]}.png"
                output_path = os.path.join(output_directory, output_filename)

                binary_img.save(output_path)
            print(f"Đã chuyển đổi: {filename}")

# Sử dụng hàm
input_dir = r"C:\Users\LENOVO\Pictures\Lenovo"
output_dir = r"C:\Users\LENOVO\Pictures\Lenovo"
convert_images(input_dir, output_dir)