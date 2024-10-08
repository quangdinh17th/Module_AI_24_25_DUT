'''
bit 0 chỉ màu đen và bit 1 chỉ màu trắng
0 chỉ màu đen, và 255 chỉ màu trắng, và 127 chỉ màu xám
'''

import os
from PIL import Image

def process_images(input_folder, output_folder, threshold=225):
    files = os.listdir(input_folder)

    image_files = [f for f in files if f.lower()]

    for image_file in image_files:
        # Đường dẫn đầy đủ của file ảnh đầu vào
        input_path = os.path.join(input_folder, image_file)

        with Image.open(input_path) as img:
            # Chuyển sang ảnh xám
            gray_img = img.convert('L')

            # Tạo ảnh nhị phân (trắng trên đen)
            binary_img = gray_img.point(lambda x: 255 if x < threshold else 0, 'L')

            # Tạo tên file đầu ra cho ảnh xám
            gray_filename = f"gray_{image_file}"
            gray_path = os.path.join(output_folder, gray_filename)
            gray_img.save(gray_path)

            # Tạo tên file đầu ra cho ảnh nhị phân
            binary_filename = f"binary_{image_file}"
            binary_path = os.path.join(output_folder, binary_filename)
            binary_img.save(binary_path)
        print(f"Đã xử lý: {image_file}")

input_folder = "raw"
output_folder = "Pic_binary"
process_images(input_folder, output_folder, threshold=246)
