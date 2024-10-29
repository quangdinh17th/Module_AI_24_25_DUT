import os
from PIL import Image

def convert_images(input_directory, binary_output_directory, gray_output_directory, threshold=146):
    # Tạo thư mục đầu ra cho ảnh nhị phân nếu chưa tồn tại
    if not os.path.exists(binary_output_directory):
        os.makedirs(binary_output_directory)

    # Tạo thư mục đầu ra cho ảnh xám nếu chưa tồn tại
    if not os.path.exists(gray_output_directory):
        os.makedirs(gray_output_directory)

    # Lặp qua tất cả các tệp trong thư mục đầu vào
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_directory, filename)

            with Image.open(input_path) as img:
                # Chuyển đổi sang ảnh xám
                gray_img = img.convert('L')

                # Chuyển đổi sang ảnh nhị phân
                binary_img = gray_img.point(lambda x: 255 if x < threshold else 0, 'L')

                # Lưu ảnh nhị phân
                binary_output_filename = f"{os.path.splitext(filename)[0]}_binary.jpg"
                binary_output_path = os.path.join(binary_output_directory, binary_output_filename)
                binary_img.save(binary_output_path)

                # Lưu ảnh xám
                gray_output_filename = f"{os.path.splitext(filename)[0]}_gray.jpg"
                gray_output_path = os.path.join(gray_output_directory, gray_output_filename)
                gray_img.save(gray_output_path)

            print(f"Đã chuyển đổi: {filename}")

# Sử dụng hàm
input_dir = "Images/Original/Goc"
binary_output_dir = "Images/Binary"
gray_output_dir = "Images/Gray"
convert_images(input_dir, binary_output_dir, gray_output_dir)
