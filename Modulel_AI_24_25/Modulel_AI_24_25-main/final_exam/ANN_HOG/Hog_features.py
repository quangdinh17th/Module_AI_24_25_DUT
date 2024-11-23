import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog

input_folder = 'Program/python/final_exam/leaf_data/Data_Leaf/Goc'
resized_folder = 'Program/python/final_exam/ANN_HOG/resized_image'  # Thư mục để lưu ảnh đã resize

# Tạo thư mục nếu chưa tồn tại
os.makedirs(resized_folder, exist_ok=True)

# Đường dẫn để lưu tệp CSV đầu ra
output_csv = 'Program/python/final_exam/ANN_HOG/hog_features.csv'

# Danh sách để lưu các HOG features
hog_features_list = []

# Nhãn cho mỗi nhóm 50 ảnh
labels = ['1', '2', '3', '4', '5']

def resize_image(image, target_size):
    # Lấy kích thước gốc
    original_size = image.size
    target_width, target_height = target_size

    # Tính toán tỷ lệ
    ratio = min(target_width / original_size[0], target_height / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))

    # Resize ảnh
    resized_image = image.resize(new_size)  # Loại bỏ Image.ANTIALIAS

    # Tạo một ảnh trắng (hoặc đen) với kích thước mong muốn
    new_image = Image.new("L", (target_width, target_height))

    # Tính toán vị trí để đặt ảnh đã resize vào giữa ảnh mới
    paste_x = (target_width - new_size[0]) // 2
    paste_y = (target_height - new_size[1]) // 2

    # Dán ảnh đã resize vào ảnh mới
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image

# Duyệt qua tất cả các tệp trong thư mục
for idx, filename in enumerate(os.listdir(input_folder), start=1):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            # Đọc ảnh nhị phân bằng PIL
            img_path = os.path.join(input_folder, filename)
            image = Image.open(img_path).convert('L')  # Chuyển ảnh thành ảnh xám (grayscale)

            # Resize ảnh về kích thước cố định để tính HOG
            resized_image = resize_image(image, (128, 128))  # Resize về 128x128
            resized_image_np = np.array(resized_image)

            # Lưu ảnh đã resize vào thư mục
            resized_image.save(os.path.join(resized_folder, filename))

            # Tính toán các đặc trưng HOG
            hog_features = hog(resized_image_np, orientations=8, pixels_per_cell=(16, 16),
                               cells_per_block=(1, 1), visualize=False)

            # Chuyển thành mảng 1D để lưu trữ
            hog_features = hog_features.flatten()

            # Xác định nhãn dựa trên chỉ số của ảnh
            label = labels[(idx - 1) // 50]  # Chia nhóm mỗi 50 ảnh

            # Lưu kết quả vào danh sách
            hog_features_list.append({
                'HOG': list(hog_features),  # Lưu toàn bộ đặc trưng HOG dưới dạng danh sách
                'Tên Ảnh': filename,  # Thêm tên ảnh vào
                'Labels': label  # Thêm nhãn vào
            })

        except Exception as e:
            print(f"Lỗi xảy ra khi xử lý ảnh {img_path}: {str(e)}")

# Chuyển danh sách HOG features thành DataFrame nếu không rỗng
if hog_features_list:
    df = pd.DataFrame(hog_features_list)

    # Tách từng giá trị của HOG thành các cột riêng biệt
    hog_df = pd.DataFrame(df['HOG'].to_list(), columns=[f'HOG_{i + 1}' for i in range(len(df['HOG'][0]))])

    # Kết hợp các cột tên ảnh và HOG trước
    temp_df = pd.concat([df[['Tên Ảnh']], hog_df], axis=1)

    # Thêm cột nhãn ở cuối
    final_df = pd.concat([temp_df, df[['Labels']]], axis=1)

    # Lưu DataFrame vào tệp CSV
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"Kết quả đã được lưu vào tệp CSV: {output_csv}")
else:
    print("Không có đặc trưng HOG nào được tính toán.")