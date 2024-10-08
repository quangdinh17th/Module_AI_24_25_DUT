### 1. Có 3 ảnh gồm: 2 ảnh train (positive object and negative object) và 1 ảnh test đều nền trơn
### 2. Xác định kích thước của vật thể trong ảnh cần test, ở đây là 50x540 pixels
### 3. Ảnh train có vật thể được cắt sát vào vật thể (Ex: data/raw/train2.jpg)
### 4. Sử dụng "resize_img_train.py" để thay đổi kích thước 2 ảnh train về 50x540
### 5. Chuyển đổi cả 3 ảnh về dạng ảnh nhị phân (Binary) sử dụng "convert_img_RGB-Gray-Bin.py"
### 8. Tiến hành nhận dạng vật thể cây bút trong ảnh test sử dụng "main.py", với ảnh đầu vào là RGB hoặc Binary
### 7. So sánh 2 ảnh nhị phân khác nhau bằng cách sô sánh từng điểm ảnh thông qua mảng
### 8. Kết quả sẽ khoanh vùng vật thể cây bút trên ảnh test, cho biết tại khung cửa sổ trượt thứ bao nhiêu và tính mismatch so với ảnh train (pos)

#### "Number_sliding_windown-save_txt/py" xem bức ảnh test có bao nhiêu khung trượt và các ảnh nhị phân trong khung trượt được lưu ở dạng mảng nhị phân thông qua file .txt
