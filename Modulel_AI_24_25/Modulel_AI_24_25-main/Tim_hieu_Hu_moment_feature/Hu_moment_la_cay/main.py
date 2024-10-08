import numpy as np
from PIL import Image

# Load image and convert to grayscale
filename = "Pic_binary/binary_la.jpg"

im = Image.open(filename).convert("L")

# Convert to binary image
threshold = 128
im_bin = im.point(lambda p: 255 if p > threshold else 0)

# Convert to NumPy array
data = np.array(im_bin)

# Calculate m00 (total number of white pixels, i.e., pixels with value 255)
m00 = np.sum(data == 255) * 255
print(f"total number of white pixels: {m00}")

# Image dimensions
height, width = data.shape

# Calculate centroid coordinates
x_centroid = np.sum(np.arange(width) * np.sum(data, axis=0)) / m00
y_centroid = np.sum(np.arange(height) * np.sum(data, axis=1)) / m00

print(f'Centroid Coordinates: x = {x_centroid}, y = {y_centroid}')


def central_moment(data, p, q, x_centroid, y_centroid):
    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    return np.sum(((x - x_centroid) ** p) * ((y - y_centroid) ** q) * data)


# Tính toán các mômen trung tâm lên đến bậc 3
moments = {}
for p in range(4):
    for q in range(4):
        if p + q <= 3:
            moments[f'm_{p}{q}'] = central_moment(data, p, q, x_centroid, y_centroid)

# In kết quả moment trung tâm
print("\nCentral Moments:")
for key, value in moments.items():
    print(f"{key} = {value}")


# Calculate central normalized moments

def central_normalized_moment(moments, p, q):
    return moments[f'm_{p}{q}'] / (m00 ** (((p + q) / 2) + 1))


# Tính toán các mômen chuẩn hóa trung tâm lên đến bậc 3
normalized_moments = {}
for p in range(4):
    for q in range(4):
        if 0 < p + q <= 3:  # Bỏ qua M_00 vì nó luôn bằng 1
            normalized_moments[f'M_{p}{q}'] = central_normalized_moment(moments, p, q)

# In kết quả mômen chuẩn hóa trung tâm
print("\nCentral Normalized Moments:")
for key, value in normalized_moments.items():
    print(f"{key} = {value}")


def calculate_hu_moments(M):
    S = {}
    S[1] = M['M_20'] + M['M_02']
    S[2] = (M['M_20'] - M['M_02']) ** 2 + 4 * (M['M_11'] ** 2)
    S[3] = (M['M_30'] - 3 * M['M_12']) ** 2 + (3 * M['M_21'] - M['M_03']) ** 2
    S[4] = (M['M_30'] + M['M_12']) ** 2 + (M['M_03'] + M['M_21']) ** 2
    S[5] = (M['M_30'] - 3 * M['M_12']) * (M['M_30'] + M['M_12']) * (
            (M['M_30'] + M['M_12']) ** 2 - 3 * (M['M_03'] + M['M_21']) ** 2) + \
           (3 * M['M_21'] - M['M_03']) * (M['M_03'] + M['M_21']) * (
                   3 * (M['M_30'] + M['M_12']) ** 2 - (M['M_03'] + M['M_21']) ** 2)
    S[6] = (M['M_20'] - M['M_02']) * ((M['M_30'] + M['M_12']) ** 2 - (M['M_03'] + M['M_21']) ** 2) + \
           4 * M['M_11'] * (M['M_30'] + M['M_12']) * (M['M_03'] + M['M_21'])
    S[7] = (3 * M['M_21'] - M['M_03']) * (M['M_30'] + M['M_12']) * (
            (M['M_30'] + M['M_12']) ** 2 - 3 * (M['M_03'] + M['M_21']) ** 2) - \
           (M['M_30'] - 3 * M['M_12']) * (M['M_03'] + M['M_21']) * (
                   3 * (M['M_30'] + M['M_12']) ** 2 - (M['M_03'] + M['M_21']) ** 2)
    return S


hu_moments = calculate_hu_moments(normalized_moments)

# Print Hu's invariant moments
print("\nHu's Invariant Moments:")
for i, moment in hu_moments.items():
    print(f"S{i} = {moment}")

# Optionally, calculate log of absolute values for better numerical stability
log_hu_moments = {f"log|S{i}|": -np.sign(m) * np.log10(abs(m)) for i, m in hu_moments.items()}
print("\nLogarithm of absolute values of Hu's Moments:")
for key, value in log_hu_moments.items():
    print(f"{key} = {value}")
