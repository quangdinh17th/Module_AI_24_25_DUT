import numpy as np
from PIL import Image
import os
import csv
import pandas as pd  # Thêm thư viện pandas để dễ dàng xử lý CSV


# Function to calculate central moments
def central_moment(data, p, q, x_centroid, y_centroid):
    y, x = np.ogrid[:data.shape[0], :data.shape[1]]
    return np.sum(((x - x_centroid) ** p) * ((y - y_centroid) ** q) * data)


# Function to calculate central normalized moments
def central_normalized_moment(moments, p, q, m00):
    return moments[f'm_{p}{q}'] / (m00 ** (((p + q) / 2) + 1))


# Function to calculate Hu moments
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


# Function to process a single image and return log Hu moments
def process_image(image_path):
    im = Image.open(image_path).convert("L")
    threshold = 128
    im_bin = im.point(lambda p: 255 if p > threshold else 0)
    data = np.array(im_bin)

    m00 = np.sum(data == 255) * 255

    # Calculate image centroid
    height, width = data.shape
    x_centroid = np.sum(np.arange(width) * np.sum(data, axis=0)) / m00
    y_centroid = np.sum(np.arange(height) * np.sum(data, axis=1)) / m00

    # Calculate central moments up to order 3
    moments = {}
    for p in range(4):
        for q in range(4):
            if p + q <= 3:
                moments[f'm_{p}{q}'] = central_moment(data, p, q, x_centroid, y_centroid)

    # Calculate central normalized moments
    normalized_moments = {}
    for p in range(4):
        for q in range(4):
            if 0 < p + q <= 3:
                normalized_moments[f'M_{p}{q}'] = central_normalized_moment(moments, p, q, m00)

    # Calculate Hu's moments
    hu_moments = calculate_hu_moments(normalized_moments)

    # Calculate log of absolute values of Hu's moments
    log_hu_moments = {f"log|S{i}|": round(-np.sign(m) * np.log10(abs(m)),5) if m != 0 else 0 for i, m in
                      hu_moments.items()}
    return log_hu_moments


# Function to process a folder of images and save results to CSV with one-hot encoding
def process_images_in_folder_with_one_hot(input_folder, output_csv, num_classes=5, samples_per_class=50):
    results = []  # Tạo danh sách để lưu kết quả

    # Iterate through the folder and process each image
    for idx, filename in enumerate(sorted(os.listdir(input_folder), key=lambda x: int(x.split('(')[1].split(')')[0]))):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Filter for image files
            image_path = os.path.join(input_folder, filename)  # Full path to image
            log_hu_moments = process_image(image_path)  # Calculate log Hu moments

            # Generate integer label
            label = idx // samples_per_class  # Determine label based on index

            # Prepare row: image name, Hu moments, and integer label
            row = [filename] + [log_hu_moments[f"log|S{i}|"] for i in range(1, 8)] + [label]
            results.append(row)  # Add row to results

    # Write data to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Image'] + [f"log|S{i}|" for i in range(1, 8)] + ['Label']  # Column names
        writer.writerow(header)  # Write header to CSV
        writer.writerows(results)  # Write all rows to CSV


# Example usage
input_folder = "Images/Binary"  # Input folder with binary images
output_csv = "Output/dulieu_hu.csv"  # Output CSV file

process_images_in_folder_with_one_hot(input_folder, output_csv)  # Process folder and save results
