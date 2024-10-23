import streamlit as st
import cv2
import numpy as np
from skimage import measure
from sklearn.cluster import KMeans

# Define the image processing function with K-means
def process_image(image):
    # Otsu's Thresholding
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Region Growing
    def region_growing(img, seed):
        rows, cols = img.shape
        region = np.zeros_like(img)
        seed_value = img[seed]
        stack = [seed]
        region[seed] = 255
        visited = set(stack)

        while stack:
            x, y = stack.pop()
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (i == 0 and j == 0) or (x + i < 0 or x + i >= rows) or (y + j < 0 or y + j >= cols):
                        continue

                    if (x + i, y + j) not in visited and abs(int(img[x + i, y + j]) - int(seed_value)) < 20:
                        region[x + i, y + j] = 255
                        stack.append((x + i, y + j))
                        visited.add((x + i, y + j))

        return region

    # Seed point for region growing
    seed_point = (100, 100)  # Adjust as necessary
    region_grown_image = region_growing(image, seed_point)

    # Region Splitting
    def split(image, threshold):
        h, w = image.shape
        if h <= 1 or w <= 1 or np.std(image) < threshold:
            return np.full((h, w), np.mean(image))
        
        top_left = split(image[:h//2, :w//2], threshold)
        top_right = split(image[:h//2, w//2:], threshold)
        bottom_left = split(image[h//2:, :w//2], threshold)
        bottom_right = split(image[h//2:, w//2:], threshold)

        top_combined = np.hstack((top_left, top_right))
        bottom_combined = np.hstack((bottom_left, bottom_right))

        return np.vstack((top_combined, bottom_combined))

    split_image = split(image, 10)

    # Region Merging
    def merge(image, labels, threshold):
        regions = measure.regionprops(labels)
        for region in regions:
            coords = region.coords
            region_mean = np.mean(image[coords[:, 0], coords[:, 1]])
            if np.std(image[coords[:, 0], coords[:, 1]]) < threshold:
                image[coords[:, 0], coords[:, 1]] = region_mean
        return image

    labels = measure.label(thresholded_image)
    merged_image = merge(image.copy(), labels, 10)

    # Region Splitting and Merging
    def split_and_merge(image, threshold):
        split_img = split(image, threshold)
        labels = measure.label(split_img)
        return merge(split_img, labels, threshold)

    split_merge_image = split_and_merge(image, 10)

    # K-means Clustering
    def kmeans_clustering(image, k=3):
        # Reshape the image into a 2D array
        pixels = image.reshape((-1, 1)).astype(np.float32)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)
        labels = kmeans.labels_.reshape(image.shape)

        # Create the segmented image using the cluster centers
        segmented_image = kmeans.cluster_centers_[labels]
        return segmented_image

    kmeans_image = kmeans_clustering(image, k=3)

    return {
        "Original": image,
        "Thresholded (Otsu's)": thresholded_image,  # Label added here
        "Region Grown": region_grown_image,
        "Region Split": split_image,
        "Region Merged": merged_image,
        "Split and Merged": split_merge_image,
        "K-means Clustered": kmeans_image
    }

# Normalize image for display
def normalize_image(img):
    img = img.astype(np.float32)
    img -= img.min()  # Shift to 0
    if img.max() > 0:
        img /= img.max()  # Scale to [0, 1]
    return (img * 255).astype(np.uint8)  # Convert to 8-bit

# Streamlit application
st.title("Image Processing with Segmentation Techniques")
st.write("Created by Pragya and Jayaditya")  # Added line

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Process the image
    processed_images = process_image(gray_image)

    # Display the processed images
    for title, img in processed_images.items():
        st.subheader(title)
        st.image(normalize_image(img), channels="GRAY" if title != "Original" else "RGB")
