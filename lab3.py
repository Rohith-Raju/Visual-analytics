import cv2
import numpy as np
import os
from scipy.spatial.distance import cdist
from glob import glob

def extract_features(image):
    image = cv2.resize(image, (256, 256))
    
    avg_color = np.mean(image, axis=(0, 1))  
    
    center = (image.shape[0] // 2, image.shape[1] // 2)
    center_patch = image[center[0]-10:center[0]+10, center[1]-10:center[1]+10]
    center_avg_color = np.mean(center_patch, axis=(0, 1))
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    luminance = np.mean(gray_image)
    
    edges = cv2.Canny(gray_image, 100, 200)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    hist_b = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
    hist_r = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
    color_histogram = np.concatenate([hist_b, hist_g, hist_r])
    
    return {
        "avg_color": avg_color,
        "center_avg_color": center_avg_color,
        "luminance": luminance,
        "edge_density": edge_density,
        "color_histogram": color_histogram
    }

def load_images(folder_path):
    image_paths = glob(os.path.join(folder_path, "*.jpg"))
    images = [cv2.imread(image_path) for image_path in image_paths]
    return images, image_paths

def compute_distance_matrix(feature_vectors):
    return cdist(feature_vectors, feature_vectors, metric="euclidean")

def rank_images(distance_matrix, chosen_index):
    distances = distance_matrix[chosen_index]
    ranked_indices = np.argsort(distances)
    return ranked_indices

def main():
    folder_path = "./Lab3.1"  
    images, image_paths = load_images(folder_path)
    
    all_features = [extract_features(image) for image in images]
    
    chosen_index = 2
    print(f"Chosen Image: {image_paths[chosen_index]}")
    
    feature_names = ["avg_color", "center_avg_color", "luminance", "edge_density", "color_histogram"]
    for feature_name in feature_names:
        print(f"\nRanking images based on feature: {feature_name}")
        
        feature_vectors = np.array([features[feature_name] for features in all_features])
        
        if len(feature_vectors.shape) == 1:
            feature_vectors = feature_vectors.reshape(-1, 1)
        
        distance_matrix = compute_distance_matrix(feature_vectors)
        
        ranked_indices = rank_images(distance_matrix, chosen_index)
        
        print("Ranked Images by Similarity:")
        for rank, index in enumerate(ranked_indices[1:], start=1):  
            print(f"Rank {rank}: {image_paths[index]} (Distance: {distance_matrix[chosen_index, index]:.2f})")

if __name__ == "__main__":
    main()
