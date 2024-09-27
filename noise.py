import numpy as np
import sys

# Load features from a .npy file
def load_features(file_path):
    try:
        features = np.load(file_path, allow_pickle=True)
        print(f"Successfully loaded features from {file_path}")
        return features
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading features: {e}")
        return None

# Add Gaussian noise directly to the features (curvatures or shapes)
def add_gaussian_noise(features, mean=0, std=0.1):
    if features is None:
        print("Features are None, skipping noise addition.")
        return None
    noise = np.random.randn(*features.shape) * std + mean  # Generate Gaussian noise
    noisy_features = features + noise  # Add the noise to the features
    return noisy_features

# Flatten the list of arrays into a single array
def flatten_features(features):
    return np.concatenate([np.ravel(f) for f in features])

# Calculate L2 distance between original and noisy features
def calculate_l2_distance(original_features, noisy_features):
    if original_features is None or noisy_features is None:
        print("One or both feature sets are None, cannot calculate L2 distance.")
        return None
    # Flatten the feature lists if they contain arrays of different lengths
    original_features_flat = flatten_features(original_features)
    noisy_features_flat = flatten_features(noisy_features)
    
    # Calculate L2 distance between the flattened arrays
    l2_dist = np.linalg.norm(original_features_flat - noisy_features_flat)
    return l2_dist

def main(dataset_name):
    curvatures_file = f"/home/biaslab/Zhen/feature_extraction/{dataset_name}_curvatures.npy"
    shapes_file = f"/home/biaslab/Zhen/feature_extraction/{dataset_name}_shapes.npy"
    color_file = f"/home/biaslab/Zhen/feature_extraction/{dataset_name}_color_features.npy"
    texture_file = f"/home/biaslab/Zhen/feature_extraction/{dataset_name}_textures.npy"
   
    curvatures = load_features(curvatures_file)
    noisy_curvatures = add_gaussian_noise(curvatures)

    shapes = load_features(shapes_file)
    noisy_shapes = add_gaussian_noise(shapes)

    color = load_features(color_file)
    noisy_color = add_gaussian_noise(color)
    
    textures = load_features(texture_file)
    noisy_textures = add_gaussian_noise(textures)

    

    if curvatures is not None and noisy_curvatures is not None:
        curvature_l2_distance = calculate_l2_distance(curvatures, noisy_curvatures)
        print(f"L2 distance for curvatures: {curvature_l2_distance}")
    else:
        print("Cannot calculate L2 distance for curvatures due to missing data.")

    if shapes is not None and noisy_shapes is not None:
        shape_l2_distance = calculate_l2_distance(shapes, noisy_shapes)
        print(f"L2 distance for shapes: {shape_l2_distance}")
    else:
        print("Cannot calculate L2 distance for shapes due to missing data.")

    if color is not None and noisy_color is not None:
        color_l2_distance = calculate_l2_distance(color, noisy_color)
        print(f"L2 distance for color: {color_l2_distance}")
    else:
        print("Cannot calculate L2 distance for color due to missing data.")

    if textures is not None and noisy_textures is not None:
        texture_l2_distance = calculate_l2_distance(textures, noisy_textures)
        print(f"L2 distance for textures: {texture_l2_distance}")
    else:
        print("Cannot calculate L2 distance for textures due to missing data.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python noise.py <dataset_name>")
        print("Example: python noise.py mnist or python noise.py fashion_mnist")
    else:
        dataset_name = sys.argv[1]
        main(dataset_name)


