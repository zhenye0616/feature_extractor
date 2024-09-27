import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST,CIFAR10, SVHN
import medmnist
from medmnist import INFO, Evaluator
from skimage.feature import graycomatrix, graycoprops
import sys


def load_dataset(dataset_name):
    """
    Load the specified dataset (MNIST or FashionMNIST).

    Parameters:
    - dataset_name: The name of the dataset to load ('mnist' or 'fashion_mnist').

    Returns:
    - dataset: The loaded dataset.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    
    if dataset_name.lower() == 'mnist':
        print("Loading MNIST dataset...")
        dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset_name.lower() == 'fashion_mnist':
        print("Loading Fashion MNIST dataset...")
        dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
    else:
        raise ValueError("Invalid dataset name. Choose either 'mnist' or 'fashion_mnist'.")
    
    return dataset


def load_medmnist_dataset(dataset_name):
    info = INFO[dataset_name]
    print("f Loadinng {info} ...")
    DataClass = getattr(medmnist, info['python_class'])
    transform = transforms.Compose([transforms.ToTensor()])

    # Load train dataset
    train_dataset = DataClass(split='train', transform=transform, download=True)
    return train_dataset


def load_cifar10():
    print("Loading CIFAR10 ...")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    return cifar10_dataset


def load_svhn():
    print("Loading SVHN ...")
    transform = transforms.Compose([
        transforms.ToTensor()  # Convert the images to tensors
    ])
    svhn_dataset = SVHN(root='./data', split='train', download=True, transform=transform)
    return svhn_dataset


def calculate_curvature(contour, k=5):
    curvatures = []
    contour = np.squeeze(contour)
    for i in range(k, len(contour) - k):
        p1 = contour[i - k]
        p2 = contour[i]
        p3 = contour[i + k]

        v1 = p1 - p2
        v2 = p3 - p2

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            cosine_angle = 0
        else:
            cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)

        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        curvature = np.abs(angle)
        
        curvatures.append(curvature)

    return np.array(curvatures)


def extract_curvatures(image_tensor):
    # Convert tensor to NumPy array and reshape to a 2D grayscale image
    image = image_tensor.squeeze().numpy()

    # Check the number of channels in the image
    if len(image.shape) == 3:  # Multi-channel image
        if image.shape[2] == 3 or image.shape[2] == 4:
            # If it's an RGB or RGBA image, convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # If the image has an unusual number of channels (like 28), take the first channel or mean
            #print(f"Image has {image.shape[2]} channels. Selecting the first channel.")
            image = image[:, :, 0]  # np.mean(image, axis=2)

    # Ensure the image is in uint8 format
    image = (image * 255).astype(np.uint8)

    # Threshold the image
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    cnt = contours[0]

    # Calculate curvature (simplified curvature calculation)
    curvatures = calculate_curvature(cnt)
    return curvatures


def extract_shapes(image_tensor):
    # Convert tensor to NumPy array
    image = image_tensor.squeeze().numpy()

    # Ensure the image is grayscale (single channel)
    if len(image.shape) == 3:  # Multi-channel image
        if image.shape[2] == 3 or image.shape[2] == 4:
            # If it's an RGB or RGBA image, convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # If the image has an unusual number of channels (like 28), take the first channel or mean
            #print(f"Image has {image.shape[2]} channels. Selecting the first channel.")
            image = image[:, :, 0]  # np.mean(image, axis=2)

    # Convert the image to uint8
    image = (image * 255).astype(np.uint8)

    # Threshold the image to create a binary version
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = contours[0]

    # Calculate shape features (area, perimeter)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # Calculate aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h != 0 else 0  # Avoid division by zero

    return area, perimeter, aspect_ratio


def extract_color_features(image_tensor):
    # Convert tensor to NumPy array
    image = image_tensor.squeeze().numpy()

    if image.shape[0] == 3:  # RGB image
        red_channel = image[0]
        green_channel = image[1]
        blue_channel = image[2]

        # Compute color features (mean and std for each channel)
        red_mean, red_std = np.mean(red_channel), np.std(red_channel)
        green_mean, green_std = np.mean(green_channel), np.std(green_channel)
        blue_mean, blue_std = np.mean(blue_channel), np.std(blue_channel)

        return [red_mean, red_std, green_mean, green_std, blue_mean, blue_std]
    else:
        # If the image is not in RGB format, return None
        print("Image not in RGB format")
        return None
    

def extract_glcm_features(image_tensor):
    """
    Extract texture features using Gray Level Co-occurrence Matrix (GLCM).
    
    Parameters:
    - image_tensor: Input image tensor
    
    Returns:
    - glcm_features: List of GLCM texture features
    """
    # Convert tensor to NumPy array
    image = image_tensor.squeeze().numpy()

    # Check the number of channels in the image
    if len(image.shape) == 3:
        if image.shape[2] == 3:  # RGB image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 32:  # If there are 32 channels, take the first 3 or average them
            image = np.mean(image[:, :, :3], axis=2)  # Average the first three channels
            image = image.astype(np.uint8)  # Ensure it's in uint8 format
        else:
            raise ValueError("Invalid number of channels in input image: {}".format(image.shape[2]))
    elif len(image.shape) == 2:  # Grayscale image
        image = image.astype(np.uint8)  # Ensure it's in uint8 format
    else:
        raise ValueError("Invalid image shape: {}".format(image.shape))

    # Ensure image is in uint8 format
    image = (image * 255).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Compute texture properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    # Return texture features as a list
    glcm_features = [contrast, dissimilarity, homogeneity, energy, correlation, asm]
    
    return glcm_features    



def main(dataset_name):
    shape_features = []
    curvature_features = []
    color_features = []
    texture_features = []

    if dataset_name == 'svhn':
        dataset = load_svhn()
        #Handle SVHN dataset
    elif dataset_name == 'cifar10':
        #Handle CIFAR10 dataset
        dataset = load_cifar10()
    elif dataset_name in INFO:
        # Handle MedMNIST datasets
        dataset = load_medmnist_dataset(dataset_name)
    else:
        # Handle regular MNIST/FashionMNIST datasets
        dataset = load_dataset(dataset_name)

    for i in range(len(dataset)):
        image_tensor, _ = dataset[i]
        curvature = extract_curvatures(image_tensor)
        shape = extract_shapes(image_tensor)
        color = extract_color_features(image_tensor)
        textures = extract_glcm_features(image_tensor)
        if curvature is not None:
            curvature_features.append(curvature)
        if shape is not None:
            shape_features.append(shape)
        if color is not None:
            color_features.append(color)
        if textures is not None:
            texture_features.append(textures)
        
    curvature_features = np.array(curvature_features, dtype=object)
    shape_features = np.array(shape_features, dtype=object)
    color_features = np.array(color_features, dtype=object)
    texture_features = np.array(texture_features, dtype=object)

    print("Curvature features shape:", np.shape(curvature_features))
    print("Shape features shape:", np.shape(shape_features))
    print("Color features shape:", np.shape(color_features))
    print("texture features shape:", np.shape(texture_features))


    # Dynamically name the files based on the dataset name
    curvature_file = f'{dataset_name}_curvatures.npy'
    shape_file = f'{dataset_name}_shapes.npy'
    color_file = f'{dataset_name}_color_features.npy'
    texture_file = f'{dataset_name}_textures.npy'

    try:
        np.save(curvature_file, curvature_features)
        np.save(shape_file, shape_features)
        np.save(color_file, color_features)
        np.save(texture_file, texture_features)
    except Exception as e:
        print("Error saving file:", e)

    print(f"Shape, curvature, color and texture features are extracted and saved for {dataset_name}.")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python noise.py <dataset_name>")
        print("Valid dataset names: 'mnist' or 'fashion_mnist', or 'MedMNIST'(e.g., 'pathmnist') or 'svhn' or 'cifar10' dataset names")
    else:
        dataset_name = sys.argv[1]
        main(dataset_name)
