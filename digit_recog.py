#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
data=pd.read_csv("D:/Datasets/speech_dataset.csv")
data.info()
data.describe()


# In[ ]:


get_ipython().system('pip install opencv-python scikit-image matplotlib numpy sklearn mahotas')


# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.datasets import load_digits
from skimage import measure
from mahotas import moments
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load digit dataset
digits = load_digits()
X, y = digits.images, digits.target

# Visualize some sample digits
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i], cmap='gray')
    plt.title(f'Label: {y[i]}')
plt.show()

# Feature extraction functions

# 1. Pixel Intensity (flatten the image)
def pixel_intensity(image):
    return image.flatten()

# 2. Harris Corner Detection
def corner_detection(image):
    gray = np.float32(image)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    return corners.flatten()

# 3. Histogram of Oriented Gradients (HOG)
def hog_features(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
    return fd, hog_image

# 4. Local Binary Patterns (Texture Features)
def texture_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    return lbp.flatten()

# 5. Edge Detection (Sobel)
def edge_detection(image):
    edges = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
    return edges.flatten()

# 6. Zernike Moments (using mahotas)
import mahotas

# Function to compute Zernike moments
def zernike_moments(image):
    # Convert image to binary
    thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
    # Compute Zernike moments with radius (adjust depending on the image size)
    radius = min(thresh.shape) // 2
    zernike_moments = mahotas.features.zernike_moments(thresh, radius)
    return zernike_moments


# Combine all features into a feature vector
def extract_features(image):
    pi = pixel_intensity(image)
    cd = corner_detection(image)
    hog_feat, hog_img = hog_features(image)
    lbp = texture_features(image)
    ed = edge_detection(image)
    zm = zernike_moments(image)
    
    # Concatenate all features into a single feature vector
    feature_vector = np.concatenate([pi, cd, hog_feat, lbp, ed, zm])
    return feature_vector, hog_img

# Apply feature extraction on all images in the dataset
feature_vectors = []
hog_images = []

for img in X:
    fv, hog_img = extract_features(img)
    feature_vectors.append(fv)
    hog_images.append(hog_img)

feature_vectors = np.array(feature_vectors)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.3, random_state=42)

# Train a Support Vector Machine (SVM) for classification
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Visualize HOG features for the first image
plt.figure(figsize=(5, 5))
plt.imshow(hog_images[0], cmap='gray')
plt.title('HOG Features')
plt.show()


# In[ ]:


get_ipython().system('pip install mahotas')


# In[ ]:


import mahotas
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# Load digit dataset
digits = load_digits()
X, y = digits.images, digits.target

# Display some sample images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i], cmap='gray')
    plt.title(f'Label: {y[i]}')
plt.axis('off')
plt.show()

# Feature extraction functions

# 1. Pixel Intensity (flatten the image)
def pixel_intensity(image):
    return image.flatten()

# 2. Harris Corner Detection with visualization
def corner_detection(image):
    gray = np.float32(image)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    # Plot corners
    plt.imshow(image, cmap='gray')
    plt.imshow(corners > 0.01 * corners.max(), cmap='jet', alpha=0.5)
    plt.title("Corner Detection")
    plt.show()
    return corners.flatten()

# 3. Histogram of Oriented Gradients (HOG) with visualization
def hog_features(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG Features")
    plt.show()
    return fd, hog_image

# 4. Local Binary Patterns (Texture Features) with visualization
def texture_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    plt.imshow(lbp, cmap='gray')
    plt.title("Local Binary Pattern (Texture)")
    plt.show()
    return lbp.flatten()

# 5. Edge Detection (Sobel) with visualization
def edge_detection(image):
    edges = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection (Sobel)")
    plt.show()
    return edges.flatten()

# 6. Zernike Moments (using mahotas) with visualization
def zernike_moments(image):
    thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
    radius = min(image.shape) // 2
    zernike_moments = mahotas.features.zernike_moments(thresh, radius)
    plt.imshow(thresh, cmap='gray')
    plt.title("Binary Image for Zernike Moments")
    plt.show()
    return zernike_moments

# Combine all features into a feature vector
def extract_features(image):
    pi = pixel_intensity(image)
    cd = corner_detection(image)
    hog_feat, _ = hog_features(image)
    lbp = texture_features(image)
    ed = edge_detection(image)
    zm = zernike_moments(image)
    feature_vector = np.concatenate([pi, cd, hog_feat, lbp, ed, zm])
    return feature_vector

# Apply feature extraction on all images in the dataset
feature_vectors = []
for img in X:
    feature_vectors.append(extract_features(img))
feature_vectors = np.array(feature_vectors)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.3, random_state=42)

# Train a Support Vector Machine (SVM) for classification
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict on the test set
y_pred = svm.predict(X_test)

# Model Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Additional Analysis: Distribution of HOG Features
sample_hog, _ = hog_features(X[0])
plt.hist(sample_hog, bins=20, color='blue', edgecolor='black')
plt.title("Distribution of HOG Features (Sample Image)")
plt.xlabel("HOG Feature Value")
plt.ylabel("Frequency")
plt.show()


# In[2]:


import mahotas
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

# Load digit dataset
digits = load_digits()
X, y = digits.images, digits.target


# In[3]:


plt.figure(figsize=(5,5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i], cmap='gray')
    plt.title(f'Label: {y[i]}')
    plt.axis('off')
plt.show()


# In[4]:


def pixel_intensity(image):
    return image.flatten()


# In[5]:


def corner_detection(image):
    gray = np.float32(image)  # Convert to float for precision
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    # Plot corners
    plt.imshow(image, cmap='gray')
    plt.imshow(corners > 0.01 * corners.max(), cmap='jet', alpha=0.5)
    plt.title("Corner Detection")
    plt.show()
    return corners.flatten()


# In[6]:


def hog_features(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG Features")
    plt.show()
    return fd, hog_image


# In[7]:


def texture_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    plt.imshow(lbp, cmap='gray')
    plt.title("Local Binary Pattern (Texture)")
    plt.show()
    return lbp.flatten()


# In[8]:


def edge_detection(image):
    edges = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=5)
    plt.imshow(edges, cmap='gray')
    plt.title("Edge Detection (Sobel)")
    plt.show()
    return edges.flatten()


# In[9]:


def zernike_moments(image):
    thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
    radius = min(image.shape) // 2
    zernike_moments = mahotas.features.zernike_moments(thresh, radius)
    plt.imshow(thresh, cmap='gray')
    plt.title("Binary Image for Zernike Moments")
    plt.show()
    return zernike_moments


# In[10]:


def extract_features(image):
    pi = pixel_intensity(image)
    cd = corner_detection(image)
    hog_feat, _ = hog_features(image)
    lbp = texture_features(image)
    ed = edge_detection(image)
    zm = zernike_moments(image)
    feature_vector = np.concatenate([pi, cd, hog_feat, lbp, ed, zm])
    return feature_vector


# In[11]:


feature_vectors = [extract_features(img) for img in X]
feature_vectors = np.array(feature_vectors)

X_train, X_test, y_train, y_test = train_test_split(feature_vectors, y, test_size=0.3, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)


# In[ ]:


y_pred = svm.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

sample_hog, _ = hog_features(X[0])
plt.hist(sample_hog, bins=20, color='blue', edgecolor='black')
plt.title("Distribution of HOG Features (Sample Image)")
plt.xlabel("HOG Feature Value")
plt.ylabel("Frequency")
plt.show()


# In[ ]:




