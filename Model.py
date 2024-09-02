import numpy as np
import pandas as pd
from PIL import Image
import os
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

sound_types = ['breathing_deep']

def preprocess_image(file_path):
    try:
        img = Image.open(file_path)
        img = img.resize((224, 224))  # Resize image as required by your CNN model
        img = np.array(img) / 255.0   # Normalize pixel values
        return img
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return None

def load_data(csv_file, image_dir):
    data = pd.read_csv(csv_file)
    
    data = data[data['Health status (e.g. : positive_mild, healthy,etc.)'].isin(['Positive', 'Negative'])]
    
    status_dict = data.set_index('User ID')['Health status (e.g. : positive_mild, healthy,etc.)'].to_dict()
    
    image_paths = []
    for sound_type in sound_types:
        image_paths.extend(glob.glob(os.path.join(image_dir, '202*', '*', f'{sound_type}-*-mel_spectrogram.png')))
    
    image_data = []
    labels = []
    for img_path in tqdm(image_paths, desc="Processing images"):
        user_id = os.path.basename(img_path).split('-')[1]
        if user_id in status_dict:
            img = preprocess_image(img_path)
            if img is not None:
                image_data.append(img)
                labels.append(1 if status_dict[user_id] == 'Positive' else 0)
    
    X = np.array(image_data)
    y = np.array(labels)
    
    # Print some diagnostics
    print(f"Total images: {len(X)}")
    print(f"Total labels: {len(y)}")
    print(f"Positive samples: {len(data[data['Health status (e.g. : positive_mild, healthy,etc.)'].isin(['Positive'])])}")
    print(f"Negative samples: {len(data[data['Health status (e.g. : positive_mild, healthy,etc.)'].isin(['Negative'])])}")
    
    return X, y

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification output
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = build_model(X_train[0].shape)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    return model, history, X_test, y_test

def predict_random_users(csv_file, image_dir, model, num_samples=3):
    data = pd.read_csv(csv_file)
    
    data = data[data['Health status (e.g. : positive_mild, healthy,etc.)'].isin(['Positive', 'Negative'])]
    
    positive_ids = data[data['Health status (e.g. : positive_mild, healthy,etc.)'] == 'Positive']['User ID'].tolist()
    negative_ids = data[data['Health status (e.g. : positive_mild, healthy,etc.)'] == 'Negative']['User ID'].tolist()
    
    selected_positive_ids = random.sample(positive_ids, num_samples)
    selected_negative_ids = random.sample(negative_ids, num_samples)
    
    selected_ids = selected_positive_ids + selected_negative_ids
    
    for user_id in selected_ids:
        actual_status = data[data['User ID'] == user_id]['Health status (e.g. : positive_mild, healthy,etc.)'].values[0]
        for sound_type in sound_types:
            img_path = glob.glob(os.path.join(image_dir, '202*', '*', f'{sound_type}-{user_id}-mel_spectrogram.png'))
            if img_path:
                img = preprocess_image(img_path[0])
                prediction = model.predict(np.expand_dims(img, axis=0))
                predicted_status = 'Positive' if prediction > 0.5 else 'Negative'
                actual_status_str = 'Positive' if actual_status == 'Positive' else 'Negative'
                print(f"User ID: {user_id}, Sound Type: {sound_type}, Actual COVID Status: {actual_status_str}, Predicted COVID Status: {predicted_status}")

csv_file = '/Users/manraj/Desktop/Coswara-Data/newDataframe.csv'
image_dir = '/Users/manraj/Desktop/Coswara-Data/Extracted_data'

X, y = load_data(csv_file, image_dir)

model, history, X_test, y_test = train_model(X, y)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

librosa_csv_file = '/Users/manraj/Desktop/Coswara-Data/librosa_features.csv'
predict_random_users(librosa_csv_file, image_dir, model)

y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)

conf_matrix = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
