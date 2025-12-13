import numpy as np
from PIL import Image
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from typing import List

import argparse

import os
import pandas as pd

def read_paths_and_labels(input_folder: str):
    train_csv_path = os.path.join(input_folder, 'train_data.csv')
    # test_csv_path = os.path.join(input_folder, 'test_data.csv')

    train_df = pd.read_csv(train_csv_path)
    # test_df = pd.read_csv(test_csv_path)

    return train_df

def load_and_preprocess_images(df: pd.DataFrame, data_root: str):
    filepaths = [os.path.join(data_root, fp) for fp in df['image_path'].tolist()]
    labels = df['label'].tolist()

    images = []
    labels_encoded = []
    for imagePath, label in zip(filepaths, labels):
        image = Image.open(imagePath).convert("RGB")
        image = image.resize((224, 224))
        image = np.array(image)
        assert image.shape == (224, 224, 3), f"Unexpected image shape: {image.shape}, expected (224, 224, 3)"
        images.append(image)

        if label == "Acne":
            labels_encoded.append(1)
        else:
            labels_encoded.append(0)
    
    images_array = np.array(images)
    labels_encoded = np.array(labels_encoded, dtype=np.float32)

    return images_array, labels_encoded 

def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, dest='input_folder', help='Input folder containing the train and test csv files')
    parser.add_argument('--labels_folder', type=str, dest='labels_folder', help='Folder containing the labels')
    parser.add_argument('--output_folder', type=str, dest='output_folder', help='Output folder')

    args = parser.parse_args()
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))
    
    labels_folder = args.labels_folder
    print('Labels folder:', labels_folder)
    train_df = read_paths_and_labels(labels_folder)
    output_folder = args.output_folder

    X_train, y_train = load_and_preprocess_images(train_df, args.input_folder)
    print('Shapes:')
    print(X_train.shape)
    print(len(y_train))

    model = create_model()
    model.fit(X_train, np.array(y_train), epochs=1, batch_size=32)

    model_output_path = os.path.join(output_folder, 'trained_model.h5')
    model.save(model_output_path)
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    main()