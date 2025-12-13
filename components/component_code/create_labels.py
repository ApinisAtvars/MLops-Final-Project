import os
import argparse
import logging
from glob import glob
import math
import random
import csv

def main():
    SEED = 25_05_2004
    random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data_root", type=str, help="Path to the root folder containing folders with images to label")
    parser.add_argument("--output_folder", type=str, help="Path to output folder where labels will be saved")
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("Input data root:", args.path_to_data_root)
    print("Output folder:", args.output_folder)
    
    os.makedirs(args.output_folder, exist_ok=True)

    training_datapaths = []
    testing_datapaths = []

    # Process train data
    train_dir = os.path.join(args.path_to_data_root, "train")
    if os.path.exists(train_dir):
        for class_folder in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_folder)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    full_image_path = os.path.join(class_path, image_name)
                    if os.path.isfile(full_image_path):
                        # Store relative path instead of absolute path
                        rel_path = os.path.relpath(full_image_path, args.path_to_data_root)
                        training_datapaths.append((rel_path, class_folder))
    
    # Process test data
    test_dir = os.path.join(args.path_to_data_root, "test")
    if os.path.exists(test_dir):
        for class_folder in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_folder)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    full_image_path = os.path.join(class_path, image_name)
                    if os.path.isfile(full_image_path):
                        # Store relative path instead of absolute path
                        rel_path = os.path.relpath(full_image_path, args.path_to_data_root)
                        testing_datapaths.append((rel_path, class_folder))

    # Shuffle
    random.shuffle(training_datapaths)
    random.shuffle(testing_datapaths)

    # Save
    train_output_path = os.path.join(args.output_folder, "train_data.csv")
    test_output_path = os.path.join(args.output_folder, "test_data.csv")

    with open(train_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(training_datapaths)
    
    with open(test_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(testing_datapaths)

    print(f"Saved {len(training_datapaths)} training samples to {train_output_path}")
    print(f"Saved {len(testing_datapaths)} testing samples to {test_output_path}")

if __name__ == "__main__":
    main()