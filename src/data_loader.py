import yaml
import numpy as np
# import tensorflow as tf
# from tensorflow import keras
import os
import shutil

def load_data(params_yaml_path):
    with open(params_yaml_path) as yaml_file:
        params_yaml = yaml.safe_load(yaml_file)

    local_data = params_yaml['data_source']['local_path']
    destination_dir = params_yaml['split']['dir']
    
    # Define folder names for train and val
    train_folder = params_yaml['split']['train']
    val_folder = params_yaml['split']['test']  # assuming val is referred to as 'test' in your YAML

    # Create destination directories if they don't exist
    train_path = os.path.join(destination_dir, train_folder)
    val_path = os.path.join(destination_dir, val_folder)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Define the categories (subfolders like 'cats', 'dogs', 'snakes') that i have these categories
    categories = os.listdir(os.path.join(local_data, 'train'))  # assuming 'train' subfolder contains categories

    # Copy the data for each category in train and val folders
    for category in categories:
        # Copy train data
        src_train_category = os.path.join(local_data, 'train', category)
        dst_train_category = os.path.join(train_path, category)
        shutil.copytree(src_train_category, dst_train_category, dirs_exist_ok=True)

        # Copy val data
        src_val_category = os.path.join(local_data, 'val', category)
        dst_val_category = os.path.join(val_path, category)
        shutil.copytree(src_val_category, dst_val_category, dirs_exist_ok=True)

    print("Data has been successfully copied to the destination directory.")



if __name__ == "__main__":
    yaml_file_path = r"C:\Users\LokeshKanna\Downloads\dvc_version2\params.yaml"

    load_data(yaml_file_path)