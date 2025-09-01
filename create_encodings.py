import os
import face_recognition  # ✅ Fixed incorrect import
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import pandas as pd


def _get_training_dirs(training_dir_path):
    return [x[0] for x in os.walk(training_dir_path)][1:]


def _get_training_labels(training_dir_path):
    return [x[1] for x in os.walk(training_dir_path)][0]


def _get_each_labels_files(training_dir_path):
    return [x[2] for x in os.walk(training_dir_path)][1:]


def _filter_image_files(training_dir_path):
    exts = [".jpg", ".jpeg", ".png"]

    training_folder_files_list = []
    for list_files in _get_each_labels_files(training_dir_path):
        filtered_files = [file for file in list_files if os.path.splitext(file)[1].lower() in exts]
        training_folder_files_list.append(filtered_files)

    return training_folder_files_list


def _zipped_folders_labels_images(training_dir_path, labels):
    return list(zip(_get_training_dirs(training_dir_path),
                    labels,
                    _filter_image_files(training_dir_path)))


def create_dataset(training_dir_path, labels):
    X = []
    for folder_path, label, images in _zipped_folders_labels_images(training_dir_path, labels):
        for file_name in images:
            file_path = os.path.join(folder_path, file_name)
            img = face_recognition.load_image_file(file_path)
            img_encodings = face_recognition.face_encodings(img)

            if len(img_encodings) > 1:
                print(f'\x1b[0;37;43mMore than one face found in {file_path}. Only considering the first face.\x1b[0m')
            if len(img_encodings) == 0:
                print(f'\x1b[0;37;41mNo face found in {file_path}. Ignoring file.\x1b[0m')
            else:
                print(f'Encoded {file_path} successfully.')
                X.append(np.append(img_encodings[0], label))

    return X


encoding_file_path = './encoded-images-data.csv'
training_dir_path = './training-images'
labels_fName = "labels.pkl"

# Get the folder names in training-dir as labels
labels = _get_training_labels(training_dir_path)
le = LabelEncoder().fit(labels)
labels_num = le.transform(labels)
num_classes = len(le.classes_)
dataset = create_dataset(training_dir_path, labels_num)
df = pd.DataFrame(dataset)

# Backup the existing encoding file
if os.path.isfile(encoding_file_path):
    print(f"{encoding_file_path} already exists. Backing up.")
    os.rename(encoding_file_path, f"{encoding_file_path}.bak")

df.to_csv(encoding_file_path, index=False)

print(f"{num_classes} classes created.")
print(f'\x1b[6;30;42mSaving labels pickle to {labels_fName}\x1b[0m')

with open(labels_fName, 'wb') as f:
    pickle.dump(le, f)

print(f'\x1b[6;30;42mTraining Image encodings saved in {encoding_file_path}\x1b[0m')
