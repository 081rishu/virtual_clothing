import zipfile
import os

zip_file_path = '../dataset/people-clothing-segmentation.zip'
extract_to = '../dataset'

if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Dataset extracted to {extract_to}")
else:
    print(f"Zip file {zip_file_path} not found.")