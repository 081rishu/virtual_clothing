import os

dataset_dir = "../dataset/"
os.makedirs(dataset_dir, exist_ok=True)

kaggle_json = os.path.expanduser('kaggle.json')
if not os.path.exists(kaggle_json):
    print("Error: kaggle.json file not found!")
    exit(1)

os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
print("Kaggle API credentials verified.")
print(f"Downloading dataset to {dataset_dir}...")


result = os.system(f'kaggle datasets download -d rajkumarl/people-clothing-segmentation -p {dataset_dir}')
if result == 0:
    print("Dataset downloaded successfully.")
else:
    print("Error downloading dataset.")

    