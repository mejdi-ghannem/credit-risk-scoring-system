import os
import sys

# Add project root to import path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data_preparation import prepare_dataset

def main():
    raw_data_path = os.path.join(project_root, "data", "raw")
    processed_data_path = os.path.join(project_root, "data", "processed")

    # Ensure processed folder exists
    os.makedirs(processed_data_path, exist_ok=True)

    print("Preparing training data...")
    train_data = prepare_dataset(raw_data_path, is_train=True)
    train_data.to_csv(os.path.join(processed_data_path, "train_clean.csv"), index=False)

    print("Preparing test data...")
    test_data = prepare_dataset(raw_data_path, is_train=False)
    test_data.to_csv(os.path.join(processed_data_path, "test_clean.csv"), index=False)

    print("✅ Data preparation completed and saved.")

if __name__ == "__main__":
    main()
