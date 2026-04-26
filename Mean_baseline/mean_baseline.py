import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm

# MEAN BASELINE PREDICTOR

# Minimal transform to get target_height as tensor
def apply_minimal_transforms(examples):
    return {
        "target_height": [torch.tensor(float(h), dtype=torch.float32) for h in examples["height"]]
    }


def evaluate_mean(dataset_split, baseline_value, device, criterion, split_name, batch_size=16):
    loader = DataLoader(dataset_split, batch_size=batch_size)
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating on {split_name} set"):
            targets = batch["target_height"].to(device)
            predictions = torch.full_like(targets, baseline_value)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

    mse = total_loss / total_samples
    return mse, total_samples


def compute_mse_from_heights(heights, baseline_value):
    squared_errors = [(height - baseline_value) ** 2 for height in heights]
    return sum(squared_errors) / len(squared_errors)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data with minimal transforms
    dataset = load_from_disk("./cleaned_data")
    dataset = dataset.with_transform(apply_minimal_transforms)

    # 2. Compute Mean Height from Training Set
    train_heights = []
    for example in tqdm(dataset["train"], desc="Computing mean from train set"):
        train_heights.append(example["target_height"].item())
    
    mean_height = sum(train_heights) / len(train_heights)
    print(f"Mean height from training set: {mean_height:.2f}")
    print(f"Number of training samples: {len(train_heights)}")

    criterion = nn.MSELoss()

    train_mse, train_samples = evaluate_mean(
        dataset["train"], mean_height, device, criterion, "train"
    )
    print(f"Train MSE: {train_mse:.2f}")

    # 3. Evaluate on Validation Set
    val_mse, val_samples = evaluate_mean(
        dataset["validation"], mean_height, device, criterion, "validation"
    )
    print(f"Validation MSE: {val_mse:.2f}")
    print(f"Number of validation samples: {val_samples}")


    # 4. Evaluate on Test Set
    test_mse, test_samples = evaluate_mean(
        dataset["test"], mean_height, device, criterion, "test"
    )
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Number of test samples: {test_samples}")

    # 5. Compute Global Mean from All Data
    all_heights = train_heights + [example["target_height"].item() for example in dataset["validation"]] + [example["target_height"].item() for example in dataset["test"]]
    global_mean_height = sum(all_heights) / len(all_heights)
    print(f"\nGlobal mean height from all data: {global_mean_height:.2f}")

    # 6. Evaluate Global Mean on All Cleaned Data Combined
    global_mse = compute_mse_from_heights(all_heights, global_mean_height)
    print(f"Global MSE (all cleaned data combined): {global_mse:.2f}")
    print(f"Number of global samples: {len(all_heights)}")

if __name__ == "__main__":
    main()

#./env/bin/python mean_baseline.py
