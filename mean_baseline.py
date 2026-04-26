import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms
from tqdm import tqdm

# MEAN BASELINE PREDICTOR

# Minimal transform to get target_height as tensor
def apply_minimal_transforms(examples):
    return {
        "target_height": [torch.tensor(float(h), dtype=torch.float32) for h in examples["height"]]
    }

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
    print(f"Average MSE per training sample: {(mean_height/len(train_heights)):.2f}")


    # 3. Evaluate on Validation Set
    val_loader = DataLoader(dataset["validation"], batch_size=16)
    criterion = nn.MSELoss()
    val_loss = 0.0
    val_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating on validation set"):
            targets = batch["target_height"].to(device)
            predictions = torch.full_like(targets, mean_height)
            loss = criterion(predictions, targets)
            val_loss += loss.item() * targets.size(0)  # Multiply by batch size to get sum
            val_samples += targets.size(0)
    val_mse = val_loss / val_samples
    print(f"Validation MSE: {val_mse:.2f}")
    print(f"Number of validation samples: {val_samples}")
    print(f"Average MSE per validation sample: {(val_mse/val_samples):.2f}")


    # 4. Evaluate on Test Set
    test_loader = DataLoader(dataset["test"], batch_size=16)
    test_loss = 0.0
    test_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            targets = batch["target_height"].to(device)
            predictions = torch.full_like(targets, mean_height)
            loss = criterion(predictions, targets)
            test_loss += loss.item() * targets.size(0)
            test_samples += targets.size(0)
    test_mse = test_loss / test_samples
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Number of test samples: {test_samples}")
    print(f"Average MSE per test sample: {(test_mse/test_samples):.2f}")

    # 5. Compute Global Mean from All Data
    all_heights = train_heights + [example["target_height"].item() for example in dataset["validation"]] + [example["target_height"].item() for example in dataset["test"]]
    global_mean_height = sum(all_heights) / len(all_heights)
    print(f"\nGlobal mean height from all data: {global_mean_height:.2f}")

    # 6. Evaluate Global Mean on Test Set
    global_test_loss = 0.0
    test_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating global mean on test set"):
            targets = batch["target_height"].to(device)
            predictions = torch.full_like(targets, global_mean_height)
            loss = criterion(predictions, targets)
            global_test_loss += loss.item() * targets.size(0)
            test_samples += targets.size(0)
    global_test_mse = global_test_loss / test_samples
    print(f"Test MSE with global mean: {global_test_mse:.2f}")
    print(f"Number of global samples: {test_samples}")
    print(f"Average MSE per test sample with global mean: {(global_test_mse/test_samples):.2f}")

if __name__ == "__main__":
    main()

#./env/bin/python mean_baseline.py