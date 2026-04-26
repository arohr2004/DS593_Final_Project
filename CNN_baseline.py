import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv

# ==========================================
# SIMPLE CNN FOR HEIGHT REGRESSION
# ==========================================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Reduced filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced filters
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)  # Adjusted for 32 channels
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)  # Adjusted
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ==========================================
# IMAGE PREPROCESSING
# ==========================================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def apply_transforms(examples):
    return {
        "pixel_values": [image_transform(img.convert("RGB")) for img in examples["image"]],
        "target_height": [torch.tensor(float(h), dtype=torch.float32) for h in examples["height"]]
    }

# ==========================================
# TRAINING AND EVALUATION
# ==========================================
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["pixel_values"].to(device)
            targets = batch["target_height"].to(device)
            predictions = model(images)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
    mse = total_loss / total_samples
    return mse

def main():
    # ==========================================
    # DEVICE CONFIGURATION
    # ==========================================
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 1. Load Data
    dataset = load_from_disk("./cleaned_data")
    dataset = dataset.with_transform(apply_transforms)
    
    train_samples = len(dataset["train"])
    val_samples = len(dataset["validation"])
    test_samples = len(dataset["test"])
    
    plot_dir = "/projectnb/cds593/593_arohr/Plots/CNN_baseline"
    os.makedirs(plot_dir, exist_ok=True)
    
    train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)  # Increased batch size
    val_loader = DataLoader(dataset["validation"], batch_size=32)
    test_loader = DataLoader(dataset["test"], batch_size=32)

    # 2. Model, Optimizer, Loss
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    epochs = 5  # Reduced epochs
    train_mses = []
    val_mses = []
    csv_data = [["Epoch", "Train_MSE", "Val_MSE", "Train_Samples", "Val_Samples"]]  # CSV header

    print("\nStarting CNN Training...")
    
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = batch["pixel_values"].to(device)
            targets = batch["target_height"].to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
        train_mse = train_loss / train_samples
        train_mses.append(train_mse)
        
        # --- VALIDATION ---
        val_mse = evaluate(model, val_loader, criterion, device)
        val_mses.append(val_mse)
        
        print(f"Epoch {epoch+1}: Train MSE: {train_mse:.2f} | Val MSE: {val_mse:.2f}")
        csv_data.append([epoch + 1, train_mse, val_mse, train_samples, val_samples])

    # ==========================================
    # FINAL EVALUATION ON TEST SET (after training)
    # ==========================================
    test_mse = evaluate(model, test_loader, criterion, device)
    test_samples = len(dataset["test"])
    print(f"\nFINAL TEST MSE: {test_mse:.2f}")

    # ==========================================
    # COMPUTE AVERAGE MSE OVER EPOCHS
    # ==========================================
    avg_train_mse = sum(train_mses) / len(train_mses)
    avg_val_mse = sum(val_mses) / len(val_mses)
    print(f"\nAverage MSE over {epochs} epochs:")
    print(f"Train: {avg_train_mse:.2f} | Val: {avg_val_mse:.2f}")

    # Write average row to CSV as well
    csv_data.append(["Average", avg_train_mse, avg_val_mse, train_samples, val_samples])
    
    # Write test result to CSV
    csv_data.append(["Test", test_mse, "", "", test_samples])

    # ==========================================
    # SAVE MSE HISTORY TO CSV
    # ==========================================
    csv_filename = "cnn_mse_history.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
    print(f"\nSaved MSE history to {csv_filename}")

    # ==========================================
    # PLOTTING
    # ==========================================
    epoch_list = list(range(1, epochs + 1))
    
    # MSE over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_mses, label='Train MSE', marker='o')
    plt.plot(epoch_list, val_mses, label='Validation MSE', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('CNN Baseline: MSE over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'cnn_mse_epochs.png'))
    plt.show()

    # Final MSE comparison (final epoch train/val and test)
    plt.figure(figsize=(8, 5))
    splits = ['Train (Final Epoch)', 'Validation (Final Epoch)', 'Test (Final)']
    final_mses = [train_mses[-1], val_mses[-1], test_mse]
    plt.bar(splits, final_mses, color=['blue', 'orange', 'green'])
    plt.ylabel('MSE')
    plt.title('CNN Baseline: Final Epoch Train/Val MSE and Test MSE')
    plt.savefig(os.path.join(plot_dir, 'cnn_final_mse.png'))
    plt.show()

    # Average MSE over epochs (cumulative average)
    cumulative_train_mse = [sum(train_mses[:i+1])/(i+1) for i in range(len(train_mses))]
    cumulative_val_mse = [sum(val_mses[:i+1])/(i+1) for i in range(len(val_mses))]

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, cumulative_train_mse, label='Cumulative Avg Train MSE', marker='o')
    plt.plot(epoch_list, cumulative_val_mse, label='Cumulative Avg Validation MSE', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('CNN Baseline: Cumulative Average MSE over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'cnn_mse_avg_epochs.png'))
    plt.show()

    # Average final MSE comparison
    plt.figure(figsize=(8, 5))
    splits_2 = ['Train', 'Validation', 'Test']
    avg_mses = [avg_train_mse, avg_val_mse, test_mse]
    plt.bar(splits_2, avg_mses, color=['navy', 'darkorange', 'darkgreen'])
    plt.ylabel('Average MSE')
    plt.title('CNN Baseline: Average Train/Val MSE and Final Test MSE')
    for idx, value in enumerate(avg_mses):
        plt.text(idx, value + 0.5, f"{value:.2f}", ha='center', va='bottom')
    plt.savefig(os.path.join(plot_dir, 'cnn_final_mse_avg.png'))
    plt.show()

    print(f"\nPlots saved to {plot_dir}:")
    print("cnn_mse_epochs.png, cnn_final_mse.png, cnn_mse_avg_epochs.png, cnn_final_mse_avg.png")

if __name__ == "__main__":
    main()