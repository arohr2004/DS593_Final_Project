import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms
from transformers import AutoModel
from tqdm import tqdm
import csv  # Added for CSV exporting

# ==========================================
# 1. ARCHITECTURE 
# ==========================================
class DINOv2ForHeightRegression(nn.Module):
    def __init__(self, freeze_backbone=False): 
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :] 
        predicted_height = self.regressor(cls_token)
        return predicted_height.squeeze(-1)

# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def apply_transforms(examples):
    # Return a brand new dictionary so the raw PIL images get left behind!
    return {
        "pixel_values": [image_transform(img.convert("RGB")) for img in examples["image"]],
        "target_height": [torch.tensor(float(h), dtype=torch.float32) for h in examples["height"]]
    }

# ==========================================
# 3. THE FINE-TUNING LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    dataset = load_from_disk("./augmented_data")
    dataset = dataset.with_transform(apply_transforms)
    
    train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=16)

    # 2. Load Model & Weights
    model = DINOv2ForHeightRegression(freeze_backbone=False).to(device)
    
    # You might want to handle exceptions here if dinov2.pth doesn't exist yet
    try:
        model.load_state_dict(torch.load("./dinov2.pth"), strict=False)
    except FileNotFoundError:
        print("Initial weights ./dinov2.pth not found. Starting from huggingface base weights.")

    # 3. Differential Learning Rates 
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5}, 
        {'params': model.regressor.parameters(), 'lr': 1e-4}
    ])
    
    criterion = nn.MSELoss()
    
    # --- MODIFICATION: Set epochs to 5 ---
    epochs = 5
    
    # --- MODIFICATION: Tracking variables for best model and CSV data ---
    best_val_mse = float('inf')
    csv_data = [["Epoch", "Train_MSE", "Val_MSE"]] # Setup header row

    print("\nStarting Fine-Tuning...")
    
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = batch["pixel_values"].to(device)
            targets = batch["target_height"].to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = batch["pixel_values"].to(device)
                targets = batch["target_height"].to(device)
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                
        train_mse = train_loss/len(train_loader)
        val_mse = val_loss/len(val_loader)
        
        print(f"Result -> Train MSE: {train_mse:.2f} | Val MSE: {val_mse:.2f}")

        # --- MODIFICATION: Record data for CSV ---
        csv_data.append([epoch + 1, train_mse, val_mse])
        
        # --- MODIFICATION: Save the best model ---
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            print(f"--> New best validation MSE ({best_val_mse:.2f})! Saving model weights...\n")
            torch.save(model.state_dict(), "./dinov2_BEST.pth")
        else:
            print("\n")

    # ==========================================
    # 4. Save the MSE History to CSV
    # ==========================================
    csv_filename = "mse_history.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)
    print(f"Saved MSE history for graphing to {csv_filename}")

    # ==========================================
    # 5. FINAL EVALUATION (TEST SET)
    # ==========================================
    print("\n--- FINAL EVALUATION ON UNSEEN TEST SET ---")
    
    # --- MODIFICATION: Load the BEST weights before testing ---
    print("Loading the best model weights for the test set...")
    model.load_state_dict(torch.load("./dinov2_BEST.pth"))
    
    test_loader = DataLoader(dataset["test"], batch_size=16)
    model.eval() 
    test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = batch["pixel_values"].to(device)
            targets = batch["target_height"].to(device)
            predictions = model(images)
            loss = criterion(predictions, targets)
            test_loss += loss.item()
            
    test_mse = test_loss / len(test_loader)
    print(f"\nFINAL TEST MSE: {test_mse:.2f}")
    
if __name__ == "__main__":
    main()