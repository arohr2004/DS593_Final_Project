import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms
from transformers import AutoModel
from tqdm import tqdm

# ==========================================
# 1. ARCHITECTURE: DINOv2 + REGRESSION HEAD
# ==========================================
class DINOv2ForHeightRegression(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()

        self.backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        
        #Freeze DINOv2 so we only train the new head
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Regression head to predict height from DINOv2's CLS token
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

# DINOv2 demands 224x224 images, normalized with specific ImageNet colors
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def apply_transforms(examples):
    # Convert HF PIL images to PyTorch tensors dynamically
    examples["pixel_values"] = [image_transform(img.convert("RGB")) for img in examples["image"]]
    examples["target_height"] = [float(h) for h in examples["height"]]
    return examples

# ==========================================
# 3. THE TRAINING LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading prepared dataset...")
    dataset = load_from_disk("./cleaned_data")
    
    dataset = dataset.with_transform(apply_transforms)

    train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=32)

    model = DINOv2ForHeightRegression(freeze_backbone=True).to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) 

    epochs = 5
    print("Starting Training...")
    
    for epoch in range(epochs):
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
                
        train_mse = (train_loss/len(train_loader)) 
        val_mse = (val_loss/len(val_loader)) 
        
        print(f"Result -> Train MSE: {train_mse:.2f} cm | Val MSE: {val_mse:.2f} cm\n")

    # ==========================================
    # 4. SAVING THE MODEL
    # ==========================================
    print("\nSaving the trained model...")
    torch.save(model.state_dict(), "./dinov2.pth")
    print("Model saved to ./dinov2.pth")

    # ==========================================
    # 5. FINAL EVALUATION (TEST SET)
    # ==========================================
    print("\n--- FINAL EVALUATION ON UNSEEN TEST SET ---")

    test_loader = DataLoader(dataset["test"], batch_size=32)

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
    print(f"FINAL TEST MSE: {test_mse:.2f}")

if __name__ == "__main__":
    main()