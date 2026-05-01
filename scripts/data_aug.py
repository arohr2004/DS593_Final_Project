import random
from PIL import ImageDraw
from datasets import load_dataset, DatasetDict

# Filter out entries with missing heights (-1)
def filter_missing_height(example):
    return example["height"] != -1

# Augmentation function to black out the bottom of the image
def apply_bottom_occlusion(example):
    # Ensure the image is in RGB format to prevent color space issues
    img = example["image"].convert("RGB")
    
    # 50% chance to apply the occlusion (so the model learns both full and occluded bodies)
    if random.random() > 0.5:
        example["image"] = img
        return example
        
    width, height = img.size
    
    # Randomly occlude between 20% and 60% of the bottom of the image
    occlude_ratio = random.uniform(0.2, 0.6)
    occlude_height = int(height * occlude_ratio)
    
    x0 = 0
    y0 = height - occlude_height
    x1 = width
    y1 = height
    
    # Draw the black rectangle
    img_copy = img.copy() 
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    
    # Replace the original image with the augmented one
    example["image"] = img_copy
    return example

def main():
    print("Loading original dataset...")
    dataset = load_dataset("alecccdd/celeb-fbi") 

    print("Filtering out missing heights...")
    cleaned_dataset = dataset.filter(filter_missing_height)

    print("Splitting into Train, Validation, and Test...")
    # Creating an 80/10/10 split
    train_val_split = cleaned_dataset['train'].train_test_split(test_size=0.1)

    final_dataset = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'], 
        'test': cleaned_dataset['test']        
    })
    
    print("Applying bottom occlusion augmentation to the training set...")
    # We use .map() to permanently alter the data before saving.
    # Note: We ONLY apply this to 'train'. Val and Test remain pristine.
    final_dataset["train"] = final_dataset["train"].map(apply_bottom_occlusion)
    
    print("Saving augmented dataset to disk...")
    final_dataset.save_to_disk("./augmented_data")
    print("Done! Dataset saved to ./augmented_data")
    
if __name__ == "__main__":
    main()