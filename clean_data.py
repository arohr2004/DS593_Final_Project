from datasets import load_dataset, DatasetDict

#all entries with missing height have a value of -1
def filter_missing_height(example):
    return example["height"] != -1

def main():
    dataset = load_dataset("alecccdd/celeb-fbi") 

    cleaned_dataset = dataset.filter(filter_missing_height)

    train_val_split = cleaned_dataset['train'].train_test_split(test_size=0.1)

    final_dataset = DatasetDict({ #80% train, 10% validation, 10% test
        'train': train_val_split['train'],
        'validation': train_val_split['test'], 
        'test': cleaned_dataset['test']        
    })
    final_dataset.save_to_disk("./cleaned_data")
    
if __name__ == "__main__":
    main()

    chmod g+s /projectnb/cds593/593_arohr

###############################################################################
#
# from datasets import load_from_disk
# from torch.utils.data import DataLoader
#
# dataset = load_from_disk("./cleaned_data")
#
# dataset.set_format(type="torch", columns=["image", "height", "weight", "gender", "age"])
#
# train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
# val_loader = DataLoader(dataset["validation"], batch_size=32)
# -----------------------------------------------------------------------------
# 
# Note: The images are loaded as raw PIL images. You may need to apply standard 
# torchvision transforms (like resizing) before passing them to the model!
#
###############################################################################