# streetclip_embedding_generation.py
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# general settings
CSV_PATH = "geoguessr_images_dataset.csv"
IMAGE_ROOT = "geoguessr_data/compressed_dataset"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints_embeddings"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
EMB_SAVE_INTERVAL = 100   # save every N images

# dataset definition
class GeoDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = row['label']
        return image, label, img_path

# label encoding
le = LabelEncoder()

train_df = pd.read_csv("train_dataset.csv")
val_df = pd.read_csv("val_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

train_df['label'] = le.fit_transform(train_df['country'])
val_df['label'] = le.transform(val_df['country'])
test_df['label'] = le.transform(test_df['country'])

# transforms for models that use torchvision preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# choose which model to use
MODEL_NAME = "StreetCLIP"   # options: "StreetCLIP", "PIGEON", "GeoEstimation"

if MODEL_NAME == "StreetCLIP":
    model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
elif MODEL_NAME in ["PIGEON", "GeoEstimation"]:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(DEVICE)
    model.fc = nn.Identity()
    processor = None
else:
    raise ValueError("Invalid MODEL_NAME")

model.eval()

# checkpoint saving
def save_embedding_checkpoint(emb_list, label_list, last_idx, path):
    """Save current embedding list to file."""
    if len(emb_list) == 0:
        return
    emb_tensor = torch.stack(emb_list)
    labels_tensor = torch.tensor(label_list)
    torch.save({
        'embeddings': emb_tensor,
        'labels': labels_tensor,
        'last_idx': last_idx
    }, path)
    print(f"Saved checkpoint at index {last_idx}")

def load_embedding_checkpoint(path):
    """Load checkpoint if file exists."""
    if os.path.exists(path):
        ckpt = torch.load(path)
        emb_list = [e for e in ckpt['embeddings']]
        labels_list = ckpt['labels'].tolist() if isinstance(ckpt['labels'], torch.Tensor) else ckpt['labels']
        last_idx = ckpt['last_idx']
        print(f"Loaded checkpoint from {path}")
        return emb_list, labels_list, last_idx
    return [], [], 0

def generate_embeddings(df, emb_save_path):
    """Generate embeddings with resume-on-interrupt logic."""
    emb_list, label_list, start_idx = load_embedding_checkpoint(emb_save_path)

    if start_idx >= len(df):
        print(f"Embeddings already complete for {emb_save_path}")
        return torch.stack(emb_list), torch.tensor(label_list)

    print(f"Resuming at index {start_idx} / {len(df)}")

    for idx in tqdm(range(start_idx, len(df)), initial=start_idx, total=len(df),
                    desc=f"{MODEL_NAME} Embeddings"):
        row = df.iloc[idx]
        img = Image.open(row['img_path']).convert("RGB")

        with torch.no_grad():
            if MODEL_NAME == "StreetCLIP":
                inputs = processor(images=img, return_tensors="pt").to(DEVICE)
                emb = model.get_image_features(**inputs).squeeze(0).cpu()
            else:
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                emb = model(img_tensor).squeeze(0).cpu()

        emb_list.append(emb)
        label_list.append(row['label'])

        if (idx + 1) % EMB_SAVE_INTERVAL == 0:
            save_embedding_checkpoint(emb_list, label_list, idx + 1, emb_save_path)

    save_embedding_checkpoint(emb_list, label_list, len(df), emb_save_path)
    return torch.stack(emb_list), torch.tensor(label_list)

# paths for checkpoint files
train_emb_path = os.path.join(CHECKPOINT_DIR, f"train_emb_{MODEL_NAME}.pt")
val_emb_path = os.path.join(CHECKPOINT_DIR, f"val_emb_{MODEL_NAME}.pt")
test_emb_path = os.path.join(CHECKPOINT_DIR, f"test_emb_{MODEL_NAME}.pt")

# generate embeddings
train_embeddings, train_labels = generate_embeddings(train_df, train_emb_path)
val_embeddings, val_labels = generate_embeddings(val_df, val_emb_path)
test_embeddings, test_labels = generate_embeddings(test_df, test_emb_path)

print(f"Finished generating {MODEL_NAME} embeddings.")
print(f"Train: {train_embeddings.shape}, Val: {val_embeddings.shape}, Test: {test_embeddings.shape}")
