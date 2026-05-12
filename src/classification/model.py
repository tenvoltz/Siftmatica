import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from PIL import Image
import random
import numpy as np
import os
import json

class EdgeRandomizer(object):
    def __init__(self, center_ratio=0.8, source_dataset=None):
        self.center_ratio = center_ratio
        self.source_dataset = source_dataset

    def __call__(self, img):
        img_np = np.array(img)
        h, w, c = img_np.shape

        center_h_start = int(h * (1 - self.center_ratio) / 2)
        center_h_end = int(h * (1 + self.center_ratio) / 2)
        center_w_start = int(w * (1 - self.center_ratio) / 2)
        center_w_end = int(w * (1 + self.center_ratio) / 2)

        center_mask = np.zeros((h, w), dtype=bool)
        center_mask[center_h_start:center_h_end, center_w_start:center_w_end] = True
        edge_mask = ~center_mask

        modified_img_np = np.copy(img_np)

        if self.source_dataset is not None and len(self.source_dataset) > 0:
            random_source_idx = random.randint(0, len(self.source_dataset) - 1)
            source_img_pil, _ = self.source_dataset[random_source_idx]
            source_img_np = np.array(source_img_pil)

            source_pixels = source_img_np.reshape(-1, 3)

            for r, col in np.argwhere(edge_mask):
                idx = random.randint(0, len(source_pixels) - 1)
                modified_img_np[r, col] = source_pixels[idx]
        else:
            center_pixels_coords = np.argwhere(center_mask)
            if len(center_pixels_coords) == 0:
                return img
            for r, col in np.argwhere(edge_mask):
                rand_center_r, rand_center_col = random.choice(center_pixels_coords)
                modified_img_np[r, col] = img_np[rand_center_r, rand_center_col]

        return Image.fromarray(modified_img_np)

class MinecraftTextureClassifier(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MinecraftDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = []
        self._load_data()

    def _load_data(self):
        image_filenames = [f for f in os.listdir(self.root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        class_names = sorted(list(set([os.path.splitext(f)[0] for f in image_filenames])))
        self.idx_to_class = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        for filename in image_filenames:
            path = os.path.join(self.root_dir, filename)
            class_name = os.path.splitext(filename)[0]
            label = self.class_to_idx[class_name]

            self.image_paths.append(path)
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
            image = background
        else:
            image = image.convert('RGB')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class CustomSubset(torch.utils.data.Dataset):
    def __init__(self, subset_dataset, transform=None):
        self.subset_dataset = subset_dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset_dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset_dataset)
    

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    print("Finished Training")

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    base_data = MinecraftDataset(root_dir='./data/block', transform=None)

    train_transform = transforms.v2.Compose([
        transforms.Resize((16, 16)),
        EdgeRandomizer(center_ratio=0.8, source_dataset=base_data),
        transforms.ToTensor(),
        transforms.v2.GaussianNoise(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((16, 16)),
        EdgeRandomizer(center_ratio=0.8, source_dataset=base_data),
        transforms.ToTensor(),
        transforms.v2.GaussianNoise(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    base_data = MinecraftDataset(root_dir='./data/block/', transform=None)

    test_data = MinecraftDataset(root_dir='./data/block/', transform=None)
    test_data.image_paths = base_data.image_paths * 2
    test_data.labels = base_data.labels * 2
    test_dataset = CustomSubset(test_data, transform=test_transform)

    train_data = MinecraftDataset(root_dir='./data/block/', transform=None)
    train_data.image_paths = base_data.image_paths * 18
    train_data.labels = base_data.labels * 18
    train_dataset = CustomSubset(train_data, transform=train_transform)


    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    NUM_CLASSES = len(test_data.class_to_idx)
    BATCH_SIZE = 32

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinecraftTextureClassifier(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=25)
    print("Starting evaluation...")
    evaluate_model(model, test_loader)

    model_path = './src/classification/minecraft_texture_classifier.pth'
    class_path = './src/classification/minecraft_class_names.json'

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    with open(class_path, 'w') as f:
        json.dump(test_data.idx_to_class, f)
    print(f"Class names saved to {model_path}")