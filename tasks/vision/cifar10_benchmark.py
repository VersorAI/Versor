import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys
import time
import json
from datetime import datetime

# Add parent directory to path to import Model
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Model.model import VersorTransformer

def train_cifar10():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    batch_size = 128
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model Setup
    # Using a slightly larger model for CIFAR-10
    embed_dim = 16 # 16 * 32 = 512 internal dims
    n_heads = 4
    n_layers = 4
    num_classes = 10
    
    class VisionVersorWrapper(nn.Module):
        def __init__(self, num_classes=10, embed_dim=16, n_heads=4, n_layers=2):
            super().__init__()
            self.patch_size = 4
            self.embed_dim = embed_dim
            self.input_dim = 3 * self.patch_size * self.patch_size
            self.patch_emb = nn.Linear(self.input_dim, embed_dim * 32)
            self.pos_emb = nn.Parameter(torch.randn(1, 64, embed_dim, 32) * 0.02)
            self.transformer = VersorTransformer(embed_dim, n_heads, n_layers, n_classes=num_classes, use_rotor_pool=False)
            
        def forward(self, x):
            B, C, H, W = x.shape
            # Patchify
            x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * self.patch_size * self.patch_size)
            # Embed and Add Position
            h = self.patch_emb(x).view(B, -1, self.embed_dim, 32)
            h = h + self.pos_emb
            logits = self.transformer(h)
            return logits

    model = VisionVersorWrapper(num_classes, embed_dim, n_heads, n_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    results = {
        "epochs": [],
        "train_loss": [],
        "test_acc": []
    }
    
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, data in enumerate(pbar):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (i + 1)})
            
        scheduler.step()
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy on test images: {accuracy:.2f}%')
        
        results["epochs"].append(epoch + 1)
        results["train_loss"].append(running_loss / len(trainloader))
        results["test_acc"].append(accuracy)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"results/cifar10_results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    train_cifar10()
