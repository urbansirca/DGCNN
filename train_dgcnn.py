import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
from torcheeg.models import DGCNN
from torcheeg.model_selection import KFoldPerSubject

# import scheduler
from torch.optim.lr_scheduler import StepLR

# Config
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

dataset = SEEDDataset(
    root_path='/Users/urbansirca/datasets/SEED/Preprocessed_EEG',
    io_path='/Users/urbansirca/Desktop/FAX/Master\'s AI/MLGraphs/DGCNN/.torcheeg/datasets_1768912535105_zLBYu',
    offline_transform=transforms.BandDifferentialEntropy(band_dict={
        "delta": [1, 4],
        "theta": [4, 8],
        "alpha": [8, 14],
        "beta": [14, 31],
        "gamma": [31, 49]
    }),
    online_transform=transforms.ToTensor(),
    label_transform=transforms.Compose([
        transforms.Select('emotion'),
        transforms.Lambda(lambda x: x + 1)  # SEED labels: -1,0,1 -> 0,1,2
    ])
)

# Split dataset (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))





# print sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# print per class
train_labels = [label for _, label in train_dataset]
test_labels = [label for _, label in test_dataset]
print(f"Train class distribution: {torch.bincount(torch.tensor(train_labels))}")
print(f"Test class distribution: {torch.bincount(torch.tensor(test_labels))}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = DGCNN(
    in_channels=5, num_electrodes=62, hid_channels=32, num_layers=2, num_classes=3
)
model = model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

batch_losses = []
train_accuracies = []
test_accuracies = []
# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        
        batch_losses.append(loss.item())
        train_accuracies.append(100. * correct / total)
        


    train_acc = 100. * correct / total
    

    # Evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            _, predicted = outputs.max(1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()

    test_acc = 100. * test_correct / test_total

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

print("Training complete!")

# save model
torch.save(model.state_dict(), 'ckpts/dgcnn_seed_model.pth')
print("Model saved!")
