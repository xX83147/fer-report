import random
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# ========= 0. 固定随机种子 =========
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# ========= 1. 路径 =========
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR = BASE_DIR / "data" / "test"

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ========= 2. 超参数 =========
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)
print("CUDA available:", torch.cuda.is_available())


# ========= 3. 数据预处理 =========
# 这里改成 1 通道输入，并把尺寸从 48x48 放大到 96x96
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# ========= 4. 数据集 =========
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("类别顺序:", train_dataset.class_to_idx)
print("训练集大小:", len(train_dataset))
print("测试集大小:", len(test_dataset))


# ========= 5. ResNet18 baseline（改成 1 通道输入） =========
model = models.resnet18(weights=None)

# 把第一层卷积从 3 通道改成 1 通道
model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

# 把最后的全连接层改成 7 分类
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)


# ========= 6. 损失函数、优化器、调度器 =========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# ========= 7. 训练函数 =========
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ========= 8. 测试函数 =========
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ========= 9. 开始训练（含 early stopping） =========
history = []

best_acc = 0.0
patience = 5
no_improve = 0

best_acc = 0.0
patience = 5
no_improve = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Current LR: {current_lr:.6f}")

    history.append({
    "epoch": epoch + 1,
    "train_loss": train_loss,
    "train_acc": train_acc,
    "test_loss": test_loss,
    "test_acc": test_acc,
    "lr": current_lr,
})

    if test_acc > best_acc:
        best_acc = test_acc
        no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("已保存最佳模型到 best_model.pth")
    else:
        no_improve += 1
        print(f"验证集连续 {no_improve} 轮没有提升")

    if no_improve >= patience:
        print("Early stopping triggered.")
        break

csv_path = RESULTS_DIR / "resnet18_history.csv"

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr"]
    )
    writer.writeheader()
    writer.writerows(history)

print(f"训练历史已保存到: {csv_path}")

epochs = [x["epoch"] for x in history]
train_accs = [x["train_acc"] for x in history]
test_accs = [x["test_acc"] for x in history]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, test_accs, label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet18 Accuracy Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

fig_path = FIGURES_DIR / "resnet18_acc_curve.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"准确率曲线已保存到: {fig_path}")
print("训练完成，最佳测试准确率:", best_acc)