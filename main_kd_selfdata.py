import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision import transforms, models, datasets


# =========================
# 1. 基本配置
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 7
IMAGE_SIZE = 96
BATCH_SIZE = 64
SELF_BATCH_SIZE = 4
NUM_EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4

TEMPERATURE = 4.0
LAMBDA_KD = 0.7
LAMBDA_SELF = 0.3

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
SELF_DATA_DIR = "data/selfdata"

TEACHER_TYPE = "resnet18"          # 可改成 "resnet50"
TEACHER_CKPT = "resnet18_best_model.pth"   # 如果换teacher，这里也改
SAVE_PATH = "mobilenetv3_self_kd_best.pth"


# =========================
# 2. self data 数据集
# =========================
class SelfImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.exists(root_dir):
            raise ValueError(f"self data 路径不存在: {root_dir}")

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.splitext(f.lower())[1] in exts
        ]
        self.image_paths.sort()

        if len(self.image_paths) == 0:
            raise ValueError(f"在 {root_dir} 里没有找到图片文件。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")   # 灰度，与FER2013对齐

        if self.transform is not None:
            img = self.transform(img)

        return img


# =========================
# 3. teacher / student 模型
#    都返回 logits, feat
# =========================
class ResNetTeacher(nn.Module):
    def __init__(self, num_classes=7, backbone="resnet18"):
        super().__init__()

        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=None)
            feat_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=None)
            feat_dim = 2048
        else:
            raise ValueError("backbone 只能是 resnet18 或 resnet50")

        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        self.backbone.fc = nn.Linear(feat_dim, num_classes)
        self.feat_dim = feat_dim

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.backbone.fc(feat)
        return logits, feat


class MobileNetV3Student(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.backbone = models.mobilenet_v3_small(weights=None)

        first_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False
        )

        self.feat_dim = 576
        self.backbone.classifier[3] = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.backbone.classifier(feat)
        return logits, feat


# =========================
# 4. feature 对齐层
# =========================
class FeatureProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)


# =========================
# 5. 损失函数
# =========================
def kd_loss_fn(student_logits, teacher_logits, T=4.0):
    s_log_prob = F.log_softmax(student_logits / T, dim=1)
    t_prob = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T * T)


# =========================
# 6. 测试函数
# =========================
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, _ = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# =========================
# 7. 训练函数
# =========================
def train_student_with_self_kd(
    teacher,
    student,
    projector,
    train_loader,
    test_loader,
    self_loader,
    num_epochs=20,
):
    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    best_acc = 0.0

    for epoch in range(num_epochs):
        teacher.eval()
        student.train()
        projector.train()

        total_loss = 0.0
        total_ce = 0.0
        total_kd = 0.0
        total_self = 0.0
        correct = 0
        total = 0

        self_iter = iter(self_loader)

        for images_pub, labels_pub in train_loader:
            images_pub = images_pub.to(DEVICE)
            labels_pub = labels_pub.to(DEVICE)

            try:
                images_self = next(self_iter)
            except StopIteration:
                self_iter = iter(self_loader)
                images_self = next(self_iter)

            images_self = images_self.to(DEVICE)

            with torch.no_grad():
                t_logits_pub, _ = teacher(images_pub)
                _, t_feat_self = teacher(images_self)

            s_logits_pub, _ = student(images_pub)
            _, s_feat_self = student(images_self)

            s_feat_self_proj = projector(s_feat_self)

            loss_ce = ce_criterion(s_logits_pub, labels_pub)
            loss_kd = kd_loss_fn(s_logits_pub, t_logits_pub, T=TEMPERATURE)
            loss_self = mse_criterion(s_feat_self_proj, t_feat_self)

            loss = loss_ce + LAMBDA_KD * loss_kd + LAMBDA_SELF * loss_self

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images_pub.size(0)
            total_ce += loss_ce.item() * images_pub.size(0)
            total_kd += loss_kd.item() * images_pub.size(0)
            total_self += loss_self.item() * images_pub.size(0)

            preds = torch.argmax(s_logits_pub, dim=1)
            correct += (preds == labels_pub).sum().item()
            total += labels_pub.size(0)

        train_loss = total_loss / total
        train_ce = total_ce / total
        train_kd = total_kd / total
        train_self = total_self / total
        train_acc = correct / total

        test_loss, test_acc = evaluate(student, test_loader, ce_criterion)
        scheduler.step(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  CE   : {train_ce:.4f}")
        print(f"  KD   : {train_kd:.4f}")
        print(f"  SELF : {train_self:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {
                "student": copy.deepcopy(student.state_dict()),
                "projector": copy.deepcopy(projector.state_dict()),
                "best_acc": best_acc,
                "epoch": epoch + 1,
            }
            torch.save(best_state, SAVE_PATH)
            print(f"已保存最佳模型到 {SAVE_PATH}")

    print(f"\n训练结束，最佳 Test Acc: {best_acc:.4f}")


# =========================
# 8. 主函数
# =========================
def main():
    print("Using device:", DEVICE)

    if not os.path.exists(TRAIN_DIR):
        raise ValueError(f"TRAIN_DIR 不存在: {TRAIN_DIR}")
    if not os.path.exists(TEST_DIR):
        raise ValueError(f"TEST_DIR 不存在: {TEST_DIR}")
    if not os.path.exists(SELF_DATA_DIR):
        raise ValueError(f"SELF_DATA_DIR 不存在: {SELF_DATA_DIR}")
    if not os.path.exists(TEACHER_CKPT):
        raise ValueError(f"teacher 权重不存在: {TEACHER_CKPT}")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    self_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    self_dataset = SelfImageDataset(SELF_DATA_DIR, transform=self_transform)
    self_loader = DataLoader(
        self_dataset,
        batch_size=SELF_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    print(f"train set 数量: {len(train_dataset)}")
    print(f"test set 数量: {len(test_dataset)}")
    print(f"self data 图片数量: {len(self_dataset)}")

    teacher = ResNetTeacher(num_classes=NUM_CLASSES, backbone=TEACHER_TYPE).to(DEVICE)
    teacher_ckpt = torch.load(TEACHER_CKPT, map_location=DEVICE)

    if isinstance(teacher_ckpt, dict) and "model_state_dict" in teacher_ckpt:
        state_dict = teacher_ckpt["model_state_dict"]
    elif isinstance(teacher_ckpt, dict) and "state_dict" in teacher_ckpt:
        state_dict = teacher_ckpt["state_dict"]
    elif isinstance(teacher_ckpt, dict) and "student" in teacher_ckpt:
        raise ValueError("你传进来的不是 teacher 权重，而是 student 权重。")
    else:
        state_dict = teacher_ckpt

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            new_state_dict[k] = v
        else:
            new_state_dict["backbone." + k] = v

    if "backbone.fc.weight" in new_state_dict and teacher.backbone.fc.weight.shape != new_state_dict["backbone.fc.weight"].shape:
        print("检测到 fc 层维度不一致，跳过 fc 权重加载。")
        new_state_dict.pop("backbone.fc.weight", None)
        new_state_dict.pop("backbone.fc.bias", None)

    missing, unexpected = teacher.load_state_dict(new_state_dict, strict=False)

    print("Teacher 权重加载完成。")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = MobileNetV3Student(num_classes=NUM_CLASSES).to(DEVICE)
    projector = FeatureProjector(student.feat_dim, teacher.feat_dim).to(DEVICE)

    train_student_with_self_kd(
        teacher=teacher,
        student=student,
        projector=projector,
        train_loader=train_loader,
        test_loader=test_loader,
        self_loader=self_loader,
        num_epochs=NUM_EPOCHS
    )


if __name__ == "__main__":
    main()