import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from dataset import EuroSATMSDataset


def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    kl_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / T, dim=1),
        nn.functional.softmax(teacher_logits / T, dim=1)
    )
    return alpha * ce_loss + (1 - alpha) * kl_loss * T * T


def spectral_jitter(tensor, factor_range=(0.9, 1.1)):
    factor = random.uniform(*factor_range)
    return tensor * factor

def train_transform(tensor):
    tensor = transforms.functional.resize(tensor, (32, 32))
    if random.random() > 0.5:
        tensor = transforms.functional.hflip(tensor)
    if random.random() > 0.5:
        tensor = transforms.functional.vflip(tensor)
    tensor = spectral_jitter(tensor)
    return tensor

def test_transform(tensor):
    return transforms.functional.resize(tensor, (32, 32))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = 'splits'


    with open(f'{split_dir}/train_indices.json', 'r') as f:
        train_indices = json.load(f)
    with open(f'{split_dir}/test_indices.json', 'r') as f:
        test_indices = json.load(f)


    train_dataset = EuroSATMSDataset(root=root, indices=train_indices, transform=train_transform)
    test_dataset = EuroSATMSDataset(root=root, indices=test_indices, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # student resnet18
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(channel, 32, kernel_size=3, stride=1, padding=1, bias=False)
    model.bn1 = nn.BatchNorm2d(32)
    model.layer1[0].conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.layer1[0].bn1 = nn.BatchNorm2d(64)
    model.layer1[0].downsample = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(64)
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, classes)
    model = model.to(device)

    # teacher resnet50
    teacher = models.resnet50(weights=None)
    teacher.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    teacher.bn1 = nn.BatchNorm2d(64)
    teacher.fc = nn.Linear(teacher.fc.in_features, classes)
    teacher.load_state_dict(torch.load(teacher_path))
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0

    print("training")
    epoch_loop = tqdm(range(total_epoch), desc="Training Progress")
    for epoch in epoch_loop:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{total_epoch}]", leave=False)
        for images, labels in batch_loop:
            images, labels = images.to(device), labels.to(device)

            student_logits = model(images)
            with torch.no_grad():
                teacher_logits = teacher(images)

            loss = distillation_loss(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(student_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        scheduler.step()
        epoch_loop.set_postfix(loss=running_loss, acc=100 * correct / total)
        print(f"Train Accuracy: {100 * correct / total:.2f}%")

        model.eval()
        correct = 0
        total = 0
        test_loop = tqdm(test_loader, desc="Testing", leave=False)
        with torch.no_grad():
            for images, labels in test_loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loop.set_postfix(acc=100 * correct / total)

        test_acc = 100 * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), pthname + "_best.pth")
            # print(f"Saved best model with Test Accuracy: {best_acc:.2f}%")

        torch.save(model.state_dict(), pthname + f"_epoch{epoch+1}.pth")


if __name__ == '__main__':
    root='EuroSAT_MS'
    pthname='resnet18_10classes_10epoch_student'
    total_epoch=10
    channel=13
    classes=10
    teacher_path='resnet50_teacher_10classes_20epoch_epoch20.pth'
    train()
