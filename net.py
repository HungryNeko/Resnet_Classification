import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,random_split, Subset
from tqdm import tqdm

from dataset import EuroSATMSDataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #root = 'dataset'
    split_dir = 'splits'

    # 变换：训练和测试可分别定义
    train_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # 分别实例化两个 ImageFolder（以便不同 transform）
    full_train = datasets.ImageFolder(root=root, transform=train_tf)
    full_test = datasets.ImageFolder(root=root, transform=test_tf)

    # 读取固定索引
    with open(f'{split_dir}/train_indices.json', 'r') as f:
        train_indices = json.load(f)
    with open(f'{split_dir}/test_indices.json', 'r') as f:
        test_indices = json.load(f)
    #RGB
    # train_dataset = Subset(full_train, train_indices)
    # test_dataset = Subset(full_test, test_indices)

    #TIF
    train_dataset = EuroSATMSDataset(root=root, indices=train_indices)
    test_dataset = EuroSATMSDataset(root=root, indices=test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        epoch_loop.set_postfix(loss=running_loss, acc=100 * correct / total)
        print(f"Train Accuracy: {100 * correct / total:.2f}%")
        torch.save(model.state_dict(), pthname+".pth")

        if (epoch+1)%5==0:
            #print('testing')
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

            print(f"Test Accuracy: {100 * correct / total:.2f}%")
if __name__ == '__main__':
    root='EuroSAT_MS'
    pthname='resnet18_10classes_10epoch_standard'
    total_epoch=10
    channel=13
    classes=10
    train()
    