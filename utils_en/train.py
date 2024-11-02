import os
import csv
import torch
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from dataloader import ModelNet40Dataset
from pointnet import PointNet

def train():

    if not os.path.exists('../model'):
        os.makedirs('../model')

    best_acc = 0
    batch_size = 128
    epochs = 50
    npf = 64

    print("Training Start")
    train_dataset = ModelNet40Dataset(data_dir='../data/modelnet40_hdf5_2048', split='train')
    test_dataset = ModelNet40Dataset(data_dir='../data/modelnet40_hdf5_2048', split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False, pin_memory=True)
    print("Finish Loading Data")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    model = PointNet(npf).to(device)
    print("Finish Loading Model")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_csv = '../train.csv'
    with open(train_csv, 'w', newline="") as f:
        fieldnames = ['Epoch', 'Train_loss']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print("Start Training")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for points, labels in train_loader:

            points = points.to(device, dtype=torch.float).transpose(2, 1)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(points)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        with open(train_csv, 'a', newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'Epoch': epoch + 1, 'Train_loss': train_loss,})

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for points, labels in test_loader:
                points = points.to(device, dtype=torch.float).transpose(2, 1)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(points)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total

        print(f"Epoch: {epoch+1}, Loss: {train_loss:.2f}, Accuracy: {test_acc*100:.2f}%, Time: {time.time() - start_time:.2f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), '../model/model.pt')
            print(f'Best model saved with accuracy: {test_acc * 100:.2f}')
            # Save the model with the highest accuracy on the test set

if __name__ == '__main__':
    train()
