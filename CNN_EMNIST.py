
# 1. Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import copy
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm
import matplotlib.pyplot as plt

# 2. Device Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. Data Preparation
def load_and_prepare_data():

    dataset = torchvision.datasets.EMNIST(
        root="emnist",
        split="letters",
        download=True
    )

    # Convert to float tensor and reshape
    images = dataset.data.view(-1, 1, 28, 28).float()

    # Remove non-existing class (label 0) and relabel
    labels = copy.deepcopy(dataset.targets) - 1
    letter_categories = dataset.classes[1:]

    # Normalize
    images /= torch.max(images)

    return images, labels, letter_categories


def create_dataloaders(images, labels, batch_size=32):

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.1
    )

    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=len(test_data)
    )

    return train_loader, test_loader


# 4. CNN Model Definition
class EMNISTNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 6, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(6)

        # Classifier
        self.fc1 = nn.Linear(7 * 7 * 6, 50)
        self.fc2 = nn.Linear(50, 26)

    def forward(self, x):

        x = F.max_pool2d(self.conv1(x), 2)
        x = F.leaky_relu(self.bn1(x))

        x = F.max_pool2d(self.conv2(x), 2)
        x = F.leaky_relu(self.bn2(x))

        x = x.view(x.size(0), -1)

        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        return x


# 5. Training Function
def train_model(train_loader, test_loader, epochs=10):

    model = EMNISTNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss, test_loss = [], []
    train_err, test_err = [], []

    for epoch in range(epochs):

        model.train()
        batch_loss = []
        batch_error = []

        for X, y in train_loader:

            X, y = X.to(device), y.to(device)

            y_hat = model(X)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
            batch_error.append(
                torch.mean((torch.argmax(y_hat, dim=1) != y).float()).item()
            )

        train_loss.append(np.mean(batch_loss))
        train_err.append(100 * np.mean(batch_error))

        # Evaluation
        model.eval()
        X_test, y_test = next(iter(test_loader))
        X_test, y_test = X_test.to(device), y_test.to(device)

        with torch.no_grad():
            y_hat = model(X_test)
            loss = loss_fn(y_hat, y_test)

        test_loss.append(loss.item())
        test_err.append(
            100 * torch.mean((torch.argmax(y_hat, dim=1) != y_test).float()).item()
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss[-1]:.4f} | "
            f"Test Loss: {test_loss[-1]:.4f} | "
            f"Test Error: {test_err[-1]:.2f}%"
        )

    return model, train_loss, test_loss, train_err, test_err


# 6. Visualization
def plot_metrics(train_loss, test_loss, train_err, test_err):

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(train_loss, label="Train")
    ax[0].plot(test_loss, label="Test")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(train_err, label="Train")
    ax[1].plot(test_err, label="Test")
    ax[1].set_title("Error Rate (%)")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    plt.show()


def plot_confusion_matrix(model, test_loader, labels):

    model.eval()
    X_test, y_test = next(iter(test_loader))
    X_test = X_test.to(device)

    with torch.no_grad():
        y_hat = model(X_test)

    preds = torch.argmax(y_hat, dim=1).cpu()

    C = skm.confusion_matrix(
        y_test.cpu(),
        preds,
        normalize="true"
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(C, cmap="Blues", vmax=0.05)
    plt.xticks(range(26), labels=labels)
    plt.yticks(range(26), labels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
