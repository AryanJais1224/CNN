
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

print(f"Using device: {device}")

def generate_gaussian_data(n_samples=1000, img_size=91):

    x = np.linspace(-4, 4, img_size)
    X, Y = np.meshgrid(x, x)

    widths = np.linspace(2, 20, n_samples)
    images = torch.zeros(n_samples, 1, img_size, img_size)

    for i in range(n_samples):

        ro = 1.5 * np.random.randn(2)
        G = np.exp(-((X - ro[0])**2 + (Y - ro[1])**2) / widths[i])

        # add noise
        G += np.random.randn(img_size, img_size) / 5

        # add random bar
        i1 = np.random.choice(np.arange(2, 28))
        i2 = np.random.choice(np.arange(2, 6))

        if np.random.randn() > 0:
            G[i1:i1+i2, :] = 1
        else:
            G[:, i1:i1+i2] = 1

        images[i] = torch.tensor(G).view(1, img_size, img_size)

    return images.to(device)

class MyL1Loss(nn.Module):
    def forward(self, y_hat, y):
        return torch.mean(torch.abs(y_hat - y))


class MyL2AveLoss(nn.Module):
    def forward(self, y_hat, y):
        mse = torch.mean((y_hat - y)**2)
        mean_penalty = torch.abs(torch.mean(y_hat))
        return mse + mean_penalty


class MyCorLoss(nn.Module):
    def forward(self, y_hat, y):

        eps = 1e-8

        mean_x = torch.mean(y_hat)
        mean_y = torch.mean(y)

        num = torch.sum((y_hat - mean_x) * (y - mean_y))
        den = (torch.numel(y) - 1) * torch.std(y_hat) * torch.std(y) + eps

        return -num / den

class GaussianAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 6, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 1, 3, stride=2),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_model(images, epochs=1000, batch_size=32):

    model = GaussianAutoencoder().to(device)

    # Choose loss here:
    loss_fn = MyL1Loss()
    # loss_fn = MyL2AveLoss()
    # loss_fn = MyCorLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(epochs):

        idx = np.random.choice(len(images), size=batch_size, replace=False)
        X = images[idx]

        y_hat = model(X)
        loss = loss_fn(y_hat, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")

    return model, losses

def plot_loss(losses):
    plt.figure(figsize=(8,5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (Final={losses[-1]:.4f})")
    plt.show()


def visualize_results(model, images):

    model.eval()

    idx = np.random.choice(len(images), size=10, replace=False)
    X = images[idx]

    with torch.no_grad():
        y_hat = model(X)

    fig, axs = plt.subplots(2, 10, figsize=(18,4))

    for i in range(10):

        original = X[i,0].cpu()
        output = y_hat[i,0].cpu()

        axs[0,i].imshow(original, cmap="jet")
        axs[0,i].axis("off")
        axs[0,i].set_title(f"Input\nμ={original.mean():.2f}", fontsize=9)

        axs[1,i].imshow(output, cmap="jet")
        axs[1,i].axis("off")
        axs[1,i].set_title(f"Output\nμ={output.mean():.2f}", fontsize=9)

    plt.show()
