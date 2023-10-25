import torch
import os
import glob
import cv2
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import v2
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim


CLASSES = ['up', 'down']

class WeightDataset(Dataset):
    def __init__(self, raw_data):
        self.X = raw_data[:, 0]
        self.Y = raw_data[:, 1].astype(int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        im = cv2.imread(self.X[idx], 0)/255.0
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if im.shape[0] != im.shape[1]:
            crop = (im.shape[1]-im.shape[0])//2
            im = im[:, crop:-crop]
        # flip horizontally
        if np.random.randint(10) >= 1:
            im = cv2.flip(im, 1)
        im = torch.from_numpy(im)[None, :, :]
        return im, torch.tensor([self.Y[idx]])


class ConvNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 6, 5)
        # self.fc1 = nn.Linear(6 * 5 * 5, 120)
        self.fc1 = nn.Linear(10584, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


def run():
    bsz = 16
    lr = 0.001
    epochs = 60
    
    raw_data = []
    for i, label in enumerate(CLASSES):
        paths = glob.glob(f'./imgs/{label}/*')
        raw_data.extend([[path, i] for path in paths])
        print(f'{label}: {len(paths)}')
    raw_data = np.array(raw_data)

    train_idx = np.random.choice(np.arange(raw_data.shape[0]), int(raw_data.shape[0]*0.8), replace=False)
    train_data = raw_data[train_idx]
    val_data = raw_data[np.delete(np.arange(raw_data.shape[0]), train_idx)]
    train_dataset = WeightDataset(train_data)
    val_dataset = WeightDataset(val_data)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=True)
    print('train:', len(train_dataset))
    print('validation:', len(val_dataset))


    im = next(iter(train_dl))[0][0]
    model = ConvNet(len(CLASSES))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_dl):
            x, y = data
            x = x.float()
            y = torch.squeeze(F.one_hot(y, len(CLASSES)), dim=1).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_dataset)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        for i, data in enumerate(val_dl):
            x, y = data
            y = torch.squeeze(F.one_hot(y, len(CLASSES)), dim=1).float()
            y_pred = model(x.float())
            loss = criterion(y_pred, y)
            running_loss += loss.item()
        val_loss = running_loss / len(val_dataset)
        val_losses.append(val_loss)
        print(f'[epoch {epoch+1}] t loss: {np.round(train_loss, 3)}, v loss: {np.round(val_loss, 3)}')
        
        if val_loss < best_loss:
            torch.save(model.state_dict(), './cnn_model.pth')
            best_loss = val_loss

    # final prediction accuracy
    hits = 0
    for i, data in enumerate(val_dl):
        x, y = data
        y_pred = model(x.float())
        hits += torch.sum(torch.squeeze(y) == torch.argmax(y_pred, dim=1)).item()
    print('final v accuracy:', np.round(hits/len(val_dataset), 4))

    # plot model learning
    plot_path = './learning_plot.png'
    try:
        os.remove(plot_path)
    except:
        pass
    plt.plot(range(epochs), train_losses, label='training')
    plt.plot(val_losses, label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(plot_path)
    plt.clf()
    
    # # show example with prediction
    # im = next(iter(train_dl))[0][0][0]
    # plt.imshow(im, cmap='gray')
    # pred = model(im[None,None,:,:].float())
    # plt.title(CLASSES[torch.argmax(pred).item()])
    # plt.show()


if __name__ == '__main__':
    run()