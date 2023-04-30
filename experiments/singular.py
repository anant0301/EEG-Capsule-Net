# %% [markdown]
# # Data Loaders
# 
# Start GPU Runtime for this, to speed up the above.

# %%
USE_CUDA = True

# %%
# ! pip install torch torchvision pandas tqdm scipy

# %%
import os
import numpy as np
import pandas as pd
import scipy
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, Adagrad
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# %% [markdown]
# Data Transformation

# %%
import cv2
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt

def plot_spectrogram(eeg_arr, output_dim= (28,28), sr= 200):
    freqs, times, spectrogram_x = ss.spectrogram(eeg_arr, sr)
    image =  plt.pcolormesh(times, freqs, spectrogram_x, shading='gouraud').get_array().reshape(129,3)
    res = cv2.resize(image, dsize=output_dim)
    return res


# %%
root_dir = '/home/patelanant/CapsuleNet/data/capsulenet/time/'
train_csv = '/home/patelanant/CapsuleNet/data/capsulenet/time/train-subject-1.csv'
test_csv = '/home/patelanant/CapsuleNet/data/capsulenet/time/test-subject-1.csv'

# %%
class SEEDDataset(Dataset):
    """SEED EEG dataset."""

    def __init__(self, csv_file, root_dir, transform=None, ch_idx= 0):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.channel_idx = ch_idx

    def __len__(self):
        return len(self.df['outfile_name'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            samples = []
            targets = []
            for id in idx:
                arr = np.load(os.path.join(self.root_dir, self.df['outfile_name'].iloc[idx]), allow_pickle= True)
                arr = self.transform(arr)
                samples.append(arr)
                target = self.df['label'].iloc[id]
                targets.append(target + 1)
            samples = np.array(samples)
            targets = np.array(targets)
            return samples.astype(np.float32), (targets+1).astype(np.float32)
        else:
            arr = np.load(os.path.join(self.root_dir, self.df['outfile_name'].iloc[idx]), allow_pickle= True)
            arr = self.transform(arr[self.channel_idx])
            arr = arr.reshape(1, 28,28).astype(np.float32)
            return arr, np.array(self.df['label'][idx] + 1).astype(int)

# %%
class SEEDLoader(torch.utils.data.DataLoader):
  def __init__(self, batch_size= 100, ch_idx= 0, transform= None):
      self.train_dataset = SEEDDataset(train_csv, root_dir, ch_idx= ch_idx, transform= transform)
      self.test_dataset = SEEDDataset(test_csv, root_dir, ch_idx= ch_idx, transform= transform)
      self.train_loader  = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
      self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

# %% [markdown]
# # Capsule Network Model

# %%
class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1
                             )

    def forward(self, x):
        a=  F.relu(self.conv(x))
        return a

# %%
class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9):
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0) 
                          for _ in range(num_capsules)])
    
    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), 32 * 6 * 6, -1)
        return self.squash(u)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

# %%
class DigitCaps(nn.Module):
    def __init__(self, num_capsules=3, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 1
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

# %%
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
        
    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        # classes = F.softmax(classes, dim= 3)
        classes = F.softmax(classes)
        
        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(3))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        
        reconstructions = self.reconstraction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 1, 28, 28)
        
        return reconstructions, masked

# %%
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps(
            num_capsules= 8,
            in_channels= 256,
            out_channels= 32,
            kernel_size= 9
            )
        self.digit_capsules = DigitCaps()
        self.decoder = Decoder()
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        # print("label-output", output)
        reconstructions, masked = self.decoder(output, data)
        return output, reconstructions, masked
    
    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target)# + self.reconstruction_loss(data, reconstructions)
    
    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x**2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        # print("left", left, left.shape)
        # print("right", right, right.shape)
        # print("labels", labels, labels.shape)
        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()
        return loss
    
    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        # print("reconstruction_loss", loss, loss.shape)
        return loss

# %% [markdown]
# # Training and Testing the model

# %%
from sklearn.metrics import confusion_matrix
capsule_net = CapsNet()
if USE_CUDA:
    capsule_net = capsule_net.cuda()
optimizer = Adam(capsule_net.parameters())

    # %%
for i in range(64):
    batch_size = 100
    seed = SEEDLoader(batch_size, ch_idx= i,transform= plot_spectrogram)
    nclasses = 3
    n_epochs = 30

    # %%
    layers = [layer for layer in capsule_net.modules()]
    for epoch in tqdm.tqdm(range(n_epochs)):
        capsule_net.train()
        train_loss = 0
    #     print("epoch:", epoch)
        for batch_id, (data, target) in enumerate(seed.train_loader):
            target = torch.sparse.torch.eye(3).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output, reconstructions, masked = capsule_net(data)
            loss = capsule_net.loss(data, output, target, reconstructions)
            loss.backward()
            optimizer.step()
            train_loss += loss.data
        print ("train accuracy max:", sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size))
        print ("train_loss:", train_loss)
        print ("train_loss/datalen", train_loss / len(seed.train_loader))
        traincm = confusion_matrix(np.argmax(masked.data.cpu().numpy(), 1), np.argmax(target.data.cpu().numpy(), 1))
        pd.DataFrame(traincm,
                     index = ['Negative','Neutral','Positive'], 
                     columns = ['Negative','Neutral','Positive']).to_csv("./cms/train-%d-ch%d.csv" % (epoch, i))
        capsule_net.eval()
        test_loss = 0
        for batch_id, (data, target) in enumerate(seed.test_loader):

            target = torch.sparse.torch.eye(3).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()
            output, reconstructions, masked = capsule_net(data)
            # print("test output", output)
            loss = capsule_net.loss(data, output, target, reconstructions)
            test_loss += loss.data
        print ("test accuracy max:", sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size))
        print ("test_loss:", test_loss)
        print ("test_loss/datalen", test_loss / len(seed.test_loader))
        testcm = confusion_matrix(np.argmax(masked.data.cpu().numpy(), 1), np.argmax(target.data.cpu().numpy(), 1))
        pd.DataFrame(testcm,
                     index = ['Negative','Neutral','Positive'], 
                     columns = ['Negative','Neutral','Positive']).to_csv("./cms/test-%d-ch%d.csv" % (epoch, i))

# %%
torch.save(capsulenet.state_dict(), "model-3class-sub1")

# %%

