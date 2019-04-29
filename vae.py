import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch import autograd
from torch.utils.data import DataLoader
from torch import nn

import matplotlib.pyplot as plt
#%matplotlib inline
from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

from tqdm import tqdm
import os

EPOCH = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
# True表示需下载，False表示已下载
DOWNLOAD_MNIST = False

# 训练时才会标准化
im_tfs = tfs.Compose([
    # 先将输入归一化到(0,1)，再使用公式”(x-mean)/std”，将每个元素分布到(-1,1) 
    tfs.ToTensor(),
    tfs.Normalize([0.5], [0.5])
])

train_set = MNIST('./mnist', transform=im_tfs, train=True, download=DOWNLOAD_MNIST)
train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
print(len(train_data))

# test_set = MNIST('./mnist', train=False)

# print(train_set.data.size())
# print(train_set.data[0])
# print(train_set.targets.size())
# print(train_set.targets[0])

# for i, batch in enumerate(train_data):
#     print(i)
#     # batch[0]为数据,batch[1]为标签
#     print(batch[0], batch[1])

# plt.imshow(train_set.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_set.targets[0])
# plt.show()

class VAE(nn.Module):
    def __init__(self, latent_num=2):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(28*28, 400)
        self.fc21 = nn.Linear(400, 20) # mean
        self.fc22 = nn.Linear(400, 20) # var
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 28*28)

    # q(z|x)
    def encode_q(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = nn.BatchNorm1d(h1)
        h1 = F.dropout(h1, p=0.1)
        return self.fc21(h1), self.fc22(h1)
    
    # 重参数化，使网络可以反向传播
    def reparametrize(self, mu, logvar):
        # mul逐乘
        # exp逐指数
        # exp_是exp的in-place形式
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        if torch.cuda.is_available():
            eps = eps.cuda()
        # z = mu + sigma * eps
        return eps.mul(std).add_(mu)
    
    # p(x|z)
    def decoder_p(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = nn.BatchNorm1d(h3)
        h3 = F.dropout(h3, p=0.1)
        # "nn.functional.tanh is deprecated. Use torch.tanh instead."
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode_q(x)
        z = self.reparametrize(mu, logvar)
        # 解码，同时输出均值和方差
        return self.decoder_p(z), mu, logvar 


net = VAE()
print(net)
if torch.cuda.is_available():
    net = net.cuda()

reconstruction_function = nn.MSELoss(reduction='sum')

def loss_function(recon_x, x, mu, logvar):
    MSE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD

optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
# 添加正则项，从而替代dropout
# optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) 

def to_img(x):
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0],1,28,28)
    return x

for e in tqdm(range(EPOCH)):
    # if e == 20:
    #     set_learning_rate(optimizer, 0.01) # 80 次修改学习率为 0.01
    for i, batch in enumerate(train_data):
        # print(i)
        # batch[0]为数据,batch[1]为标签
        print(batch[0], batch[1])
        print(batch[0].shape)
#         if torch.cuda.is_available():
#             batch = batch.cuda()
        img = batch[0].view(BATCH_SIZE, -1)
        #print(img.size())
        recon_img, mu, logvar = net(img)
        loss = loss_function(recon_img, img, mu, logvar) / BATCH_SIZE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (e + 1) % 5 == 0:
        # 测验模式，不使用dropout
        net.eval()

        print('epoch: {}, Loss: {:.4f}'.format(e + 1, loss.data))
    
        # 返回train模式
        net.train()

def save_model():
    # entire net
    torch.save(net, 'VAE_net1.pkl')
    # parameters
    torch.save(net.state_dict(), 'VAE_net1_params.pkl')
    
def restore_net():
    net2 = torch.load('VAE_net1.pkl')
    net2.eval()
    
def restore_params():
    # 要与原net的结构一样
    net3 = VAE(*args, **kwargs)
    net3.load_state_dict(torch.load('VAE_net1_params.pkl'))
    net3.eval()
    
# save_model()