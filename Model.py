import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np


class VAE(nn.Module):
    def __init__(self,device,T=1):
        super(VAE, self).__init__()

        self.device=device
        self.embed_dim=120
        self.dropout=nn.Dropout(0.5)
        self.T=T
        # Adjusting the input and output dimensions
        input_dim = 3549  # 1183 atoms * 3 coordinates

        # Adjusting the architecture to include one-hot encoding for labels
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048,2048)
        self.fc3=nn.Linear(2048,1024)
        self.fc4=nn.Linear(1024,1024)
        self.fc5=nn.Linear(1024,512)
        self.fc6=nn.Linear(512,512)
        self.fc7=nn.Linear(512,256)
        self.fc8=nn.Linear(256,256)
        self.fc9=nn.Linear(256,128)
        # self.fc6=nn.Linear(128,64)
        # self.fc7=nn.Linear(64,32)
        # self.fc8=nn.Linear(32,16)
        # self.fc9=nn.Linear(16,8)
        # self.fc10=nn.Linear(8,4)

        self.fc10_1 = nn.Linear(128, self.embed_dim)
        self.fc10_2 = nn.Linear(128, self.embed_dim)

        self.fc11 = nn.Linear(self.embed_dim, 128)
        # self.fc13 = nn.Linear(8, 16)
        # self.fc14 = nn.Linear(16, 32)
        # self.fc15 = nn.Linear(32, 64)
        # self.fc16 = nn.Linear(64, 128)
        self.fc12=nn.Linear(128,128)
        self.fc13 = nn.Linear(128, 256)
        self.fc14=nn.Linear(256,256)
        self.fc15 = nn.Linear(256, 512)
        self.fc16=nn.Linear(512,512)
        self.fc17 = nn.Linear(512, 1024)
        self.fc18=nn.Linear(1024,1024)
        self.fc19 = nn.Linear(1024, 2048)
        self.fc20=nn.Linear(2048,2048)
        self.fc21 = nn.Linear(2048, input_dim)


        # self.apply(self._init_weights)


    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         # module.weight.data.normal_(mean=0.0, std=1.0)
    #         #module.weight.data.fill_(1.0)
    #         if module.bias is not None:
    #             module.bias.data.normal_(mean=0.0, std=1.0)

    def to_one_hot(self, y):
        # Convert labels to one-hot encoding
        y = y.view(-1, 1).cpu().numpy()  # Convert to numpy array
        y_one_hot = self.encoder.transform(y)
        return torch.FloatTensor(y_one_hot).to(self.device)

    def encode(self, x):
        h1 = self.fc1(x)
        h1=  self.fc2(h1)
        h1 = F.relu(self.fc3(h1))
        h1 = self.fc4(h1)
        h1 = self.fc5(h1)
        h1 = F.relu(self.fc6(h1))
        h1 = self.fc7(h1)
        h1 = self.fc8(h1)
        h1 = F.relu(self.fc9(h1))
        return self.fc10_1(h1), self.fc10_2(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)*self.T
        return eps.mul(std).add_(mu)

    def set_temperature(self, T):
        self.T = T
        print(f"set_temperature {T}")



    def decode(self, z):
        h1=self.fc11(z)
        h1 = self.fc12(h1)
        h1 = self.fc13(h1)
        h1 = F.relu(self.fc14(h1))
        h1 = self.fc15(h1)
        h1 = self.fc16(h1)
        h1 = F.relu(self.fc17(h1))
        h1 = self.fc18(h1)
        h1 = self.fc19(h1)
        h1 = F.relu(self.fc20(h1))
        # h1=self.dropout(h1)
        h1 = self.fc21(h1)
        # return torch.sigmoid(h1)
        return h1

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 3549))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar,z