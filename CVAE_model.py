import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class CVAE(nn.Module):
    def __init__(self, num_classes,device,T=1):
        super(CVAE, self).__init__()

        # Assuming the range of classes for RMSD is known (num_classes)
        self.num_classes = num_classes
        self.device=device
        self.embed_dim=120
        self.dropout=nn.Dropout(0.5)
        self.T=T
        # Adjusting the input and output dimensions
        input_dim = 3549  # 1183 atoms * 3 coordinates

        # Adjusting the architecture to include one-hot encoding for labels
        self.fc1 = nn.Linear(input_dim + num_classes, 2048)
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

        self.fc11 = nn.Linear(self.embed_dim + num_classes, 128)
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
        self.fc22=nn.Linear(input_dim,input_dim)
        self.fc23 = nn.Linear(input_dim, input_dim)
        self.fc24 = nn.Linear(input_dim, input_dim)

        # self.apply(self._init_weights)

        # One-hot encoder initialization
        self.encoder = OneHotEncoder(sparse_output=False)
        self.encoder.fit(np.arange(1,num_classes+1).reshape(-1, 1))


    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=1.0)
    #         if module.bias is not None:
    #             module.bias.data.normal_(mean=0.0, std=1.0)

    def to_one_hot(self, y):
        # Convert labels to one-hot encoding
        y = y.view(-1, 1).cpu().numpy()  # Convert to numpy array
        y_one_hot = self.encoder.transform(y)
        return torch.FloatTensor(y_one_hot).to(self.device)

    def encode(self, x, y):
        y_one_hot = self.to_one_hot(y)
        con = torch.cat((x, y_one_hot), 1)
        h1 = self.fc1(con)
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



    def decode(self, z, y):
        y_one_hot = self.to_one_hot(y)
        cat = torch.cat((z, y_one_hot), 1)
        h1=self.fc11(cat)
        h1 = self.fc12(h1)
        h1 = self.fc13(h1)
        h1 = F.relu(self.fc14(h1))
        h1 = self.fc15(h1)
        h1 = self.fc16(h1)
        h1 = F.relu(self.fc17(h1))
        h1 = self.fc18(h1)
        h1 = self.fc19(h1)
        h1 = F.relu(self.fc20(h1))
        h1 = self.fc21(h1)
        h1 = self.fc22(h1)
        h1 = F.relu(self.fc23(h1))
        h1 = self.fc24(h1)
        return torch.sigmoid(h1)

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, 3549), y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar,z

# class FCResidualBlock(nn.Module):
#     def __init__(self, in_features,device):
#         super(FCResidualBlock, self).__init__()
#         self.fc1 = nn.Linear(in_features, in_features)
#         self.fc2 = nn.Linear(in_features, in_features)
#         self.device=device
#         self.to(device)
#     def forward(self, x):
#         identity = x
#         out = F.relu(self.fc1(x))
#         out = self.fc2(out)
#         #out += identity  # Add input directly to output
#
#         return F.relu(out)
# class CVAEWithResiduals(CVAE):
#     def __init__(self, num_classes, device):
#         super(CVAEWithResiduals, self).__init__(num_classes, device)
#
#         # Residual Blocks for the encoder
#         self.resblock1 = FCResidualBlock(1024,device)
#         self.resblock2 = FCResidualBlock(256,device)
#
#
#     def encode(self, x, y):
#         y_one_hot = self.to_one_hot(y)
#         con = torch.cat((x, y_one_hot), 1)
#         h1 = self.fc1(con)
#         h1 = self.fc2(h1)
#         h1 = self.resblock1(h1)  # Use residual block
#         h1 = F.relu(self.fc3(h1))
#         h1 = self.fc4(h1)
#         h1 = self.resblock2(h1)  # Use residual block
#         h1 = self.fc5(h1)
#         h1 = F.relu(self.fc6(h1))
#         h1 = self.fc7(h1)
#         h1 = self.fc8(h1)
#         h1 = F.relu(self.fc9(h1))
#         h1 = self.fc10(h1)
#         return self.fc11_1(h1), self.fc11_2(h1)
#
#     def decode(self, z, y):
#         y_one_hot = self.to_one_hot(y)
#         cat = torch.cat((z, y_one_hot), 1)
#         h1 = self.fc12(cat)
#         h1 = self.fc13(h1)
#         h1 = F.relu(self.fc14(h1))
#         h1 = self.fc15(h1)
#         h1 = self.fc16(h1)
#         h1 = F.relu(self.fc17(h1))
#         h1 = self.resblock2(h1)  # Use residual block
#         h1 = self.fc18(h1)
#         h1 = self.fc19(h1)
#         h1 = self.resblock1(h1)  # Use residual block
#         h1 = F.relu(self.fc20(h1))
#         h1 = self.fc21(h1)
#         return torch.sigmoid(h1)
#


# class LearnableLayerWithDifferentSizes(nn.Module):
#     def __init__(self, in_features, out_features_list):
#         super(LearnableLayerWithDifferentSizes, self).__init__()
#         self.layers = nn.ModuleList()
#         self.gates = nn.ParameterList()
#
#         # 为每个输出特征大小创建一个线性层和对应的门参数
#         for out_features in out_features_list:
#             self.layers.append(nn.Linear(in_features, out_features))
#             self.gates.append(nn.Parameter(torch.randn(1)))
#
#     def forward(self, x):
#         outputs = []
#         for layer, gate in zip(self.layers, self.gates):
#             gate_weight = torch.sigmoid(gate)
#             outputs.append(layer(x) * gate_weight)
#         # 将所有输出相加，合并不同层的信息
#         return torch.cat((outputs))
#
# class NASBasedCVAE(nn.Module):
#     def __init__(self, num_classes, device):
#         super(NASBasedCVAE, self).__init__()
#         self.num_classes = num_classes
#         self.device = device
#
#         input_dim = 3549 + num_classes  # Including one-hot encoded classes
#
#         # Example of applying NAS-based layers in encoder
#         self.fc1 = LearnableLayerWithDifferentSizes(input_dim, [2048, 1024, 512])
#         self.fc2 = LearnableLayerWithDifferentSizes(2048+1024+512, [1024, 512, 256])
#         self.fc3 = LearnableLayerWithDifferentSizes(1024+512+256,[512,256,128])
#         self.fc4 = LearnableLayerWithDifferentSizes(512+256+128,[256,128,64])
#         self.fc5= LearnableLayerWithDifferentSizes(256+128+64,[128,64,32])
#         self.fc6= LearnableLayerWithDifferentSizes(128+64+32,[64,32,16])
#         self.fc7=LearnableLayerWithDifferentSizes(64+32+16,[32,16,8])
#         self.fc8=LearnableLayerWithDifferentSizes(32+16+8,[16,8,4])
#
#
#         self.fc_mu = LearnableLayerWithDifferentSizes(16+8+4, [8,4,2])
#         self.fc_logvar = LearnableLayerWithDifferentSizes(16+8+4, [8,4,2])
#
#         # Example of applying NAS-based layers in decoder
#         self.fc9 = LearnableLayerWithDifferentSizes(8+4+2 + num_classes, [8, 16, 32])
#         self.fc10=LearnableLayerWithDifferentSizes(8+16+32,[16,32,64])
#         self.fc11=LearnableLayerWithDifferentSizes(16+32+64,[32,64,128])
#         self.fc12=LearnableLayerWithDifferentSizes(16+64+128,[64,128,256])
#         self.fc13=LearnableLayerWithDifferentSizes(64+128+256,[128,256,521])
#         self.fc14=LearnableLayerWithDifferentSizes(128+256+521,[256,521,1024])
#         self.fc15=LearnableLayerWithDifferentSizes(256+521+1024,[521,1024,2048])
#         self.fc16 = nn.Linear(521+1024+2048, 3549)  # Final layer to match the original input dimension
#
#         # One-hot encoder initialization
#         self.encoder = OneHotEncoder(sparse_output=False)
#         self.encoder.fit(np.arange(1, num_classes + 1).reshape(-1, 1))
#
#     def to_one_hot(self, y):
#         y = y.view(-1, 1).cpu().numpy()
#         y_one_hot = self.encoder.transform(y)
#         return torch.FloatTensor(y_one_hot).to(self.device)
#
#     def encode(self, x, y):
#         y_one_hot = self.to_one_hot(y)
#         con = torch.cat((x, y_one_hot), 1)
#         h1 = self.fc1(con)
#         h1 = self.fc2(h1)
#         h1 = F.relu(self.fc3(h1))
#         h1 = self.fc4(h1)
#         h1 = self.fc5(h1)
#         h1 = F.relu(self.fc6(h1))
#         h1 = self.fc7(h1)
#         h1 = F.relu(self.fc8(h1))
#         return self.fc_mu(h1), self.fc_logvar(h1)
#
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu)
#
#     def decode(self, z, y):
#         y_one_hot = self.to_one_hot(y)
#         cat = torch.cat((z, y_one_hot), 1)
#         h1 = self.fc9(cat)
#         h1 = self.fc10(h1)
#         h1 = F.relu(self.fc11(h1))
#         h1 = self.fc12(h1)
#         h1 = self.fc13(h1)
#         h1 = F.relu(self.fc14(h1))
#         h1 = self.fc15(h1)
#         h1 = F.relu(self.fc16(h1))
#         return torch.sigmoid(h1)
#
#
#     def forward(self, x, y):
#         mu, logvar = self.encode(x.view(-1, 3549), y)
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z, y), mu, logvar, z