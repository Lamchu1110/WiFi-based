# -*- coding: utf-8 -*-

import torch
from torch import nn

from .utils import ScaledDotProductAttention


class SKConv(nn.Module):
    def __init__(self, input_dim, output_dim, dim1, dim2, pool_dim,  M=4, G=1, r=4, stride=1 ,L=32, layout_mode='legacy'):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.2
        """
        super(SKConv, self).__init__()
        if layout_mode not in ('legacy', 'corrected'):
            raise ValueError('layout_mode must be legacy or corrected')
        self.layout_mode = layout_mode

        hidden_dim = int(input_dim/r)
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        self.M = M
        #self.features = features
        self.pool_dim = pool_dim
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, groups=G, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1,1))


        if pool_dim == 'freq':
           d = int(dim1/r)
           self.fc = nn.Sequential(nn.Linear(dim1, d),
                                nn.BatchNorm1d(d),
                                nn.ReLU(inplace=True))
           self.fcs = nn.ModuleList([])
           for i in range(M):
               self.fcs.append(
                 nn.Conv1d(in_channels= d, out_channels=dim1, kernel_size=1, stride=1)
                 )

        elif pool_dim == 'freq-time':
           d = int(dim1*dim2/r)
           self.fc = nn.Sequential(nn.Linear(dim1*dim2, d),
                                nn.BatchNorm1d(d),
                                nn.ReLU(inplace=True))
           self.fcs = nn.ModuleList([])
           for i in range(M):
               self.fcs.append(
                 nn.Conv1d(in_channels= d, out_channels=dim1*dim2, kernel_size=1, stride=1)
                 )

        elif pool_dim == 'freq-chan':
           d = int(output_dim/r)
           self.fc = nn.Sequential(nn.Conv1d(output_dim, d ,kernel_size=1, stride=1),
                                nn.BatchNorm1d(d),
                                nn.ReLU(inplace=True))
           self.fcs = nn.ModuleList([])
           for i in range(M):
               self.fcs.append(
                 nn.Conv1d(in_channels= d, out_channels=output_dim, kernel_size=1, stride=1)
                 )



        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.layout_mode == 'corrected':
            return self.forward_corrected(x)
        return self.forward_legacy(x)

    def forward_corrected(self, x):
        # Keep the physical position of every element: [B,M,C,F,T].
        feats = torch.stack([conv(x) for conv in self.convs], dim=1)
        batch, _, channels, frequency, time = feats.shape
        if frequency != self.dim1:
            raise ValueError(f'DSK frequency axis: expected {self.dim1}, got {frequency}')
        summed = feats.sum(dim=1)
        if self.pool_dim == 'freq-chan':
            encoded = self.fc(summed.mean(dim=3))  # [B,C/r,F]
            attention = torch.stack([fc(encoded) for fc in self.fcs], dim=1).unsqueeze(-1)
        elif self.pool_dim == 'freq':
            encoded = self.fc(summed.mean(dim=(1, 3))).unsqueeze(-1)
            attention = torch.stack([fc(encoded) for fc in self.fcs], dim=1).unsqueeze(2)
        elif self.pool_dim == 'freq-time':
            if time != self.dim2:
                raise ValueError(f'DSK time axis: expected {self.dim2}, got {time}')
            encoded = self.fc(summed.mean(dim=1).reshape(batch, frequency*time)).unsqueeze(-1)
            attention = torch.stack([fc(encoded) for fc in self.fcs], dim=1).reshape(batch, self.M, 1, frequency, time)
        else:
            raise ValueError(f'Unsupported DSK attention: {self.pool_dim}')
        return (feats * self.softmax(attention)).sum(dim=1)

    def forward_legacy(self, x):

        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, feats.shape[2],self.output_dim, feats.shape[3])

        feats_U = torch.sum(feats, dim=1)

        if self.pool_dim == 'freq' :
          feats_S = torch.mean(feats_U,dim=[2,3])
          feats_S = feats_S.view(batch_size, feats_S.shape[1])
          feats_Z = self.fc(feats_S).unsqueeze(2)
          attention_vectors = [fc(feats_Z) for fc in self.fcs]
          attention_vectors = torch.cat(attention_vectors, dim=1)
          attention_vectors = attention_vectors.view(batch_size, self.M, self.dim1, 1, 1)
          attention_vectors = self.softmax(attention_vectors)
        elif self.pool_dim == 'freq-time':
          feats_S = torch.mean(feats_U, dim =2)
          feats_S = feats_S.view(batch_size,feats_S.shape[1]*feats_S.shape[2])
          feats_Z = self.fc(feats_S).unsqueeze(2)
          attention_vectors = [fc(feats_Z) for fc in self.fcs]
          attention_vectors = torch.cat(attention_vectors, dim=1)
          attention_vectors = attention_vectors.view(batch_size, self.M, self.dim1*self.dim2, 1, 1)
          attention_vectors = self.softmax(attention_vectors)
          attention_vectors = attention_vectors.view(batch_size,self.M,self.dim1,1,self.dim2)
        elif self.pool_dim == 'freq-chan':
          feats_S = torch.mean(feats_U, dim = 3)
          feats_S = feats_S.view(batch_size,feats_S.shape[2],feats_S.shape[1])
          feats_Z = self.fc(feats_S)
          attention_vectors = [fc(feats_Z) for fc in self.fcs]
          attention_vectors = torch.cat(attention_vectors, dim=1)
          attention_vectors = attention_vectors.view(batch_size, self.M,self.output_dim, self.dim1, 1)
          attention_vectors = self.softmax(attention_vectors)
          attention_vectors = attention_vectors.view(batch_size,self.M,self.dim1,self.output_dim,1)

        feats_V = torch.sum(feats*attention_vectors, dim=1)
        feats_V = torch.transpose(feats_V,1,2)
        return feats_V

class SKUnit(nn.Module):
    def __init__(self, in_features, mid_features, out_features, dim1,dim2,pool_dim, M=4, G=1, r=4, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=stride, bias=False),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(inplace=True)
            )
        
        self.conv2_sk = nn.Sequential(
            SKConv(input_dim=mid_features, output_dim=out_features, dim1= dim1,dim2=dim2 ,pool_dim=pool_dim, M = 4, G=1, r=4, stride=1 ,L=32),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            )
        
        
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_features, mid_features*2, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_features*2),
            nn.ReLU(inplace=True),
            
            )
     

        if in_features == out_features: # when dim not change, input_features could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features,out_features , 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2_sk(out)
        #out = self.conv3(out)
        # print(f"Output shape: {out.size()}")
        # out = self.attention(out)
        #return self.relu(out + self.shortcut(residual))
        #return self.relu(out)
        return out
