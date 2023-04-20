import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GATConv
import torch_geometric.transforms as T
torch.manual_seed(42)

class GNN(torch.nn.Module):
    def __init__(self,num_classes,embedding_size,data_feature):
        super(GNN,self).__init()__
        self.num_classes = num_classes
        self.hidden = embedding_size
        self.data_feat = data_feature

        self.conv1 = GATConv(self.data_feat,self.hidden,heads=3,dropout=0.4)
        self.lin1 = Linear(3*self.hidden,self.hidden)
        self.pool1 = TopKPooling(self.hidden,ratio=0.75)
        self.conv2 = GATConv(self.data_feat,self.hidden,heads=3,dropout=0.4)
        self.lin2 = Linear(3*self.hidden,self.hidden)
        self.pool2 = TopKPooling(self.hidden,ratio=0.45)
        self.conv3 = GATConv(self.data_feat,self.hidden,heads=3,dropout=0.4)
        self.lin3 = Linear(3*self.hidden,self.hidden)
        self.pool3 = TopKPooling(self.hidden,ratio=0.2)

        self.linlast1 = Linear(self.hidden*2,self.hidden)
        self.linlast2 = Linear(self.hidden,self.num_classes)


    def forward(self,x,edge_index,edge_attr,batch_index):
        '''
        Forward function that implements three blocks of GATConv,TopKPooling and Linear layer
        '''
        x = self.conv1(x,edge_index)
        x = self.lin1(x)

        x,edge_index,edge_attr,batch_index, _, _ = self.pool1(x,edge_index,None,batch_index)
        x1 = torch.cat([gap(x,batch_index),gmp(x,batch_index)],dim=1)

        x = self.conv2(x,edge_index)
        x = self.lin2(x)

        x,edge_index,edge_attr,batch_index, _, _ = self.pool2(x,edge_index,None,batch_index)

        x2 = torch.cat([gap(x,batch_index),gmp(x,batch_index)],dim=1)

        x = self.conv3(x,edge_index)
        x = self.lin3(x)

        x,edge_index,edge_attr,batch_index, _, _ = self.pool3(x,edge_index,None,batch_index)

        x3 = torch.cat([gap(x,batch_index),gmp(x,batch_index)],dim=1)

        return x1+x2+x3
