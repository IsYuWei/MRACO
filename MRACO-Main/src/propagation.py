
import torch as th
from torch import nn
from dgl import function as fn#Deep Graph Library


class Propagation(nn.Module):
    def __init__(self, k, alpha, edge_drop=0.):
        super(Propagation, self).__init__()
        self._k = k#传播迭代次数
        self._alpha = alpha#平衡参数
        self.edge_drop = nn.Dropout(edge_drop)#边缘丢弃率。nn.Dropout模块用于在训练过程中以概率p随机地将输入张量中的一部分元素设置为0，这是一种正则化技术，防止过拟合。
        #在图结构中，被用于边的级别，随机丢弃图中的一部分边，以正则化模型防止过拟合。

    def forward(self, graph, feat):#定义前向传播函数
        #graph = graph.local_var().to('cuda')
        graph = graph.local_var().to('cpu')  # 将图数据移到 CPU
        norm = th.pow(graph.in_degrees().float().clamp(min=1e-12), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = th.reshape(norm, shp).to(feat.device)
        feat_0 = feat#接受graph和feat作为输入，计算归一化因子，用于归一化节点特征。
        for _ in range(self._k):
            feat = feat * norm
            graph.ndata['h'] = feat
            graph.edata['w'] = self.edge_drop(th.ones(graph.number_of_edges(), 1).to(feat.device))
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')
            feat = feat * norm
            feat = (1 - self._alpha) * feat + self._alpha * feat_0
            #进行k次信息传播，每次迭代中，首先对特征进行归一化处理，然后将归一化后的特征保存到图中，根据边权重及逆行信息聚合并更新节点特征，最后对节点特征进行混合以平衡新特征和原始特征。
        return feat#返回经信息传播后的特征
