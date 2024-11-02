import dgl
import torch.nn.functional as F
import torch as th
import torch.nn as nn
from src.utils import *
from src.propagation import Propagation


class HGDDI(nn.Module):
    def __init__(self, g,  n_drug, n_structure,n_target,n_enzyme,n_path,  args):
        super(HGDDI, self).__init__()
        self.g = g
        #self.device = th.device(args.device)
        self.device = th.device('cpu')
        self.dim_embedding = args.dim_embedding

        #节点嵌入的维度，作用是定义模型学习节点表示时的向量维度大小。嵌入维度的选择是模型的一个重要超参数，直接影响模型学习到的节点特征的表达能力和复杂度，从而影响模型的性能和泛化能力。

        self.activation = F.elu
        self.reg_lambda = args.reg_lambda



        self.num_drug = n_drug
        self.num_structure = n_structure
        self.num_target = n_target
        self.num_enzyme = n_enzyme
        self.num_path = n_path


        #初始化可学习的特征矩阵
        ##drug_feat等分别表示药物、蛋白质、疾病和副作用的特征矩阵，通过nn.parameter封装成可学习参数，并使用正态分布进行初始化
        self.drug_feat = nn.Parameter(th.FloatTensor(self.num_drug, self.dim_embedding))
        nn.init.normal_(self.drug_feat, mean=0, std=0.1)

        self.structure_feat = nn.Parameter(th.FloatTensor(self.num_structure, self.dim_embedding))
        nn.init.normal_(self.structure_feat, mean=0, std=0.1)

        self.structure_feat0 = nn.Parameter(th.FloatTensor(841, self.dim_embedding))
        nn.init.normal_(self.structure_feat0, mean=0, std=0.1)


        self.target_feat = nn.Parameter(th.FloatTensor(self.num_target, self.dim_embedding))
        nn.init.normal_(self.target_feat, mean=0, std=0.1)

        self.target_feat0 = nn.Parameter(th.FloatTensor(841, self.dim_embedding))
        nn.init.normal_(self.target_feat0, mean=0, std=0.1)




        self.enzyme_feat = nn.Parameter(th.FloatTensor(self.num_enzyme, self.dim_embedding))
        nn.init.normal_(self.enzyme_feat, mean=0, std=0.1)

        self.enzyme_feat0 = nn.Parameter(th.FloatTensor(841, self.dim_embedding))
        nn.init.normal_(self.enzyme_feat0, mean=0, std=0.1)



        self.path_feat = nn.Parameter(th.FloatTensor(self.num_path, self.dim_embedding))
        nn.init.normal_(self.path_feat, mean=0, std=0.1)

        self.path_feat0 = nn.Parameter(th.FloatTensor(841, self.dim_embedding))
        nn.init.normal_(self.path_feat0, mean=0, std=0.1)



####定义权重矩阵
        self.fc_D_D = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_S = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_T = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_E = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_D_P = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        #self.fc_D_E = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        #self.fc_D_P = nn.Linear(self.dim_embedding, self.dim_embedding).float()

        # Propagation模块，用于在图上执行信息传播。输入为图graph和特征feat，返回经信息传播后的特征feat
        self.propagation = Propagation(args.k, args.alpha, args.edge_drop)

        # Linear transformation for reconstruction
        tmp = th.randn(self.dim_embedding).float()
        self.re_D_D = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_S = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_T = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_E = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))
        self.re_D_P = nn.Parameter(th.diag(th.nn.init.normal_(tmp, mean=0, std=0.1)))

        #self.reset_parameters()#创建对角矩阵作为参数，通过reset_parameters()函数对模型中的线性层参数进行初始化。
    def reset_parameters(self):#重置模型中的参数，便利模型中的所有模块，如果发现是线性层nn.liner，将其权重初始化为0，
        #标准差为0.1的正态分布，并将偏置初始化为0.1。
        for m in HGDDI.modules(self):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)


    def forward(self, drug_structure, drug_target, drug_enzyme, drug_path,drug_drug, drug_drug_mask):
        drug_feat = th.mean(th.stack((th.mm(row_normalize(drug_drug).float(),
                                               F.relu(self.fc_D_D(self.drug_feat))),
                                      th.mm(row_normalize(drug_structure).float(),
                                               F.relu(self.fc_D_S(self.structure_feat))),
                                      th.mm(row_normalize(drug_target).float(),
                                               F.relu(self.fc_D_T(self.target_feat))),
                                      th.mm(row_normalize(drug_enzyme).float(),
                                               F.relu(self.fc_D_E(self.enzyme_feat))),
                                      th.mm(row_normalize(drug_path).float(),
                                               F.relu(self.fc_D_P(self.path_feat))),
                                      self.drug_feat), dim=1), dim=1)
        #首先，对药物相互作用、药物结构、药物靶点、药物酶、药物路径进行线性变换，并对结果应用ReLU激活函数。
        # 然后对结果进行归一化（row_normalize）和堆叠（stack），最后计算均值以得到药物特征。

        structure_feat = th.mean(th.stack((th.mm(row_normalize(drug_structure.T).float(),
                                            F.relu(self.fc_D_S(self.structure_feat0))),
                                      self.structure_feat), dim=1), dim=1)


        target_feat= th.mean(th.stack((th.mm(row_normalize(drug_target.T).float(),
                                                  F.relu(self.fc_D_T(self.target_feat0))),
                                            self.target_feat), dim=1), dim=1)

        enzyme_feat = th.mean(th.stack((th.mm(row_normalize(drug_enzyme.T).float(),
                                                  F.relu(self.fc_D_E(self.enzyme_feat0))),
                                            self.enzyme_feat), dim=1), dim=1)

        path_feat = th.mean(th.stack((th.mm(row_normalize(drug_path.T).float(),
                                                  F.relu(self.fc_D_P(self.path_feat0))),
                                            self.path_feat), dim=1), dim=1)




        #将提取的特征连接成一个节点特征张量，得到维度是（n_features,n_nodes）(特征数，节点数)
        node_feat = th.cat(( drug_feat,structure_feat, target_feat, enzyme_feat,path_feat), dim=0)

        node_feat = self.propagation(dgl.to_homogeneous(self.g), node_feat)
        #用Progation函数对信息进行传播，节点特征张张量被转换成为一个DGL图对象，更新节点特征。

        #提取节点特征张量中不同类型的特征向量，并将其移动到指定设备上，以便后续的计算和处理。
        drug_embedding = node_feat[:self.num_drug].to(self.device)
        structure_embedding = node_feat[self.num_drug:self.num_drug + self.num_structure].to(self.device)

        target_embedding = node_feat[self.num_drug + self.num_structure:self.num_drug + self.num_structure +
                                                                       self.num_target].to(self.device)

        enzyme_embedding = node_feat[self.num_drug + self.num_structure +self.num_target:self.num_drug +
                                    self.num_structure +self.num_target+self.num_enzyme].to(self.device)
        path_embedding = node_feat[-self.num_path:].to(self.device)


        #对不同类型的特征向量进行L2范数归一化，以确保他们的长度都为1，这样可以更好的展示它们之间的相对关系，同时有助于优化算法的稳定性和收敛性。
        drug_vector = l2_norm(drug_embedding)
        structure_vector = l2_norm(structure_embedding)
        target_vector = l2_norm(target_embedding)
        enzyme_vector = l2_norm(enzyme_embedding)
        path_vector = l2_norm(path_embedding)


        #计算重构模型对不同类型药物关系的重构性能，重构损失表示了重构模型在学习过程中预测值和真实值之间的差异。
        drug_structure_reconstruct = th.mm(th.mm(drug_vector, self.re_D_S), structure_vector.t())
        drug_structure_reconstruct_loss = th.sum((drug_structure_reconstruct - drug_structure.float()) ** 2)

        drug_target_reconstruct = th.mm(th.mm(drug_vector, self.re_D_T), target_vector.t())
        drug_target_reconstruct_loss = th.sum((drug_target_reconstruct - drug_target.float()) ** 2)
        drug_enzyme_reconstruct = th.mm(th.mm(drug_vector, self.re_D_E), enzyme_vector.t())
        drug_enzyme_reconstruct_loss = th.sum((drug_enzyme_reconstruct - drug_enzyme.float()) ** 2)

        drug_path_reconstruct = th.mm(th.mm(drug_vector, self.re_D_P), path_vector.t())
        drug_path_reconstruct_loss = th.sum((drug_path_reconstruct - drug_path.float()) ** 2)




        drug_drug_reconstruct = th.mm(th.mm(drug_vector, self.re_D_D), drug_vector.t())

        tmp = th.mul(drug_drug_mask.float(), (drug_drug_reconstruct - drug_drug.float()))#计算重构矩阵和真实药物矩阵之间的差异并用掩码过滤掉无效数据。
        DDI_potential = drug_drug_reconstruct - drug_drug.float()#直接计算重构矩阵和真实药物相互作用矩阵之间的差异。

        drug_drug_reconstruct_loss = th.sum(tmp ** 2)

        #other_loss = drug_structure_reconstruct_loss + drug_target_reconstruct_loss + \
                     #drug_enzyme_reconstruct_loss + drug_path_reconstruct_loss

        other_loss =  drug_structure_reconstruct_loss + drug_target_reconstruct_loss + \
                     drug_enzyme_reconstruct_loss + drug_path_reconstruct_loss

        #得到所有非偏置参数的L2范数正则化损失，用于控制模型的复杂度，防止过拟合。
        L2_loss = 0.#初始化L2_LOSS=0,用于累计所有非偏置参数的平方和
        for name, param in HGDDI.named_parameters(self):
            if 'bias' not in name:
                L2_loss = L2_loss + th.sum(param.pow(2))#遍历所有参数，除去不是非偏置参数的参数，因为我们只对非偏置参数进行处理。
                #对于非偏置参数，将所有元素的平方求和，并累加叫L2_loss中。
        L2_loss = L2_loss * 0.5#将L2_loss乘以0.5，因为在计算L2范数正则化损失是，习惯上将平方和除以2，这样在求导时能够消除系数2。


        #tloss = drug_drug_reconstruct_loss + 1.0 * other_loss
        #tloss = other_loss

        #tloss = drug_drug_reconstruct_loss
        #tloss = drug_structure_reconstruct_loss
        #tloss = drug_target_reconstruct_loss
        #tloss = drug_enzyme_reconstruct_loss
        #tloss = drug_path_reconstruct_loss

        #tloss = drug_structure_reconstruct_loss + drug_target_reconstruct_loss
        #tloss = drug_structure_reconstruct_loss + drug_enzyme_reconstruct_loss
        #tloss = drug_structure_reconstruct_loss + drug_path_reconstruct_loss
        #tloss = drug_target_reconstruct_loss + drug_enzyme_reconstruct_loss
        #tloss = drug_target_reconstruct_loss + drug_path_reconstruct_loss
        #tloss = drug_enzyme_reconstruct_loss + drug_path_reconstruct_loss





        #tloss = drug_structure_reconstruct_loss + drug_drug_reconstruct_loss
        #tloss = drug_target_reconstruct_loss + drug_drug_reconstruct_loss
        #tloss = drug_enzyme_reconstruct_loss + drug_drug_reconstruct_loss
        #tloss = drug_path_reconstruct_loss + drug_drug_reconstruct_loss


        #tloss =  drug_structure_reconstruct_loss + drug_target_reconstruct_loss + drug_enzyme_reconstruct_loss
        #tloss = drug_structure_reconstruct_loss + drug_target_reconstruct_loss + drug_path_reconstruct_loss
        #tloss = drug_path_reconstruct_loss + drug_target_reconstruct_loss + drug_enzyme_reconstruct_loss
        #tloss = drug_structure_reconstruct_loss + drug_enzyme_reconstruct_loss + drug_path_reconstruct_loss
        #tloss = drug_drug_reconstruct_loss + drug_structure_reconstruct_loss + drug_target_reconstruct_loss
        #tloss = drug_drug_reconstruct_loss + drug_structure_reconstruct_loss + drug_enzyme_reconstruct_loss
        #tloss = drug_drug_reconstruct_loss + drug_structure_reconstruct_loss + drug_path_reconstruct_loss



        tloss = drug_drug_reconstruct_loss + drug_target_reconstruct_loss + drug_enzyme_reconstruct_loss
        #tloss = drug_drug_reconstruct_loss + drug_target_reconstruct_loss + drug_path_reconstruct_loss
        #tloss = drug_drug_reconstruct_loss + drug_enzyme_reconstruct_loss + drug_path_reconstruct_loss

        #tloss = drug_drug_reconstruct_loss + drug_structure_reconstruct_loss + drug_target_reconstruct_loss + drug_enzyme_reconstruct_loss
        #tloss = drug_drug_reconstruct_loss + drug_structure_reconstruct_loss + drug_enzyme_reconstruct_loss + drug_path_reconstruct_loss
        #tloss = drug_drug_reconstruct_loss + drug_structure_reconstruct_loss + drug_target_reconstruct_loss + drug_path_reconstruct_loss
        #tloss = drug_drug_reconstruct_loss + drug_target_reconstruct_loss + drug_enzyme_reconstruct_loss + drug_path_reconstruct_loss


        return tloss, drug_drug_reconstruct_loss, L2_loss, drug_drug_reconstruct, DDI_potential
