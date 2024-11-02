# -*- coding: utf-8 -*-

import dgl
import time
import torch as th
import numpy as np
import matplotlib.pyplot as plt

from src.utils import *
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from src.model import HGDDI
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import matthews_corrcoef


def loda_data():
    network_path = '../ourdata/'

    drug_drug = np.loadtxt(network_path + 'DDadjacency_matrix.txt')
    drug_structure = np.loadtxt(network_path + 'DSadjacency_matrix.txt')
    drug_target = np.loadtxt(network_path + 'DTadjacency_matrix.txt')
    drug_enzyme = np.loadtxt(network_path + 'DEadjacency_matrix.txt')
    drug_path = np.loadtxt(network_path + 'DPadjacency_matrix.txt')
    DD_data = np.loadtxt(network_path + 'DDadjacency_matrix.txt')

    print("Load data finished.")

    return drug_structure, drug_target, drug_enzyme, drug_path, drug_drug

def ConstructGraph(drug_structure, drug_target, drug_enzyme, drug_path, drug_drug):
    num_drug = 841
    num_structure = 619
    num_target = 1333
    num_enzyme = 214
    num_path = 307

    list_drug = []
    for i in range(num_drug):
        list_drug.append((i, i))

    list_structure = []
    for i in range(num_structure):
        list_structure.append((i, i))

    list_target = []
    for i in range(num_target):
        list_target.append((i, i))

    list_enzyme = []
    for i in range(num_enzyme):
        list_enzyme.append((i, i))

    list_path = []
    for i in range(num_path):
        list_path.append((i, i))

    #list_DDI = []
    #for row in range(num_drug):
        #for col in range(num_drug):
            #if drug_drug[row, col] > 0:
                #list_DDI.append((row, col))


    #list_drug_drug = []
    list_drug_drug = []
    for row in range(num_drug):
        for col in range(num_drug):
            if drug_drug[row, col] > 0:
                list_drug_drug.append((row, col))
                list_drug_drug.append((col, row))


    list_drug_structure = []
    list_structure_drug = []
    for row in range(num_drug):
        for col in range(num_structure):
            if drug_structure[row, col] > 0:
                list_drug_structure.append((row, col))
                list_structure_drug.append((col, row))

    list_drug_target = []
    list_target_drug = []
    for row in range(num_drug):
        for col in range(num_target):
            if drug_target[row, col] > 0:
                list_drug_target.append((row, col))
                list_target_drug.append((col, row))

    list_drug_enzyme = []
    list_enzyme_drug = []
    for row in range(num_drug):
        for col in range(num_enzyme):
            if drug_enzyme[row, col] > 0:
                list_drug_enzyme.append((row, col))
                list_enzyme_drug.append((col, row))

    list_drug_path = []
    list_path_drug = []
    for row in range(num_drug):
        for col in range(num_path):
            if drug_path[row, col] > 0:
                list_drug_path.append((row, col))
                list_path_drug.append((col, row))


    g_HIN = dgl.heterograph({('drug', 'drug_drug virtual', 'drug'): list_drug,
                             ('structure', 'structure_structure virtual', 'structure'): list_structure,
                             ('target', 'target_target virtual', 'target'): list_target,
                             ('enzyme', 'enzyme_enzyme virtual', 'enzyme'): list_enzyme,
                             ('path', 'path_path virtual', 'path'): list_path,
                             ('drug', 'drug_drug interaction', 'drug'): list_drug_drug, \
                             ('drug', 'drug_structure interaction', 'structure'): list_drug_structure, \
                             ('structure', 'structure_drug interaction', 'drug'): list_structure_drug, \
                             ('drug', 'drug_target association', 'target'): list_drug_target, \
                             ('target', 'target_drug association', 'drug'): list_target_drug, \
                             ('drug', 'drug_enzyme association', 'enzyme'): list_drug_enzyme, \
                             ('enzyme', 'enzyme_drug association', 'drug'): list_enzyme_drug, \
                             ('drug', 'drug_path association', 'path'): list_drug_path, \
                             ('path', 'path_drug association', 'drug'): list_path_drug})
    g = g_HIN.edge_type_subgraph(['drug_drug interaction', 'drug_structure interaction',
                                  'structure_drug interaction', 'drug_target association',
                                  'target_drug association', 'drug_enzyme association',
                                  'enzyme_drug association', 'drug_path association',
                                  'path_drug association'
                                  ])

    return g

def TrainAndEvaluate(DDItrain, DDIvalid, DDItest, args, drug_structure, drug_target, drug_enzyme, drug_path, drug_drug):

    #device = th.device(args.device)
    device = th.device('cpu')
    #results111=[]

    # Numbers of different nodes
    num_drug = len(drug_drug.T)
    num_structure = len(drug_structure.T)
    num_target = len(drug_target.T)
    num_enzyme = len(drug_enzyme.T)
    num_path = len(drug_path.T)
    drug_drug = th.zeros((num_drug, num_drug))
    mask = th.zeros((num_drug, num_drug)).to(device)

    for ele in DDItrain:##drug_drug中存储初始预测值，mask中存储哪些相互作用是真实存在的
        drug_drug[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1

    best_valid_aupr = 0.
    best_valid_auc = 0
    test_aupr = 0.
    test_auc = 0.
    patience = 0.

    pos = np.count_nonzero(DDItest[:, 2])#统计测试集中正样本（即药物-蛋白质相互作用存在的样本）的数量，
    # 通过 DDItest[:, 2] 提取所有标签，然后使用 np.count_nonzero() 函数计算非零元素的数量，即正样本的数量。
    neg = np.size(DDItest[:, 2]) - pos#统计测试集中负样本（即药物-蛋白质相互作用不存在的样本）的数量，通过总样本数量减去正样本数量得到。
    xy_roc_sampling = []
    xy_pr_sampling = []

    g = ConstructGraph(drug_structure, drug_target, drug_enzyme, drug_path, drug_drug)

    drug_structure = th.tensor(drug_structure).to(device)
    drug_target = th.tensor(drug_target).to(device)
    drug_enzyme = th.tensor(drug_enzyme).to(device)
    drug_path = th.tensor(drug_path).to(device)
    drug_drug = drug_drug.to(device)


    model = HGDDI(g, num_drug,num_structure,num_target,num_enzyme,num_path, args)
    model.to(device)

    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for i in range(args.epochs):#共执行args.epoch次循环

        model.train()
        tloss, ddiloss, l2loss, dd_re, DDI_p = model(drug_structure, drug_target, drug_enzyme, drug_path,drug_drug,mask)
        #返回的是总损失=ddloss+other_loss+L2_loss，dd的重构损失，L2损失，潜在的DDI
        results = dd_re.detach().cpu()
        optimizer.zero_grad()
        loss = tloss
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        model.eval()


        if i % 5 == 0:#经过25次循环时执行其中的代码块
            with th.no_grad():
                print("step", i, ":", "Total_loss & DDIloss & L2_loss:", loss.cpu().data.numpy(), ",", ddiloss.item(),
                      ",", l2loss.item())

                pred_list = []
                ground_truth = []

                for ele in DDIvalid:
                    pred_list.append(results[ele[0], ele[1]])
                    ground_truth.append(ele[2])

                valid_auc = roc_auc_score(ground_truth, pred_list)
                valid_aupr = average_precision_score(ground_truth, pred_list)

                if valid_aupr >= best_valid_aupr:
                    best_valid_aupr = valid_aupr
                    # best_valid_auc = valid_auc
                    best_DDI_potential = DDI_p
                    patience = 0

                    # Calculating AUC & AUPR (pos:neg=1:10)
                    db = []
                    xy_roc = []
                    xy_pr = []
                    for ele in DDItest:
                        #db.append([results[ele[0], ele[1]], ele[2]])
                        db.append([float(results[ele[0], ele[1]]), float(ele[2])])

                    db = sorted(db, key=lambda x: x[0], reverse=True) #按第一列排序

                    tp, fp = 0., 0.
                    for i_db in range(len(db)):
                        if db[i_db][0]:
                            if db[i_db][1]:
                                tp = tp + 1
                            else:
                                fp = fp + 1
                            xy_roc.append([fp / neg, tp / pos])
                            xy_pr.append([tp / pos, tp / (tp + fp)])

                    test_auc = 0.
                    prev_x = 0.
                    for x, y in xy_roc:
                        if x != prev_x:
                            test_auc += (x - prev_x) * y
                            prev_x = x

                    test_aupr = 0.
                    prev_x = 0.
                    for x, y in xy_pr:
                        if x != prev_x:
                            test_aupr += (x - prev_x) * y
                            prev_x = x

                    y_true = [ele[2] for ele in DDItest]
                    y_pred = [1.0 if ele[0] > 0.5 else 0.0 for ele in db]
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred)
                    recall = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    MCC = matthews_corrcoef(y_true, y_pred)


                    # All unknown DDI pairs all treated as negative examples


                else:
                    patience += 1
                    if patience > args.patience:
                        print("Early Stopping")

                        # sampling (pos:neg=1:10) for averaging and plotting
                        xy_roc_sampling = []
                        xy_pr_sampling = []
                        for i_xy in range(len(xy_roc)):
                            if i_xy % 10 == 0:
                                xy_roc_sampling.append(xy_roc[i_xy])
                                xy_pr_sampling.append(xy_pr[i_xy])

                        # Record data for sampling, averaging and plotting.
                        # All unknown DDI pairs all treated as negative examples
                        break

                print('Valid auc & aupr:', valid_auc, valid_aupr, ";  ", 'Test auc & aupr:', test_auc, test_aupr)


    return test_auc, test_aupr, xy_roc_sampling, xy_pr_sampling, best_DDI_potential,precision,recall,f1,accuracy,db , y_pred, y_true,ground_truth, pred_list,MCC



if __name__ == "__main__":
    args = parse_args()
    print("执行操作，参数：", args)

    drug_s, drug_t, drug_e, drug_p, dd_original = loda_data()

    # 从创建一个正负样本为1：10的数据集data_set,用于后续模型的训练和评估
    whole_positive_index = []
    whole_negative_index = []
    for i in range(np.shape(dd_original)[0]):
        for j in range(np.shape(dd_original)[1]):
            if int(dd_original[i][j]) == 1:
                whole_positive_index.append([i, j])
            elif int(dd_original[i][j]) == 0:
                whole_negative_index.append([i, j])
    # pos:neg=1:10正负样本1：10
    negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                             size=1 * len(whole_positive_index), replace=True)
    # All unknown DDI pairs all treated as negative examples
    data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1

    test_auc_round = []  # 记录训练中测试机的AUC
    test_aupr_round = []  # AUPR
    tpr_mean = []  # tpr
    fpr = []  # fpr
    precision_mean = []  # 精确率Precision
    recall = []  # 召回率recal
    rounds = args.rounds  # 迭代轮数
    for r in range(rounds):
        print("----------------------------------------")
        # 用于记录当前训练轮数的测试集AUC和AUPR的空列表
        test_auc_fold = []
        test_aupr_fold = []

        kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
        k_fold = 0

        for train_index, test_index in kf.split(data_set[:, :2], data_set[:, 2]):
            train = data_set[train_index]
            DDItest = data_set[test_index]
            DDItrain, DDIvalid = train_test_split(train, test_size=0.05, random_state=None)  # 从当前折的训练集中划分一部分数据中作为验证集o

            k_fold += 1
            print("--------------------------------------------------------------")
            print("round ", r + 1, " of ", rounds, ":", "KFold ", k_fold, " of 5")
            print("--------------------------------------------------------------")

            time_roundStart = time.time()

            # TrainAndEvaluate返回AUC、AUPR、ROC曲线、PR曲线、DDI预测概率矩阵，保存在下列变量中
            t_auc, t_aupr, xy_roc, xy_pr, DDI_potential, precision, recall, f1, accuracy, db, y_pred, y_true,ground_truth, pred_list,MCC= TrainAndEvaluate(
                DDItrain, DDIvalid, DDItest, args, drug_s, drug_t, drug_e, drug_p, dd_original)


            #np.savetxt(f'test_db_{k_fold}.txt', db, fmt='%.6f')
            #np.savetxt(f'test_true_{k_fold}.txt', y_true, fmt='%.6f')
            #np.savetxt(f'y_pred_{k_fold}.txt', y_pred, fmt='%.6f')
            #np.savetxt(f'y_true_{k_fold}.txt', y_true, fmt='%.6f')
            #np.savetxt(f'ground_true_{k_fold}.txt', ground_truth, fmt='%.6f')
            #np.savetxt(f'pred_list_{k_fold}.txt', pred_list, fmt='%.6f')








            print("########################################################################")
            print(t_auc, t_aupr, precision, recall, f1, accuracy,MCC)
            print("########################################################################")

            time_roundEnd = time.time()
            print("Time spent in this fold:", time_roundEnd - time_roundStart)
            test_auc_fold.append(t_auc)
            test_aupr_fold.append(t_aupr)



            # pos:neg=1:1
            if not fpr:
                fpr = [_v[0] for _v in xy_roc]
            if not recall:
                recall = [_v[0] for _v in xy_pr]

            temp = [_v[1] for _v in xy_roc]
            tpr_mean.append(temp)
            temp = [_v[1] for _v in xy_pr]
            precision_mean.append(temp)

        print("Training and evaluation is OK.")

        test_auc_round.append(np.mean(test_auc_fold))
        test_aupr_round.append(np.mean(test_aupr_fold))

    t1 = time.localtime()


    # pos:neg=1:10
    tpr = (np.mean(np.array(tpr_mean), axis=0)).tolist()
    precision = (np.mean(np.array(precision_mean), axis=0)).tolist()



    start = time.time()
    end = time.time()
    print("Total time:", end - start)
