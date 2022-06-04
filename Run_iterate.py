import time
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from operator import itemgetter
from torch.utils.data import DataLoader

from model import CNFNet
from util import GraphDataset, chunks, print_network, plot_metric, read_ic

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.io import savemat, loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=8, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='c499', help='Dataset name')

args = parser.parse_args()
args.cuda = True

# Training setting
args.epochs = 50
args.num_feat = 31
args.batch_size = 8

# FC setting
args.energy_input_dim = 31 + 7 
Number_of_scaler_features = 45 

# Loading Selected Features indexes 
Dict_SF = loadmat("SF.mat")
SF = Dict_SF["Selected_Features"]

Num_iter = 1
All_MSE = []

for i in range(Num_iter):
    #Loading data 
    c = args.data # The name of dataset 
    inc_feat, feats, times, train_num, val_num, test_num = read_ic(c,Number_of_scaler_features,SF)
    
    model = CNFNet(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print_network(model)
    
    graph_loader = DataLoader(GraphDataset(train_num[:int(len(train_num) / args.batch_size) * args.batch_size]),
                              batch_size=args.batch_size, shuffle=True)
    
    args.num_instance = len(times)
    cri = nn.MSELoss()
    
    train_loss = []
    eval_loss = []
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True
    
    # Training Model
    for epoch in range(args.epochs):
        for step, ids in enumerate(graph_loader):
            t = time.time()
            model.train()
            output = model(itemgetter(*ids)(inc_feat), feats[ids])
            loss_train = cri(output, times[ids])
    
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
    
            print('Epoch: {:02d}/{:04d}'.format(epoch, step + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'time: {:.4f}s'.format(time.time() - t), end='\n')
    
            if step % 10 == 0:
                model.eval()
                val_ids = list(chunks(val_num, args.batch_size))[:-1]
                output_eval = [model(itemgetter(*_)(inc_feat), feats[_]) for _ in val_ids]
                loss_val = np.mean([cri(output_eval[_], times[val_ids[_]]).item() for _ in range(len(val_ids))])
                print("Eval loss: {}".format(loss_val), end='\n')
                train_loss.append(loss_train.item())
                eval_loss.append(loss_val.item())
                
    # print training info
    plot_metric(range(len(train_loss)), train_loss, eval_loss, '{}_train_{}'.format(c,i),
                '{}_eval_{}'.format(c,i))
    print("Optimization Finished!")
    #print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    # Testing the model 
    model.eval()
    
    test_ids = list(chunks(test_num, args.batch_size))
    output_test = [model(itemgetter(*_)(inc_feat), feats[_]) for _ in test_ids]
    loss_val = np.mean([cri(output_test[_], times[test_ids[_]]).item() for _ in range(len(test_ids))])
    print("Test loss: {}".format(loss_val))
    time_test = times[torch.cat(test_ids)].data.numpy()
    time_pred = torch.cat(output_test).data.numpy().flatten()
    plot_metric(range(time_test.size), time_pred, time_test, '{}_pred_{}'.format(c,i),
                '{}_real_{}'.format(c,i))
    
    MSE = mean_squared_error(time_test, time_pred)
    print(MSE)
    All_MSE.append(MSE)
    p1 , p2 = pearsonr(time_test, time_pred)
    sp1, sp2 = spearmanr(time_test, time_pred)
    
All_MSE = np.asarray(All_MSE)
Mean_All_MSE = np.mean(All_MSE)
Mydic = {"All_MSE":All_MSE,"Mean_All_MSE":Mean_All_MSE}
savemat('results.mat',Mydic)



