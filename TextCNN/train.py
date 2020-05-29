# -*- coding: utf-8 -*-
"""
@author: HSU, CHIH-CHAO
@Modified by Konstantin Sozykin
"""

import os
import sys
import argparse
#os.environ["CUDA_LAUNCH_BLOCKING"]= "1"

def parse_args():
    parser = argparse.ArgumentParser(description='TextCNN')
    
    parser = argparse.ArgumentParser(description='TextCNN')
    #Training args
    parser.add_argument('--root', type=str, default='/ksozykinraid/data/nlp/IMDB/',
                        help='training data in CSV format')
    
    parser.add_argument('--csv', type=str, default='imdb.csv',
                        help='training data in CSV format')
    
    parser.add_argument('--spacy-lang', type=str, default='en', 
                        help='language choice for spacy to tokenize the text')
    
    parser.add_argument('--gpu_ids','-g', type=str, default='6', 
                        help='str with gpu ids')
                        
    parser.add_argument('--pretrained', type=str, default="twitter.27B.50d",
                    help='choice of pretrined word embedding from torchtext')              
                        
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train (default: 10)')
    
    
    parser.add_argument('--fix_length', type=int, default=32,
                        help='fix length')
    
    
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='SGD weight_decay (default: 1e-4)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
    
    parser.add_argument('--val-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    
    parser.add_argument('--kernel-height', type=str, default='3,4,5',
                    help='how many kernel width for convolution (default: 3, 4, 5)')
    
    parser.add_argument('--out-channel', type=int, default=100,
                    help='output channel for convolutionaly layer (default: 100)')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate for linear layer (default: 0.5)')
    
    parser.add_argument('--num-class', type=int, default=2,
                        help='number of category to classify (default: 2)')
    
    return parser.parse_args()


args = parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_ids

import torch
import torch.optim as optim
import torch.nn.functional as F


sys.path.append('..')
from datasets import imdb as dataset
import models
from torch.optim.lr_scheduler import MultiStepLR





def train(m, device, train_itr, optimizer, epoch, max_epoch):
    m.train()
    corrects, train_loss = 0.0,0
    for batch in train_itr:
        text, target = batch.text, batch.label
        text = torch.transpose(text,0, 1)
        text, target = text.to(device), target.to(device)
        optimizer.zero_grad()
        logit = m(text)
        
        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()
        
        train_loss+= loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(train_itr.dataset)
    train_loss /= size 
    accuracy = 100.0 * corrects/size
  
    return train_loss, accuracy
    
def valid(m, device, test_itr):
    m.eval()
    corrects, test_loss = 0.0,0
    for batch in test_itr:
        text, target = batch.text, batch.label
        text = torch.transpose(text,0, 1)
        
        text, target = text.to(device), target.to(device)
        
        logit = m(text)
        loss = F.cross_entropy(logit, target)

        
        test_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(test_itr.dataset)
    test_loss /= size 
    accuracy = 100.0 * corrects/size
    
    return test_loss, accuracy



#%%

def main():
    
    print("Pytorch Version:", torch.__version__) 
    #Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    #%% Split whole dataset into train and valid set
    data_csv = "{}/{}".format(args.root,args.csv)
    train_csv = "{}/{}".format(args.root,'train.csv')
    val_csv = "{}/{}".format(args.root,'val.csv')
    
    
    dataset.split_train_valid(data_csv, train_csv, val_csv, 0.8)
        
        
    pretrained=args.pretrained
    
    imdb = dataset.IMDB_Dataset(mbsize=args.batch_size,
                       path_train = train_csv,
                       path_valid = train_csv,pretrained=pretrained,fix_length=args.fix_length)
    trainset, validset, vocab = imdb.tabular_train,imdb.tabular_valid,imdb.TEXT.vocab
    
    #%%Show some example to show the dataset
    print("Show some examples from train/valid..")
    print(trainset[0].text,  trainset[0].label)
    print(validset[0].text,  validset[0].label)
    
    train_iter, valid_iter = imdb.train_iter, imdb.valid_iter 
    name = ".".join(pretrained.split('.')[:2])
    dim = int(pretrained.split('.')[-1].replace('d',''))

    #%%Create
    kernels = [int(x) for x in args.kernel_height.split(',')]
    m = models.textCNN(vocab, args.out_channel, kernels, args.dropout , args.num_class).to(device)
  
    print(m)    
        
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    best_test_acc = -1
    
    #optimizer
    
    optimizer = optim.SGD(m.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90, 120], gamma=0.1)
    
    for epoch in range(1, args.epochs+1):
        #train loss
        tr_loss, tr_acc = train(m, device, train_iter, optimizer, epoch, args.epochs)
        print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))
        
        ts_loss, ts_acc = valid(m, device, valid_iter)
        print('Valid Epoch: {} \t Loss: {:4.2f} \t Accuracy: {}%'.format(epoch, ts_loss, ts_acc))
        
        scheduler.step()
        
        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            
            print("model saves at {:4.2f} % accuracy".format(best_test_acc))
            pth = {}
            dataset_name = args.csv.replace('.csv','')
            pth['epoch'] = epoch
            pth['dataset'] = dataset_name
            pth['state_dict'] = m.state_dict()
            pth['best_test_acc'] = best_test_acc
            pth['pretrained']=args.pretrained
            
            torch.save(pth, "{}/{}_{}_text_cnn_best.pth".format(args.root,dataset_name,args.pretrained))
            
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(ts_loss)
        test_acc.append(ts_acc)
    
if __name__ == '__main__':
    main()
