import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import sys
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pdb
import random
from random import shuffle
import argparse
import copy

parser = argparse.ArgumentParser(description='DA tagging')
parser.add_argument('--nEpoch', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--rSeed', type=int, default=100,
                    help='random seed')
parser.add_argument('--nClasses', type=int, default=43,
                    help='number of classes')
parser.add_argument('--bSize', type=int, default=30,
                    help='batch size')
parser.add_argument('--nLayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--shuffle', type=bool, default=True,
                    help='shuffling of data before each epoch')
parser.add_argument('--init', type=bool, default=True,
                    help='modified initialization or the default one')
parser.add_argument('--lr', type=float, default=0.0015,
                    help='initial learning rate')
parser.add_argument('--optim', type=str, default='Adam',
                    help='type of optimizer (SGD, Adagrad, Adam, RMSprop)')
parser.add_argument('--initM', type=str, default='xavier_normal_',
                    help='type of optimizer (xavier_uniform_,xavier_normal_,uniform_, \
                    normal_,constant_,ones_,zeros_)')
parser.add_argument('--mode', type=str, default='train',
                    help='mode of operation (dataproc or train)')
parser.add_argument('--nl', type=str, default='relu',
                    help='non linearity (relu, sigmoid, tanh')
parser.add_argument('--hiddenSize', type=int, default=256,
                    help='number of hidden units')
parser.add_argument('--lrs', type=float, default=1,
                    help='learning rate scheduling')
# for debugging purpose
parser.add_argument('--overfit', type=bool, default=False,
                    help='running an overfitting experiment')
parser.add_argument('--debug', type=bool, default=False,
                    help='printing outputs')
args = parser.parse_args()

rSeed = args.rSeed
random.seed(rSeed)
torch.manual_seed(rSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(rSeed)

if args.mode=='dataproc':
    def getData(dataSplit='train',vocab=None,tag2label=None,tag2freq=None,thresh=2000):
        dataSize = 0
        dataList = open('dataSplit/fromPaper/dataset_'+dataSplit+'.txt').readlines()
        if(not vocab): 
            vocabDict = {}
            for line in dataList:
                index = [] # bag of words feature for this utterance
                words = line.split('\t')[1].split(' ') # list of words
                for word in words:
                    if(word not in vocabDict): vocabDict[word] = 0
                    vocabDict[word] += 1
            # retaining the top 'thresh' frequent words in the vocabulary
            # sort the dict according to values
            vocabDictSorted = sorted(vocabDict,key=vocabDict.get,reverse=True)
            # retain top n keys
            vocab = vocabDictSorted[:thresh]
            vocab.insert(0,'UNK')

            outfile = open('vocab.txt','w')
            for word in vocab:
                outfile.write(word+'\n')
            outfile.close()

        if(not tag2label): tag2label = {}
        labelId = 0
        indices = [] # bag of words features
        labels = []
        for line in dataList:
            index = [] # bag of words feature for this utterance
            words = line.split('\t')[1].split(' ') # list of words
            tag = line.split('\t')[2]
            if(tag=='statement' or tag=='backchannel'):
            # if(tag=='statement'): pass 
            # else:
                if(dataSplit=='train' and tag not in tag2label):
                    tag2label[tag] = labelId
                    labelId += 1
                    # labelWeights.append(tag2freq[tag])
                labels.append(tag2label[tag])
                for word in words:
                    try: index.append(vocab.index(word))
                    except: index.append(0) # for 'UNK'
                indices.append(index)
                dataSize += 1

        data = np.zeros([dataSize,len(vocab)])
        dataTensor = [None]*dataSize

        for i in range(dataSize):
            data[i][indices[i]] = 1
            dataTensor[i] = torch.tensor(data[i],dtype=torch.float)

        if(dataSplit=='train'):
            # outfile = open('labelWeights.txt','w')
            # for weight in labelWeights:
            #     outfile.write(weight+'\n')
            # outfile.close()
            return dataTensor, labels, vocab, tag2label
        else:
            return dataTensor, labels

    def getFreqDict():
        freqFile = open('dataStats.csv','r')
        tag2freq = {}
        for line in freqFile:
            tag = line.split(',')[0]
            freq = line.split(',')[1]
            tag2freq[tag] = freq

        return tag2freq

    print('Reading data')
    tag2freq = getFreqDict()
    trainDataTensor, trainLabels, vocab, tag2label = getData('train',tag2freq=tag2freq)
    devDataTensor, devLabels = getData('dev',vocab,tag2label)
    testDataTensor, testLabels = getData('test',vocab,tag2label) 
    print(tag2label)

    print('Saving data')
    torch.save(trainDataTensor, 'trainData.pt')
    torch.save(trainLabels, 'trainLabels.pt')

    torch.save(devDataTensor, 'devData.pt')
    torch.save(devLabels, 'devLabels.pt')

    torch.save(testDataTensor, 'testData.pt')
    torch.save(testLabels, 'testLabels.pt')


elif args.mode=='train':

    trainData = torch.load('trainData.pt')
    # pdb.set_trace()
    trainData = torch.stack(trainData)
    trainLabels = torch.load('trainLabels.pt')
    batchSize = args.bSize
    if(args.overfit):
    # debugging experimentation: overfits?
        trainData = trainData[:100]
        trainLabels = trainLabels[:100]
        batchSize = 2
    devData = torch.stack(torch.load('devData.pt'))
    devLabels = torch.load('devLabels.pt')
    testData = torch.stack(torch.load('testData.pt'))
    testLabels = torch.load('testLabels.pt')

    inSize = len(open('vocab.txt','rb').readlines())

    nClasses = args.nClasses
    nHidden = args.hiddenSize
    nl = args.nl
    initialization = args.init
    # weights = torch.zeros(nClasses)
    # weightFile = open('labelWeights.txt','r').readlines()
    # for i in range(len(weightFile)):
    #     weights[i] = (1-float(weightFile[i].strip('\n')))/(nClasses-1)

    class Net(nn.Module):
        def __init__(self,nClasses,inputSize,hiddenSize,nLayers=1,nl='relu',init=False,initMethod='xavier_uniform_'):
            super(Net, self).__init__()

            # self.embed = nn.Embedding(inputSize,hiddenSize1)
            self.nl = nl
            self.nLayers = nLayers
            self.affineIn = nn.Linear(inputSize,hiddenSize)
            self.affineOut = nn.Linear(hiddenSize,nClasses)
            self.hiddenLayers = nn.Sequential()
            self.dropout = torch.nn.Dropout(p=0.25)
            for i in range(nLayers):
                self.hiddenLayers.add_module('layer'+str(i), nn.Linear(hiddenSize, hiddenSize))
            if(init):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        # nn.init.xavier_normal_(m.weight, 1)
                        getattr(nn.init,initMethod)(m.weight,1)
                        nn.init.constant_(m.bias, 0)

        def forward(self,x):
            out = getattr(F, self.nl)(self.dropout(self.affineIn(x)))
            out = getattr(F, self.nl)(self.hiddenLayers.forward(out))
            out = getattr(F, self.nl)(self.affineOut(out))

            return out

    lrs = args.lrs
    lr = args.lr
    ff = Net(nClasses,inSize,nHidden,nLayers=args.nLayers,nl=nl,init=initialization,initMethod=args.initM).cuda()
    # optimizer = getattr(optim, args.optim)(ff.parameters(),lr=lr)
    # optimizer = optim.SGD(ff.parameters(),lr=0.1)
    # optimizer = optim.Adagrad(ff.parameters(),lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(size_average=False).cuda()

    nEpoch = args.nEpoch
    iterNo = 0
    shuffle = args.shuffle
    trainLoss = []
    devLoss = []
    trainAcc = []
    devAcc = []
    weightNormDiff = []

    def validationStep(ff,devAcc,devLoss):

        totalDev = len(devData)
        correctDev = 0
        lossTotDev = 0
        for i in range(totalDev):
            input = devData[i].cuda()
            output = ff(input)
            lossTotDev += criterion(output.view(1,nClasses), torch.tensor([devLabels[i]]).cuda()).detach()
            pred = np.argmax(output.cpu().detach().numpy())
            if(pred==devLabels[i]): correctDev += 1
        print("Val acc: ", correctDev/float(totalDev))
        devAcc.append(correctDev/float(totalDev))
        devLoss.append(lossTotDev/float(totalDev))

        return devAcc, devLoss


    ff.train()
    # print(ff.affineIn.weight)
    while(iterNo<nEpoch):
        print("Epoch no: ", iterNo)
        # print(ff.training)
        # total = len(trainData)
        # print(trainData[0:batchSize])
        optimizer = getattr(optim, args.optim)(ff.parameters(),lr=lr)
        lr = lr*lrs
        if(shuffle):
            temp = list(zip(trainData,trainLabels))
            random.shuffle(temp)
            [trainData1,trainLabels1] = zip(*temp)
            trainData1 = torch.stack(trainData1)
        else:
            trainData1 = copy.deepcopy(trainData)
            trainLabels1 = copy.deepcopy(trainLabels)
        # print(trainData1[0:batchSize])
        total = 0
        correct = 0
        lossTot = 0
        nBatches = len(trainData1)//batchSize
        startIdx = 0
        for batchIdx in range(nBatches):
            optimizer.zero_grad()
            total += batchSize
            start, end = batchIdx*batchSize, (batchIdx+1)*batchSize
            data, target = trainData1[start:end].cuda(), torch.tensor(trainLabels1[start:end]).cuda()
            # input = trainData[i]
            output = ff(data)
            output = output.view(batchSize,nClasses)
            # loss = criterion(output.view(1,37), torch.tensor([trainLabels[i]]))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if(batchIdx==0): prevStepWeight = ff.affineIn.weight.detach().cpu().numpy()
            lossTot += loss.detach()
            pred = torch.max(output, 1)[1]
            correct += torch.sum(pred==target).cpu().numpy()
        #     pred = np.argmax(output.detach().cpu().numpy())
        #     if(pred==trainLabels[i]): correct += 1
            if(((batchIdx+1)%(nBatches//4))==0):
                # print(np.count_nonzero(list(ff.parameters())[0].grad.detach().cpu().numpy()))
                print("Train acc: ", correct/float(total))
                # print(ff.affineIn.weights.grad)
                if(args.debug):
                    print(target)
                    print(pred)
                weightNormDiff.append(np.sqrt(np.sum((ff.affineIn.weight.detach().cpu().numpy() - prevStepWeight)**2)))
                prevStepWeight = ff.affineIn.weight.detach().cpu().numpy()
                trainAcc.append(correct/float(total))
                trainLoss.append(lossTot/float(total))

                devAcc, devLoss = validationStep(ff,devAcc,devLoss)

        iterNo += 1

    ff.eval()
    total = len(testData)
    correct = 0
    for i in range(total):
        input = testData[i].cuda()
        output = ff(input)
        pred = np.argmax(output.cpu().detach().numpy())
        if(pred==testLabels[i]): correct += 1
    print("Test acc: ", correct/float(total))

    xaxis = np.arange(1,len(trainLoss)+1)
    fig, ax1 = plt.subplots()
    ax1.plot(xaxis,trainLoss,'g',label='train loss')
    ax1.plot(xaxis,devLoss,'r',label='dev loss')
    ax1.set_xlabel('Number of quarter epochs')
    ax1.set_ylabel('Negative log-likelihood', color='g')
    plt.legend(loc=6)

    ax2 = ax1.twinx()
    ax2.plot(xaxis,trainAcc,'b',label='train Acc')
    ax2.plot(xaxis,devAcc,'k',label='dev Acc')
    ax2.set_ylabel('Accuracy', color='b')
    plt.legend(loc=7)

    fig.tight_layout()
    plt.savefig('LossCompare.png')

    plt.figure()
    plt.plot(weightNormDiff)
    plt.xlabel("number of quarter epochs")
    plt.ylabel("norm difference in learned weights")
    plt.savefig('weightNorm.png')
    # plt.show()
    # plt.close()

else:
    print("Enter a valid argument")