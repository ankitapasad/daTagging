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

random.seed(100)

if sys.argv[1]=='dataproc':
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

    print('Saving data')
    torch.save(trainDataTensor, 'trainData.pt')
    torch.save(trainLabels, 'trainLabels.pt')

    torch.save(devDataTensor, 'devData.pt')
    torch.save(devLabels, 'devLabels.pt')

    torch.save(testDataTensor, 'testData.pt')
    torch.save(testLabels, 'testLabels.pt')


elif sys.argv[1]=='train':
    trainData = torch.load('trainData.pt')
    # pdb.set_trace()
    trainData = torch.stack(trainData)
    trainLabels = torch.load('trainLabels.pt')
    # debugging experimentation: overfits?
    trainData = trainData[:100]
    trainLabels = trainLabels[:100]
    devData = torch.stack(torch.load('devData.pt'))
    devLabels = torch.load('devLabels.pt')
    testData = torch.stack(torch.load('testData.pt'))
    testLabels = torch.load('testLabels.pt')

    inSize = len(open('vocab.txt','rb').readlines())

    # This set of parameters works successfully on the overfitting experiments
    nClasses = 2
    nHidden = 256
    # weights = torch.zeros(nClasses)
    # weightFile = open('labelWeights.txt','r').readlines()
    # for i in range(len(weightFile)):
    #     weights[i] = (1-float(weightFile[i].strip('\n')))/(nClasses-1)

    class Net(nn.Module):
        def __init__(self,nClasses,inputSize,hiddenSize,init=False):
            super(Net, self).__init__()

            # self.embed = nn.Embedding(inputSize,hiddenSize1)
            self.affine1 = nn.Linear(inputSize,hiddenSize)
            self.affine2 = nn.Linear(hiddenSize,int(hiddenSize/4))
            self.affine3 = nn.Linear(int(hiddenSize/4),nClasses)

            if(init):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

        def forward(self,x):
            out = F.relu(self.affine1(x))
            out = F.relu(self.affine2(out))
            out = F.relu(self.affine3(out))

            return out

    ff = Net(nClasses,inSize,nHidden,init=True).cuda()
    # optimizer = optim.Adam(ff.parameters(),lr=0.005)
    # optimizer = optim.SGD(ff.parameters(),lr=0.1)
    optimizer = optim.Adagrad(ff.parameters(),lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(size_average=False).cuda()

    nEpoch = 10
    iterNo = 0
    b = 2
    trainLoss = []
    devLoss = []
    trainAcc = []
    devAcc = []
    ff.train()
    while(iterNo<nEpoch):
        print("Epoch no: ", iterNo)
        total = len(trainData)
        correct = 0
        lossTot = 0
        nBatches = len(trainData)//b
        startIdx = 0
        for batchIdx in range(nBatches):
            start, end = batchIdx*b, (batchIdx+1)*b
            data, target = trainData[start:end].cuda(), torch.tensor(trainLabels[start:end]).cuda()
            # input = trainData[i]
            output = ff(data)
            output = output.view(b,nClasses)
            # loss = criterion(output.view(1,37), torch.tensor([trainLabels[i]]))
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lossTot += loss.detach()
            pred = torch.max(output, 1)[1]
            correct += torch.sum(pred==target).cpu().numpy()
        #     pred = np.argmax(output.detach().cpu().numpy())
        #     if(pred==trainLabels[i]): correct += 1

        print("Train acc: ", correct/float(total))
        trainAcc.append(correct/float(total))
        trainLoss.append(lossTot/float(total))


        ff.eval()
        total = len(devData)
        correct = 0
        lossTot = 0
        for i in range(total):
            input = devData[i].cuda()
            output = ff(input)
            lossTot += criterion(output.view(1,nClasses), torch.tensor([devLabels[i]]).cuda()).detach()
            pred = np.argmax(output.cpu().detach().numpy())
            if(pred==devLabels[i]): correct += 1
        print("Val acc: ", correct/float(total))
        devAcc.append(correct/float(total))
        devLoss.append(lossTot/float(total))
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
    ax1.set_xlabel('Number of epochs')
    ax1.set_ylabel('Negative log-likelihood', color='g')
    plt.legend(loc=6)

    ax2 = ax1.twinx()
    ax2.plot(xaxis,trainAcc,'b',label='train Acc')
    ax2.plot(xaxis,devAcc,'k',label='dev Acc')
    ax2.set_ylabel('Accuracy', color='b')
    plt.legend(loc=7)

    fig.tight_layout()
    plt.savefig('LossCompare_dm.png')
    # plt.show()
    # plt.close()

else:
    print("Enter a valid argument")