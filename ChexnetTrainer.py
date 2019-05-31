import os
import numpy as np
import time
import sys
import subprocess

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from DatasetGenerator import DatasetGenerator


#-------------------------------------------------------------------------------- 

class ChexnetTrainer ():

    #---- Train the densenet network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):

#        print(get_gpu_memory_map())
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda()
                
        #-------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        loss = torch.nn.CrossEntropyLoss(size_average = True)
        
        #---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        #---- TRAIN THE NETWORK

#        print(get_gpu_memory_map())
        
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            print("Training epoch " +str(epochID+ 1) + " of " + str(trMaxEpoch))
            
            torch.cuda.empty_cache()
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            torch.cuda.empty_cache()
            
            ChexnetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            print("Epoch training complete, starting epoch validation... ")
            
            torch.cuda.empty_cache()
            
            lossVal, losstensor = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            print("Epoch validation complete, recording results.")

            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime

            torch.cuda.empty_cache()
            scheduler.step(losstensor.item())

            torch.cuda.empty_cache()
            if lossVal < lossMIN:
                lossMIN = lossVal
                torch.cuda.empty_cache()
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                     
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        
        for batchID, (input, target) in enumerate (dataLoader):
            torch.cuda.empty_cache()            
            target = target.cuda(async = True)
                 
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)


            #torch.reshape(varOutput, (2 * classCount, 3)) #check syntax
            #torch.reshape(varTarget, (2 * classCount,)) #check 

            varOutput = varOutput.view(2*classCount, 3)
            varTarget = varTarget.view(2* classCount,) 
            torch.cuda.empty_cache()
            lossvalue = loss(varOutput, varTarget.long())
            torch.cuda.empty_cache()
            
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            torch.cuda.empty_cache()
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        with torch.no_grad():
            model.eval()
            lossVal = 0
            lossValNorm = 0
            
            losstensorMean = 0
            
            for i, (input, target) in enumerate (dataLoader):
                torch.cuda.empty_cache()
#                print("Loading data to start validation.")
                target = target.cuda(async=True)
                
                with torch.no_grad():
                    varInput = torch.autograd.Variable(input)
                    varTarget = torch.autograd.Variable(target)
 #                   print("Feeding data into Densenet model.")
                    varOutput = model(varInput)
  #                  print("Data passed through model successfully.")
                    varTarget = varTarget.view(2*classCount,)
                    varOutput = varOutput.view(2* classCount, 3)
                    torch.cuda.empty_cache()
                    losstensor = loss(varOutput, varTarget.long())
                    losstensorMean += losstensor
                    torch.cuda.empty_cache()
                    lossVal += losstensor.item()
                    lossValNorm += 1
                    
                    torch.cuda.empty_cache()
                    
                    outLoss = lossVal / lossValNorm
                    losstensorMean = losstensorMean / lossValNorm
                    torch.cuda.empty_cache()
                    
            return outLoss, losstensorMean
               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        torch.cuda.empty_cache()
        
        for i in range(classCount):

            try:
                    outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                    pass
            
        return outAUROC
        
        
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        CLASS_NAMES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda() 
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()

        with torch.no_grad():
            model.eval()
            
            for i, (input, target) in enumerate(dataLoaderTest):
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
            
                bs, n_crops, c, h, w = input.size()
                varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
            
                out = model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
                
                outPRED = torch.cat((outPRED, outMean.data), 0)
                
            aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
            aurocMean = np.array(aurocIndividual).mean()
                
            print ('AUROC mean ', aurocMean)
        
            for i in range (0, len(aurocIndividual)):
                print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        
     
        return
#-------------------------------------------------------------------------------- 

def get_gpu_memory_map():


    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used',
         '--format—csv,nounits,noheader'], encoding='utf-8')
    
    # Convert lines into a dictionary
    
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory = dict(zip(range(len(gpu_memory)), gpu_memory))
    print(str(gpu_memory_map))
    return gpu_memory_map



