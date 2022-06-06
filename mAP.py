
'''
Author: @shreejalt

Date: 12-04-2022

Description: Script to calculate mAP on PascalVOC2012 and COCO standards

License: Open to use for all. Reference taken from: https://github.com/rafaelpadilla/Object-Detection-Metrics

'''
import argparse
import numpy as np
import os
from prettytable import PrettyTable
from tqdm import tqdm
from collections import defaultdict
import json

class MAP:

    def __init__(
        self,
        detPath,
        gtPath,
        iou=0.5,
        points=0,
        names='model.names',
        confidence=0.5,
        cocoFlag=False,
        logName='results_map.json',
        ignoreDiff=False
    ):
      
        self.detPath = detPath
        self.gtPath = gtPath
        self.logName = logName
        self.cocoFlag = cocoFlag
        self.ignoreDiff = ignoreDiff

        assert(iou > 0.0 and iou <= 1.0)
        assert (points in [0, 11, 101])

        self.iou = iou
        self.points = points
        self.confidence = confidence

        with open(names, 'r') as f:
            classes = f.readlines()
            self.classes = {idx: line.strip() for idx, line in enumerate(classes)}
        self.numClasses = len(self.classes.keys())
        
        print('Total number of classes specified in %s file: %d' % (names, len(self.classes.keys())))

        self.gtBoxes, self.detBoxes, self.detScores = self.preprocessData()

        print('Total Files : %d' % (len(self.gtBoxes[0].keys())))


    def preprocessData(self):

        finalGTFiles = list()
        gtBoxes = [{} for _ in range(self.numClasses)]
        gtFiles = os.listdir(self.gtPath)

        for filename in gtFiles:
        
            anns = np.loadtxt(os.path.join(self.gtPath, filename)).astype(int)
            anns = anns if anns.ndim > 1 else anns.reshape(1, -1)
        
            if self.ignoreDiff: # Added difficulty flah support 
                anns = anns[np.where(anns[:, 1] == 0)[0]]
            anns = np.delete(anns, 1, 1) if anns.shape[1] == 6 else anns

            if anns.shape[0] > 0:
                for i in range(self.numClasses):
                    dict_temp = {}
                    dict_temp[filename] = anns[np.where(anns[:, 0] == i)][:, 1:]
                    
                    gtBoxes[i].update(dict_temp)
                finalGTFiles.append(filename)

        #Make Dictionary of Detections
        detBoxes = [{} for _ in range(self.numClasses)]
        detScores = [{} for _ in range(self.numClasses)] 
        
        for filename in finalGTFiles:
            
            if os.path.getsize(os.path.join(self.detPath, filename)) <= 1:
                anns = np.array([])
            else:
                anns = np.loadtxt(os.path.join(self.detPath, filename))
        
            if anns.shape[0] == 0:
                for i in range(self.numClasses):
                    dict_temp = {}
                    dict_temp[filename] = np.array([])
                    detBoxes[i].update(dict_temp)
                    detScores[i].update(dict_temp)
            else:
                anns = anns if anns.ndim > 1 else anns.reshape(1, -1)
                for i in range(self.numClasses):
                    dict_temp_bbox, dict_temp_score = {}, {}
                    dict_temp_bbox[filename] = anns[np.where(anns[:, 0] == i)][:, 2:]
                    dict_temp_score[filename] = np.round(anns[np.where(anns[:, 0] == i)][:, 1], 2)
                    detBoxes[i].update(dict_temp_bbox)
                    detScores[i].update(dict_temp_score)   
        
        print("Processed Detections..!\n")

        return gtBoxes, detBoxes, detScores
    

    def calcmAP(self):

        if self.cocoFlag:
            print('Calculating on COCO standards: IoU[0.5,...,0.95] | Points: 101')
            self.points = 101
        else:
            print('Calculating on VOC Standards IoU: 0.5')
        IoUList = np.arange(0.5, 0.95, 0.05) if self.cocoFlag else [self.iou]
        classmAPDict = defaultdict(dict)
        APList = [[] for _ in range(self.numClasses)]

        for iou in IoUList:
            for i in tqdm(range(self.numClasses)):
                dataAP = self.calAPClass(self.gtBoxes[i], self.detBoxes[i], self.detScores[i], IoU=iou)
                classmAPDict[self.classes[i]][iou] = dataAP
                APList[i].append(dataAP['AP'])
    
        mAPList = np.array([np.mean(np.array(APList[i])) for i in range(self.numClasses)])

        # Print the statistics of the AP@0.5
        table = PrettyTable(['Class/IoU@0.5', 'AP', 'TP', 'FP', 'FN', 'Precision', 'Recall'])
        for cls in classmAPDict.keys():

            table.add_row(
                [
                    cls,
                    '%.3f' % classmAPDict[cls][0.5]['AP'],
                    '%.3f' % classmAPDict[cls][0.5]['TP'],
                    '%.3f' % classmAPDict[cls][0.5]['FP'],
                    '%.3f' % classmAPDict[cls][0.5]['FN'],
                    '%.3f' % classmAPDict[cls][0.5]['Precision'],
                    '%.3f' % classmAPDict[cls][0.5]['Recall']
                ]
            )
        print(table)
        if self.cocoFlag:
            headingTable = ['class/IoU'] + ["%.2f" % iou for iou in IoUList]
            valueTable = list()
            for i in range(self.numClasses):
                valueTable.append([self.classes[i]] + ["%.2f" % ap for ap in APList[i]])

            cocoTable = PrettyTable(headingTable)
            cocoTable.add_rows(valueTable)
            print(cocoTable)

        classmAPDict['mAP'] = np.mean(mAPList)
        print('mAP : %.3f ' % (np.mean(mAPList) * 100))
        results = json.dumps(classmAPDict, indent=4)

        with open(self.logName, 'w') as f:
            f.write(results)
    
    def calAPClass(self, gtBoxes, detBoxes, detScores, IoU=0.5):

        '''
        Input: Ground Truth and Detection Boxes
            Format: [TLX, TLy, BRx, BRy] GT BOXES Dictionary | bbox1 ::: NX4, DETECTION BOXES Dictionary | bbox2 ::: MX4, DETECTION SCORES Dictionary
                    IoU

        Output: Calculate AP of a particular class

        Description: Calculate mask of boxes i.e calculate TPs and FPs
        '''

        maskBoxes = self.getMaskBoxes(gtBoxes, detBoxes, IoU)

        totalGTs = sum([x.shape[0] for _, x in gtBoxes.items()])
        totalDets = sum([x.shape[0] for _, x in detBoxes.items()])

        scoresList = list()
        masksList = list()

        for imgID in detBoxes.keys():
            scoresList.append(detScores[imgID].tolist())
            masksList.append(maskBoxes[imgID].tolist())

        scoresList = np.array(sum(scoresList, []))
        masksList = np.array(sum(masksList, []))

        #Check assertion of masks and scores
        assert(len(scoresList) == len(masksList))

        sortedIdxs = np.argsort(-scoresList)
        scoresList = scoresList[sortedIdxs]
        masksList = masksList[sortedIdxs]
        
        truePostive = np.cumsum(masksList)
        falsePositive = np.cumsum(1 - masksList)

        TP = np.interp(-self.confidence, -scoresList, truePostive)
        FP = np.interp(-self.confidence, -scoresList, falsePositive)
        FN = totalGTs - TP
    
        precision = truePostive / (truePostive + falsePositive + 1e-7)
        recall = truePostive / (totalGTs + 1e-7)
        
        preConf = np.interp(-self.confidence, -scoresList, precision)
        recConf = np.interp(-self.confidence, -scoresList, recall)

        averagePrecision = self.calAveragePrecision(precision, recall)
        
        return {
            'AP': averagePrecision,
            'Precision': preConf,
            'Recall': recConf,
            'PrecisionList': precision.tolist(),
            'RecallList': recall.tolist(),
            'totalGTs': totalGTs,
            'totalDets': totalDets,
            'TP': TP,
            'FP': FP,
            'FN': FN
        }


    def calAveragePrecision(self, precision, recall):

        '''
        Input: List of Precision/ Recall | Points = 0/11/101

        Output: Average Precision 

        Description: Calculate Precision and Recall based on PR Curve

        '''

        mprecision = np.concatenate(([0.], precision, [0.]))
        mrecall = np.concatenate(([0.], recall, [1.]))
        mprecision = np.flip(np.maximum.accumulate(np.flip(mprecision)))

        if self.points == 0: # Continues interpolation
            idxs = np.where(mrecall[1:] != mrecall[:-1])[0]
            averagePrecision = np.sum((mrecall[idxs + 1] - mrecall[idxs]) * mprecision[idxs + 1])

        elif self.points == 11: # 11 points sampling interpolation Pascal VOC 2012
            idxs = np.linspace(0, 1, 11)  
            averagePrecision = np.trapz(np.interp(idxs, mrecall, mprecision), idxs) 
        
        elif self.points == 101: # 101 points sampling interpolation COCO
            idxs = np.linspace(0, 1, 101)  
            averagePrecision = np.trapz(np.interp(idxs, mrecall, mprecision), idxs) 
        
        return averagePrecision


    def getMaskBoxes(self, gtBoxes, detBoxes, IoU): 

        '''
        Input: Ground Truth and Detection Boxes
            Format: [TLX, TLy, BRx, BRy] GT BOXES Dictionary | bbox1 ::: NX4, DETECTION BOXES Dictionary | bbox2 ::: MX4, DETECTION SCORES Dictionary

        Output: Dictionary mask of detections ::: TP - True  / FP - False | Key ::: Image Name | Value ::: Mask Array

        Description: Calculate mask of boxes i.e calculate TPs and FPs

        '''
        
        maskBoxes = dict()
        for imgID in detBoxes.keys():
            gtBox = gtBoxes[imgID]
            if detBoxes[imgID].shape[0] > 0:
                detBox = detBoxes[imgID]
                maskBoxes[imgID] = self.getmAPSingle(gtBox, detBox, IoU)
            else:
                maskBoxes[imgID] = np.array([])
        return maskBoxes


    def getmAPSingle(self, gtBox, detBox, IoU):

        '''
        Input: Ground Truth and Detection Boxes
            Format: [TLX, TLy, BRx, BRy] GT BOXES Shape | bbox1 ::: NX4, DETECTION BOXES Shape | bbox2 ::: MX4 

        Output: Mask of Detections ::: TP - True  / FP - False

        Description: Calculate mask of boxes i.e calculate TPs and FPs

        '''

        detMask = np.zeros(detBox.shape[0], dtype=bool)
        
        if gtBox.shape[0] == 0:
            return detMask      
        iou = self.vecIoU(gtBox, detBox)
        iouValid = iou[np.where(iou > IoU)]
        sortiouValid = np.argsort(iouValid)[::-1]
        gtIdx, detIdx = np.where(iou > IoU)
    
        gtIdxValid, detIdxValid = list(), list()
        for idx in sortiouValid:
            if (gtIdx[idx] not in gtIdxValid) and (detIdx[idx] not in detIdxValid):
                gtIdxValid.append(gtIdx[idx])
                detIdxValid.append(detIdx[idx])
                detMask[detIdx[idx]] = 1
    
        return detMask

    def vecIoU(self, bbox1, bbox2):
        '''
        Input: Ground Truth and Detection Boxes
            Format: [TLX, TLy, BRx, BRy] GT BOXES Shape | bbox1 ::: NX4, DETECTION BOXES Shape | bbox2 ::: MX4 

        Output: IoU Matrix Shape | NXM

        Description: Calculate vectorized IoU between ground truths and detections.
        '''

        x11, y11, x12, y12 = np.split(bbox1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bbox2, 4, axis=1)

        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))        

        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)        
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)        
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)        
        return iou

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--det", type=str, default="detection-results", help="Full path to detection results folder")
    parser.add_argument("-g", "--gt", type=str, default="ground-truths", help="Full path to ground truth folder")
    parser.add_argument("-i", "--iou", type=float, default=0.5, help='Calculate AP at a particular IoU')
    parser.add_argument("-p", "--points", type=int, default=0, help='Interpolation value: 0: Continues / 11: PascalVOC2012 Challenge')
    parser.add_argument("-n", "--names", default="model.names", help="Full path of file containing names of classes index wise")
    parser.add_argument("-o", "--output", default='map_log.json', type=str, help="File to dump the output")

    parser.add_argument("-c", "--confidence", default=0.5, type=float, help="Confidence at which Precision/Recall is calculated")
    parser.add_argument("-ig", "--ignore", action='store_true', help="Flag to ignore the difficult anotations")
    parser.add_argument('--coco', action='store_true', help='COCO Standard of calculation')
    args = parser.parse_args()
    calculator = MAP(
        detPath=args.det,
        gtPath=args.gt,
        iou=args.iou,
        points=args.points,
        names=args.names,
        confidence=args.confidence,
        cocoFlag=args.coco,
        logName=args.output,
        ignoreDiff=args.ignore
    )
    
    calculator.calcmAP()








    