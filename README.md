# mAP Calculator - PascalVOC2012 and COCO Standards #

This folder contains the script to calculate the mAP on COCO/Pascal VOC 2012 Standards. There are three modes of calculation supported as of now. i) Continues Point AP ii) 11-Point AP. iii) 101-Point AP. Continuous point AP is preferred as it is more standardized and accurate. 

Usage of the same script is mentioned below

___

### Pre-requisites before running the evaluation script ###

- #### Installing the requirements
`python = 3.X` 

`pip3 install -r requirements.txt`

- #### Folder containing Ground Truths files(Text Files). 

Each file will be containing annotations following the format: **[Class_ID TLX  TLy  BRx BRy**].
For example, there can be a file name `frame_001.txt` containing the following lines.

```
frame_001.txt
0 100. 200. 221.2 250.3
0 120.3 150.6 200.1 200.9
```

- #### Folder containing Detection files(Text Files).

`NOTE: WHENEVER YOU GENERATE DETECTION FILES, DUMP THE RESULTS AT A VERY LOW CONFIDENCE THRESHOLD AROUND 0.001. THIS WILL ONLY GIVE YOU THE RIGHT mAP NUMBERS`

Each file will be containing detection results following the format:  **[Class_ID Conf TLx TLy BRx BRy] **
For example, there can be a filename frame_001.txt containing the following lines.
```
frame_001.txt
0 0.95 98 198.3 210.3 239.6
```

- #### File `model.names` that contains the names of the classes - index wise. 

For example if the detector model is inferred on two classes say person and car, then **model.names** file will follow below given format.
```
model.names
person
car
```

NOTE: 
Please specify according to the class index that is being used in the detection files. 
Here, class person corresponds to index 0 as it is in the first line of model.names file and so on.

___

### Running the script ###

You can run the script by passing command line arguments with the following given switches. 
After running the command, log file `evaluationLog.txt` will be generated containing the AP results.

`$python mAP.py [-h] [-d DET] [-g GT] [-i IOU] [-p POINTS]
                           [-n NAMES] [-c CONFIDENCE]`

```
Arguments:

  -d DET, --det DET     Full path to detection results folder
  -g GT, --gt GT        Full path to ground truth folder
  -i IOU, --iou IOU     Calculate AP at a particular IoU
  -p POINTS, --points POINTS
                        Interpolation value: 0: Continues / 11: PascalVOC2012 Challenge
  -n NAMES, --names NAMES
                        Full path of file containing names of classes index wise
  -o OUTPUT, --output OUTPUT
                        File to dump the output
  -c CONFIDENCE, --confidence CONFIDENCE
                        Confidence at which Precision/Recall is calculated
  -ig, --ignore         Flag to ignore the difficult anotations
  --coco                COCO Standard of calculation

```

Here in switch will contain the name of the model which you want to specify.

`-c` switch is used to calculate Precision and Recall at a mentioned confidence. Default is 0.5

`-n` switch is the path to the model.names file. Default is model.names.

`-ig` switch helps to remove the difficult annotations if ON. Remember, the format of the file should be `classID, Diff(0/1), Tx, TLy, BRx, BRy`

`-o` switch is the path to dump the results. After running the script, results for every class and for every IoU will get dumped in `map_log.json`(Default). 

`Demo Command:  $ python3 mAP.py -g "$path/to/gt/files" -d "$path/to/detection/files" -i 0.5 -n coco.names -p 0 -c 0.5`

You can expect the output something like given below on VOC standards

```
+---------------+-------+-------+------+-------+-----------+--------+-------+--------+
| Class/IoU@0.5 |   AP  |   TP  |  FP  |   FN  | Precision | Recall |  GTs  |  Dets  |
+---------------+-------+-------+------+-------+-----------+--------+-------+--------+
|     Person    | 0.609 |  4271 | 338  |  5307 |   0.927   | 0.446  |  9578 | 409501 |
|    Bicycle    | 0.330 |  466  | 430  |  1437 |   0.520   | 0.245  |  1903 | 66510  |
|      Car      | 0.639 | 12138 | 842  | 24718 |   0.935   | 0.329  | 36856 | 408778 |
|   Motorcycle  | 0.221 |   35  | 292  |   53  |   0.107   | 0.398  |   88  | 23424  |
|      Bus      | 0.127 |  100  | 1002 |  119  |   0.091   | 0.457  |  219  | 21520  |
|     Truck     | 0.198 |  725  | 1875 |  5277 |   0.279   | 0.121  |  6002 | 60693  |
|    Backpack   | 0.183 |  136  | 136  |  845  |   0.500   | 0.139  |  981  | 15229  |
|    Handbag    | 0.210 |  125  | 140  |  1081 |   0.472   | 0.104  |  1206 | 26157  |
|    Suitcase   | 0.133 |   58  | 318  |  147  |   0.154   | 0.283  |  205  | 95856  |
+---------------+-------+-------+------+-------+-----------+--------+-------+--------+
mAP : 29.435 
```

___

### References ###

I have referred to the official matlab code of [mAP calculation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit) and excellent written explanation and code by [@rafaelpadilla](https://github.com/rafaelpadilla/Object-Detection-Metrics) to understand the mAP calculation and extend it to the COCO standards. 
___

`TODO`: COCO small/medium/large AP calculation is in progress. Will be updated soon.
___

