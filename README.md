# mAP Calculation Steps #

This folder contains the script to calculate the mAP on COCO/Pascal VOC 2012 Standards. There are two modes of calculation supported as of now. i) Continues Point AP ii) 11-Point AP. Continuous point AP is preferred as it is more standardized and accurate. 
Usage of the same script is mentioned below

___

### Pre-requisites before running the evaluation script ###

- Folder containing Ground Truths files(Text FIles). 
```
Each file will be containing annotations following the format: [Class ID TLX  TLy  BRx BRy].
For example, there can be a file name frame_001.txt containing the following lines.

frame_001.txt
0 100. 200. 221.2 250.3
0 120.3 150.6 200.1 200.9
```

`Normalized coordinates are also supported now. Refer th script usage section`

- Folder containing Detection files(Text Files).

`NOTE: WHENEVER YOU GENERATE DETECTION FILES, DUMP THE RESULTS AT A VERY LOW CONFIDENCE THRESHOLD AROUND 0.005. THIS WILL ONLY GIVE YOU THE RIGHT mAP NUMBERS`

```
Each file will be containing detection results following the format: [Class ID Conf TLx TLy BRx BRy].
For example, there can be a filename frame_001.txt containing the following lines.

frame_001.txt
0 0.95 98 198.3 210.3 239.6
```

- File `model.names` that contains the names of the classes - index wise. 
```
For example if the detector model is inferred on two classes say person and head, then model.names file will follow below given format.

model.names
person
head

NOTE: 
Please specify according to the class index that is being used in the detection files. 
Here, class person corresponds to index 0 as it is in the first line of model.names file and so on.
```

___

### Running the script ###

You can run the script by passing command line arguments with the following given switches. 
After running the command, log file `evaluationLog.txt` will be generated containing the AP results.

`$python evaluationScript.py [-h] [-d DET] [-g GT] [-i IOU] [-p POINTS]
                           [-n NAMES] [-c CONFIDENCE] [-in INSTANCE]`

```
Arguments:

  -h, --help            show this help message and exit
  -d DET, --det DET     Full path to detection results folder
  -g GT, --gt GT        Full path to ground truth folder
  -i IOU, --iou IOU     Calculate AP at a particular IoU
  -p POINTS, --points POINTS
                        Interpolation value: 0: Continues / 11: PascalVOC2012
                        Challenge
  -n NAMES, --names NAMES
                        Full path of file containing names of classes index
                        wise
  -c CONFIDENCE, --confidence CONFIDENCE
                        Confidence at which Precision/Recall is calculated
  -in INSTANCE, --instance INSTANCE
                        Name of the instance/model(Seprated by underscore)
  -ig IGNORE, --ignore IGNORE
                        Flag to ignore the difficult anotations 0/1: No/Yes
  -un UNNORMALIZE, --unnormalize UNNORMALIZE
                        Flag of Unnormalization 0/1: No/Yes
  -wi WIDTH, --width WIDTH
                        Width of the image for unnormalization
  -he HEIGHT, --height HEIGHT
                        Height of the image for unnormalization

```

Here in switch will contain the name of the model which you want to specify.

`-c` switch is used to calculate Precision and Recall at a mentioned threshold. Default is 0.5

`-n` switch is the name of the model.names file. Default is model.names. You can keep this file in the same folder as `evaluationScript.py`

`-un` switch helps to unnormalize the coordinates in YOLO format to un-normalized one. Provide width and height of the data when the flag is ON

`-ig` switch helps to remove the difficult annotations if ON. Remember, the format of the file should be `classID, Diff(0/1), Tx, TLy, BRx, BRy`
`Demo Command:  $ python evaluationScript.py -g "$path/to/gt/files" -d "$path/to/detection/files" `

___

`UPDATE: COCO mAP IOU CALCULATION 0.5:0.95 IS IN TESTING. PascalVOC2012 IS READY TO USE`

___
