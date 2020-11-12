## This repository contains all the Notebooks and files containing python code snippets which helped us to develop final script : dataset_generator.py

## Data-augmenation-coco-format

[Data Augmentation for Bounding Boxes.ipynb](https://github.com/asheesh1202/data-augmenation-coco-format/blob/master/Data%20Augmentation%20for%20Bounding%20Boxes.ipynb) is the main jupyter notebook for data augmentaion which contains multiple munction that can perform various augmentaion.

## Examples

|Operation| Input1 | Input 2| 
|---|---|---|
|None|![](https://i.ibb.co/XVBQWRC/frame1.jpg)|![](https://i.ibb.co/VNGvK99/frame0.jpg)|
|Horizontal Shift|![](https://i.ibb.co/gD0Z5mF/frame1-w-shifted-0.jpg)|![](https://i.ibb.co/BtTGMqB/frame4-w-shifted-3.jpg)|
|Skew Rotation|![](https://i.ibb.co/921kF7N/frame1-rotated-4.jpg)|![](https://i.ibb.co/S6pcPzT/frame0-rotated-1.jpg)|

please look into [Data Augmentation for Bounding Boxes.ipynb](https://github.com/asheesh1202/data-augmenation-coco-format/blob/master/Data%20Augmentation%20for%20Bounding%20Boxes.ipynb) for further examples.

## Input format
 Input must be provided with .jpg format along with annotation file with .xml format with same file name. The path of the folder will be be provided as below
 
 ```Python
 xml_to_csv("Path/for/Input/Folder")
 ```
 
 Output images and xml file will be generated in the generated folder `aug` inside the input folder.
 
 ## Repo also contains util scripts to convert xml to csv, extract images from vedio, xml data parsers.
 
