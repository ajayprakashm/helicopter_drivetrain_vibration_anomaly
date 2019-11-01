# Anomaly detection in Helicopter Drivetrain vibration
This dataset is downloaded from https://web.archive.org/web/20150226050451/http://qsun.eng.ua.edu:80/cpw_web_new/data.htm. Here, IGBOutput7.mat, IGBOutput21.mat and IGBOutput34.mat are used to identify 'Normal','Slightly Faulty' and 'Severly Faulty'.

Dataset consists of helicopter drivetrain vibration data.

## Objective
To classify data into normal, slightly faulty and severly faulty. 

## Prerequisites
Install following packages:
```
pip install numpy
pip install pandas
pip install sklearn
pip install keras
pip install seaborn
```
## Python Algorithm
This algorithm is written in python language. Keras deep learning autoencoder model is used to separte into three classes.
