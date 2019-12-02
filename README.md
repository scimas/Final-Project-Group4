# ASL Recognition
American Sign Language recognition using convolutional neural networks.

## Usage:
Make the `run.sh` script executable.
```
chmod u+x run.sh
```
To train the model:
```
./run.sh train
```
To evaluate the model:
```
./run.sh evaluate
```
To predict on real time data:
```
./run.sh predict
```
Note: You will need to install the IP Webcam app on you android smartphone to access the camera for images. Or modify the predict script according to your needs.

## Data:
The data can be obtained from [Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist).

Uncompress it into the data directory.

The directory structure should look like this:
```
root
├── Code
│   ├── evaluate.py
│   ├── model
│   │   └── 
│   ├── modelling.py
│   ├── preprocess.py
│   └── train.py
├── data
│   ├── american_sign_language.PNG
│   ├── amer_sign2.png
│   ├── amer_sign3.png
│   ├── sign_mnist_test.csv
│   └── sign_mnist_train.csv
├── Group-Proposal
│   └── Final Project Proposal.pdf
├── README.md
└── run.sh
```
