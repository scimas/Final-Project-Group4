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
