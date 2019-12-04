# ASL Recognition
American Sign Language recognition using convolutional neural networks.

## Usage:
Make the `run.sh` script executable.
```
chmod u+x run.sh
```
The run script assumes python 3.x is being used and it is accessible as `python`. If not, change all instances of `python` to `python3` in the run script.
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
Note: You will need to install the IP Webcam [link](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en_US) app or other similar application on your android smartphone to access the camera for images. Or modify the predict script according to your needs.

## Data:
The original data was obtained from [Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist).
We have included out modified data in the repository.

Uncompress it into the data directory.

The directory structure should look like this:
```
root
├── Code
│   ├── evaluate.py
│   ├── model
│   │   └── 
│   ├── modelling.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── data
│   └── augmented.csv
├── Group-Proposal
│   └── Final Project Proposal.pdf
├── ishita-jain-individual-project
│   ├── Code
│   │   └── train1-ij.py
│   └── Individual-Final-Project-Report
│       └── ishita-jain-final-report.pdf
├── Jyothi-Ganji-individual-project
│   ├── Code
│   │   └── mywork.py
│   └── Individual-Final-Project-Report
│       └── Jyothi-Ganji-final-project.pdf
├── Final-Group-Presentation
│   └── final presentation.pdf
├── Final-Group-Project-Report
│   └── FINAL REPORT.pdf
├── Mihir-Gadgil-individual-project
│   └── Individual-Final-Project-Report
│       └── Mihir-Gadgil-final-project.pdf
├── README.md
└── run.sh
```
