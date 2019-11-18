#!/bin/bash
if [ "$1" = "train" ]; then
    python Code/train.py
elif [ "$1" = "evaluate" ]; then
    python Code/evaluate.py
elif [ "$1" = "predict" ]; then
    python Code/predict.py
elif [ "$1" = "help" ]; then
    echo "Usage:"
    echo "./run.sh train     # To train the model"
    echo "./run.sh evaluate  # To evaluate the trained model on test data"
else
    echo "Invalid command. See './run.sh help' for more information"
fi
