# Graph Convolutional Network based Virus-Human Protein-Protein Interaction Prediction for Novel Viruses
This repository contains the code of a three-stage machine learning pipeline that generates and uses hybrid embeddings for PHI prediction. The details of this study and experimental results are published in the paper below:
## citation

## System requirements

 The code is tested on Python version >= 3.6. Following Python libraries are required:
- Pip
- Pandas
- Numpy
- Scikit-learn
- NetworkX
- Gensim
- TensorFlow / Keras
- Pickle 
- Joblib

The libraries shown below are installed automatically by the code using Pip:
- StellarGraph
- Interpret
- SentencePiece
- Bio

## Usage

######Data format

######Demo

Run **main.py** file in command line without passing no argument for testing **Adeno virus** dataset with all the proposed embedding methods. The **PHISTO** dataset is used for training in demo.

######Quick start

The model can be tested by running **main.py** file in command line or calling **DeepPHI()** function as an in the code block. Use the following command for calling in command line:

`$ python3 main.py train_dataset.xlsx test_dataset.xlsx experiment_type classifier_type`

or just calling as a function:

`DeePHI(train_dataset.xlsx, test_dataset.xlsx, experiment_type, classifier_type)`

