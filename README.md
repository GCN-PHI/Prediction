# Graph Convolutional Network based Virus-Human Protein-Protein Interaction Prediction for Novel Viruses
This repository contains the code of a three-stage machine learning pipeline that generates and uses hybrid embeddings for PHI prediction. The details of this study and the experimental results are published in the paper below:
## citation

## System requirements

 The code is tested on Python version >= 3.6. Following Python libraries are required:
- Pip3
- Pandas
- Numpy
- Scikit-learn
- NetworkX
- Gensim
- TensorFlow / Keras
- Pickle 
- Joblib

The libraries shown below are installed automatically by the code using Pip3:
- StellarGraph
- Interpret
- SentencePiece
- Bio

## Usage

**Data format**

The interactions are given in `Excel (.xlsx)` file. The file is in a simple format:

| HOST  | VIRUS | HOST_SEQ | VIRUS_SEQ |
| ------------- | ------------- | ------------- | ------------- |
| P13861 | P03259 | MSHIQ... | MRTEM... |

where the **HOST** is the unique id of a host protein and **VIRUS** is the unique id of the interacting virus protein. **HOST_SEQ** and **VIRUS_SEQ** are the aminoacid sequences of the host and virus proteins, respectively. Please check ***Data/Adenoviridae.xlsx*** for an example file.


**Demo**

Run **main.py** file in command line without passing no argument for testing **Adeno virus** dataset with all the proposed embedding methods. The **PHISTO** dataset is used for training in demo.

**Quick start**

The model can be tested by running **main.py** file in command line or calling **DeepPHI()** function in the code block. Use the following command for calling in Terminal:

`$ python3 main.py train_dataset_path test_dataset_path experiment_type classifier_type fts, lpp, lem, lpc`

or just calling as a function:

`DeePHI(train_dataset_path, test_dataset_path, experiment_type, classifier_type, fts, lpp, lem, lpc)`

| Parameter  | Description |
| ------------- | ------------- |
| train_dataset_path  | The path of the interaction dataset used for training GraphSAGE, preliminary classifier and final classifier (in holdout experiments).    |
| test_dataset_path  | The path of the test interaction dataset.  |
| experiment_type  | Type of the experiment. ***5_fold*** and ***Holdout*** are valid types.  |
| classifier_type  | Type of the final interaction classifier. ***SVM***, ***NN***, ***RF***, ***LR***, and ***GA2M*** are the available methods. |
| fts  | **(optional, default = False)** Pass ***True*** for freezing sampled negative edges between experiments.  |
| lpp  | **(optional, default = False)** Pass ***True*** for loading pre-trained preliminary classifier model instead of training a new one.  |
| lem  | **(optional, default = False)** Pass ***True*** for loading pre-trained biological feature embedding model instead of training a new one.  |
| lpc  | **(optional, default = False)** Pass ***True*** for loading pre-trained interaction classifier model instead of training a new one.  |


