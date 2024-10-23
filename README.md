# SenSeqNet: A Deep Learning Framework for Cellular Senescence Detection from Protein Sequences

This repository includes source code, datasets, and models for the study:

Hanli Jiang, Li Lin, Dongliang Deng, Jianyu Ren, Xin Yang, Lubin Liu (2024) (in submission)

Codes - Jupyter Notebooks
* 1_BiLSTM.ipynb - Implementation of the BiLSTM model.
* 2_CNN_LSTM.ipynb - Combined Convolutional Neural Network and LSTM model.
* 3_CNN.ipynb - Convolutional Neural Network model.
* 4_ESM2_features.ipynb - Extracting features from the ESM2 model for protein sequences.
* 5_Figures.ipynb - Code for generating figures used in the manuscript.
* 6_LSTM.ipynb - Long Short-Term Memory (LSTM) model.
* 7_Manu_features.ipynb - Manual feature extraction and processing.
* 8_ML_models.ipynb - Collection of different machine learning models and their evaluation.
* 9_pCNNLSTM.ipynb - Parallel CNN and LSTM model architecture.
* 10_RNN.ipynb - Implementation of the Recurrent Neural Network model.
* 11_SenSeqNet.ipynb - Main notebook consolidating the SenSeqNet model and experiments.
  
Data
Data_cdHit_0.4/ - Clustering results with a threshold of 0.4.

* negative_0.4.fasta - FASTA file containing negative sequences at 0.4 similarity threshold.
* positive_0.4.fasta - FASTA file containing positive sequences at 0.4 similarity threshold.

ESM2_features/ - Contains features extracted using ESM2.

* esm_features_labels.zip - Compressed file containing extracted features and corresponding labels.
Manu_features/ - Manually extracted features for model training.

* manu_features_labels.zip - Compressed file containing manually extracted features and corresponding labels.
Original_data/ - Contains raw or unprocessed datasets used in the study.

Models
Deep_learning_models/ - Contains trained deep learning models in .pth format.

* BiLSTM_model.pth - Trained BiLSTM model weights.
* CNN_LSTM_model.pth - Trained CNN-LSTM model weights.
* CNN_model.pth - Trained CNN model weights.
* LSTM_model.pth - Trained LSTM model weights.
* Parallel_CNN_LSTM_model.pth - Trained parallel CNN-LSTM model weights.
* RNN_model.pth - Trained RNN model weights.

Machine_learning_models/ - Contains traditional machine learning models and their implementations.

* ML_models.ipynb - Code for running and evaluating machine learning models.

Contact
For inquiries, please contact Hanli Jiang at hhanli.jiang[@]mail.utoronto.ca
