# -*- coding: utf-8 -*-


#Packages to be included
! pip install stellargraph
! pip install interpret
! pip install sentencepiece
! pip install Bio

import sys
import pip
import pandas as pd
import numpy as np
import networkx as nx
import random
import keras.backend as K
import joblib
import sentencepiece as spm
import urllib.parse
import urllib.request
import pickle
import math
import random

from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.utils import plot_history
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from Bio import SeqIO
from Bio.KEGG import REST
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing, model_selection
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, confusion_matrix,auc, precision_recall_curve
from sklearn.model_selection import KFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from src import data_prep as DP, biological_emb as BE, models as Model, prediction as PD, GraphSAGE as SAGE, preliminary as PRE 


test_results_log={}

def DeepPHI(training_dataset, test_dataset, exp_type, emb_model, classifier_model, freeze_training_samples, load_pretrained_predictor, load_embedding_model, load_pretrained_classifier):
    g_nx_pos, seq_dict, host_prots, pathogen_prots, n_edges = DP.load_base_graph(training_dataset)
    print("Trainin dataset has been loaded.")

    print("Test dataset: ", test_dataset.replace(".xlsx", ""))
    if freeze_training_samples:
        x_ho = list(np.load("holdout_edges.npy",allow_pickle=True))
        y_ho = list(np.load("holdout_labels.npy",allow_pickle=True))
        all_edges = list(np.load("all_edges.npy",allow_pickle=True))
        g_nx_holdout = nx.read_graphml("holdout_graph")
        with open('seq_dict_ho.pickle', 'rb') as handle:
          seq_dict_ho = pickle.load(handle)
        human_prots = load_human_prots()
    else:
        g_nx_holdout, seq_dict_ho, x_ho, y_ho, all_edges, human_prots = load_holdout(test_dataset, host_prots, pathogen_prots, seq_dict)
        np.save("holdout_edges",x_ho)
        np.save("holdout_labels",y_ho)
        np.save("all_edges",all_edges)
        nx.write_graphml(g_nx_holdout, "holdout_graph")
        with open('seq_dict_ho.pickle', 'wb') as handle:
          pickle.dump(seq_dict_ho, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Test dataset has been loaded.")

    all_seq_dict = {**seq_dict, **seq_dict_ho, **human_prots}
    all_dict = protein_embedding(emb_model, host_prots, all_seq_dict, load_embedding_model)
    print("Sequence embeddings are generated.")

    set_node_features(g_nx_pos, all_dict)
    set_node_features(g_nx_holdout, all_dict)

    g_nx_all = general_graph(g_nx_pos, g_nx_holdout)  # The training graph contains all the prots in training and test sets but only training edges.

    if freeze_training_samples:
        train_edges, train_labels, gcn_edges, gcn_labels = load_edges_from_memory()
    else:
        train_edges, train_labels, gcn_edges, gcn_labels = prep_train_edges(g_nx_pos, n_edges)

    G = StellarGraph.from_networkx(g_nx_pos, node_features="feature")
    GCN_Model = GraphSAGE_model(G, gcn_edges, gcn_labels)
    print("GraphSAGE training is done.")
    
    g_nx_all, appended_edges = preliminary_predictor(train_edges, train_labels, g_nx_all, all_dict, all_edges, load_pretrained_predictor)
    hybrid_emb_dict = GCN_embeddings(GCN_Model, g_nx_all)
    print("Hybrid embeddings are generated.")

    if exp_type == "Holdout":
        interaction_classifier(classifier_model, hybrid_emb_dict, train_edges, train_labels, x_ho, y_ho, load_pretrained_classifier)
    elif exp_type == "5_fold":
        interaction_classifier(classifier_model, hybrid_emb_dict, x_ho, y_ho, None, None, load_pretrained_classifier)

def test_embedding_methods(training_dataset, test_dataset, exp_type, emb_model, classifier_model):
  aa_emb_list = ["Doc2Vec-regular", "Doc2Vec-unified", "Doc2Vec-multi", "CTD", "PSSM"]
  freeze_training_samples = False
  for emb_type in aa_emb_list:
    DeepPHI(training_dataset, test_dataset, exp_type, emb_type, classifier_model, freeze_training_samples)
    freeze_training_samples = True

def main(training_dataset = "PHISTO_DATA.xlsx", test_dataset = "Adenoviridae.xlsx", 
         exp_type = "5_fold", emb_model = "Doc2Vec-multi", classifier_model = "GA2M", 
         freeze_training_samples = False, load_pretrained_predictor = False, load_embedding_model = False, load_pretrained_classifier = False):
  ##for testing different embedding methods
  # test_embedding_methods(training_dataset, test_dataset, exp_type, emb_model, classifier_model, freeze_training_samples, load_pretrained_predictor, load_embedding_model, load_pretrained_classifier) 
  # break

  DeepPHI(training_dataset, test_dataset, exp_type, emb_model, classifier_model, freeze_training_samples, load_pretrained_predictor, load_embedding_model, load_pretrained_classifier)

if __name__ == "__main__":
  if len(sys.argv)> 0:
    if len(sys.argv[0]) < 5:
      print("Some of the arguments are missing. Please refer to the readme file.")
    else:
      training_dataset = str(sys.argv[0])
      test_dataset = str(sys.argv[1])
      exp_type = str(sys.argv[2])
      emb_model = str(sys.argv[3])
      classifier_model = str(sys.argv[4])
      if len(sys.argv) > 5:
        freeze_training_samples = str(sys.argv[5])
      if len(sys.argv) > 6:
        load_pretrained_predictor = str(sys.argv[5])
      if len(sys.argv) > 7:
        load_embedding_model = str(sys.argv[5])
      if len(sys.argv) > 8:
        load_pretrained_classifier = str(sys.argv[5])
      main(training_dataset, test_dataset, exp_type, emb_model, classifier_model, freeze_training_samples, load_pretrained_predictor, load_embedding_model, load_pretrained_classifier)
  else:
    main()
