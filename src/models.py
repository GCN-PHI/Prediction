
# Classifier and helper methods

def weighted_binary_crossentropy(y_true, y_pred):
  weights = (y_true * 9.) + 1.
  bce = K.binary_crossentropy(y_true, y_pred)
  weighted_bce = K.mean(bce * weights)
  return weighted_bce

def calc_metric(metric, y_true, y_pred):
    if metric == "accuracy":
        return accuracy_score(y_pred,y_true)
    elif metric == "Macro F1-score":
        return f1_score(y_pred, y_true, average='macro')
    elif metric == "AUC-PR":
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        return auc(recall, precision)
    elif metric == "AUC-ROC":
        return roc_auc_score(y_true, y_pred)
    elif metric == "Precision":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        return tp / (tp + fp)
    elif metric == "Specificity":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        return tn / (tn + fp)
    elif metric == "Recall":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        return tp / (tp + fn)
    elif metric == "NPV":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        return tn / (tn + fn)
    elif metric == "Confusion":
        return confusion_matrix(y_true, y_pred)
    else:
        return None
        
def NN_CLF(shape):
    model = Sequential()
    model.add(Dense(32, input_dim=shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=weighted_binary_crossentropy, optimizer='adam', metrics=['accuracy'])
    return model

def binary_classifier(name, classifier_type, X, y):
    if classifier_type == "SVM":
        clf = SVC()
    elif classifier_type == "NN":
        y = list(y)
        for i in range(len(y)):
          y[i] = int(y[i])
        clf = NN_CLF(len(X[0]))
    elif classifier_type == "RF":
        clf = RandomForestRegressor(n_estimators=100)
    elif classifier_type == "LR":
        clf = LogisticRegression(max_iter=1000)
    elif classifier_type == "GA2M":
        clf = ExplainableBoostingClassifier(interactions=100)
    else:
        clf = None
    print(clf)
    clf.fit(X, y)
    return clf

# edge embeddings from nodes

def edge_embedding(emb1, emb2, method):
    emb_len = len(emb1)
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)
    if method == "Average":
        edge_emb = (emb1 + emb2) / 2
    elif method == "Hadamard":
        edge_emb = (emb1 * emb2)
    elif method == "W_L1":
        edge_emb = np.absolute(emb1 - emb2)
    elif method == "W-L2":
        edge_emb = np.square(emb1 - emb2)
    elif method == "Concat":
        edge_emb = np.concatenate((emb1, emb2),axis = 0)
    return edge_emb.tolist()