
def print_results(y, preds, title, c_type):
    _title = str(title)
    metric_list = ["accuracy", "Macro F1-score", "AUC-PR", "Precision", "Recall", "Specificity", "NPV", "AUC-ROC"]
    if c_type == "NN":
        preds = [x[0] for x in preds]
        preds = np.rint(preds).astype(np.int32)
    if c_type =="RF" or c_type == "RF2" or c_type == "GA2M-R":
        preds = np.rint(preds).astype(np.int32)
    test_results_log[_title] = {}
    for _metric in metric_list:
        test_results_log[_title][_metric] = round(calc_metric(_metric, y, preds), 3)
        print("Test "+_metric+": ", test_results_log[_title][_metric])


def interaction_classifier(clf_model, embed_dict, x_tra, y_tra, x_ho=None, y_ho=None, load_model = False):
    methods = ["Concat"]
    for method in methods:
        x_edges = []
        for k in range(len(x_tra)):
            _tr_emb = edge_embedding(embed_dict[x_tra[k][0]],embed_dict[x_tra[k][1]],method)
            x_edges.append(_tr_emb)
        if x_ho != None:
            print("Holdout Experiment") 
            print("Training has been started...", len(x_edges))
            if load_model:
              edge_classifier = binary_classifier("Holdout classifier", "RF", x_edges,y_tra)
              with open('~/Data/RF_final.pkl','wb') as f:
                pickle.dump(edge_classifier,f)
            else:
              with open('~/Data/RF_final.pkl', 'rb') as f:
                edge_classifier = pickle.load(f)
            print("Training is ended...")
            test_preds = []
            for i in range(0, len(x_ho),10000):
                print(i)
                x_edges_ho = []
                last = i+10000
                if last > len(x_ho):
                    last = len(x_ho)
                for k in range(i, last):
                    _e_emb = edge_embedding(embed_dict[x_ho[k][0]],embed_dict[x_ho[k][1]],method)
                    x_edges_ho.append(_e_emb)
                test_ho_embeddings = np.asarray(x_edges_ho)
                _test_preds = edge_classifier.predict(test_ho_embeddings)
                test_preds.extend(_test_preds)
            title = "Holdout tests"
            print_results(y_ho, test_preds, title, clf_model)

        else:
            fold_count = 5
            k_fold = KFold(n_splits = fold_count, shuffle = False) # The training dataset is already shuffled.
            fold_number = 0
            for train_index, test_index in k_fold.split(x_edges):
                fold_number += 1
                x_ho_train = [x_edges[i] for i in train_index]
                y_ho_train = [y_tra[i] for i in train_index]
                x_ho_test = [x_edges[i] for i in test_index]
                y_ho_test = [y_tra[i] for i in test_index]
                print("# of possitive training edges: ",y_ho_train.count(1), ", # of negative training edges: ", y_ho_train.count(0))
                best_model = binary_classifier("5_fold classifier", clf_model, x_ho_train,y_ho_train)
                _ho_edge_embeddings = np.asarray(x_ho_test)
                print("# of possitive test edges: ",y_ho_test.count(1), ", # of negative test edges: ", y_ho_test.count(0))
                test_preds = best_model.predict(_ho_edge_embeddings)
                print("# of possitive predicted edges: ",np.count_nonzero(test_preds == 1), ", # of negative predicted edges: ", np.count_nonzero(test_preds == 0))
                title = "Fold " + str(fold_number)
                print_results(y_ho_test, test_preds, title, clf_model)
