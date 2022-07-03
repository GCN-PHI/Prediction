# Predict possible edges to attach holdout proteins to the training graph

def preliminary_predictor(X, y, g_nx_all, all_dict, edge_set, load_pretrained):
    edge_embeddings = [] # The list of edge embeddings for both negative and possitive edges in training graph
    holdout_edge_emgeddings = [] # Embeddings of any possible edge in the holdout graph
    edge_preds = [] # The binary prediction of edges
    edges_for_mapping = [] # The edges that is used for attaching holdout proteins to the training graph
    _all_dict = all_dict.copy()
    if len(all_dict[X[0][0]]) >= 256:
        pca = PCA(n_components = 128)
        train_list = []
        for prot in all_dict:
            train_list.append(all_dict[prot])
        pca.fit(train_list)
        prot_list = []
        emb_list = []
        for prot in all_dict:
            prot_list.append(prot)
            emb_list.append(all_dict[prot])
        emb_list = pca.transform(emb_list)
        for i in range(len(emb_list)):
            _all_dict[prot_list[i]] = emb_list[i]
        print("PCA applied.")
    if load_pretrained:
      edge_predictor = joblib.load("../Data/edge_predictor_"+ str(len(_all_dict[X[0][0]])) + ".joblib")
    else:
      for k in range(len(X)):
          edge_embeddings.append(edge_embedding(_all_dict[X[k][0]],_all_dict[X[k][1]],"Concat"))
      # Training the edge predictor for mapping holdout proteins
      edge_predictor = binary_classifier("Edge_classifier", "RF", edge_embeddings, y)
      joblib.dump(edge_predictor, "../Data/edge_predictor_"+ str(len(_all_dict[X[0][0]])) + ".joblib")
        
    print("Training complated for edge predictor.")
    edge_embeddings = None

    n_possible_edges = len(edge_set)
    for k in range(0, n_possible_edges, 100000):
        last = k+100000
        if last > n_possible_edges:
            last = n_possible_edges
        holdout_edge_emgeddings = [edge_embedding(_all_dict[edge[0]],_all_dict[edge[1]],"Concat") for edge in edge_set[k:last]]
        if k % 100000 == 0 or last == n_possible_edges:
            _edge_preds = edge_predictor.predict(holdout_edge_emgeddings) # Classify the edges in the current batch 
            edge_preds.extend(_edge_preds)
            holdout_edge_emgeddings = []
            print(k)

    del edge_predictor
    holdout_edge_emgeddings = None
    edge_classifier = None
    rank_dict = {}
    for i in range(len(edge_set)):
        edge = edge_set[i]
        if edge[0] not in rank_dict:
            rank_dict[edge[0]] = []
        current_size = len(rank_dict[edge[0]])
        if current_size < 50:
            rank_dict[edge[0]].append((edge_preds[i], edge[1]))
            rank_dict[edge[0]].sort(key=lambda x:x[0], reverse = True)
        else:
            for i in range(current_size):
                if edge_preds[i] > rank_dict[edge[0]][i][0]:
                    rank_dict[edge[0]].insert(i, (edge_preds[i], edge[1]))
                    del rank_dict[edge[0]][-1]
                    break

    for prot in rank_dict:
        for prot2 in rank_dict[prot]:
            edges_for_mapping.append([prot, prot2[1]])

    g_nx_all.add_edges_from(edges_for_mapping)

    print("# of edges predicted as positive: ", len(edges_for_mapping), "# of all possible edges", len(edge_set))
    edge_embeddings = None
    return g_nx_all, edges_for_mapping