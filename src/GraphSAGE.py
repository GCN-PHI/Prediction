# Training GraphSAGE Model 
def GraphSAGE_model(G, X, y):
    combine_method = "concat"
    val_size = 0.2
    batch_size = 50
    epochs = 10  
    num_samples = [10, 5] # the number of neighbors will be aggregated on each layer 
    layer_sizes = [128] * len(num_samples) 

    val_count = int(len(X)*val_size) # Validation data for GraphSAGE model
    val_edges = X[0:val_count]
    val_labels = y[0:val_count]
    gcn_edges = list(X[val_count+1:len(X)-1])
    gcn_labels = list(y[val_count+1:len(y)-1])

    # This generator feeds the edges to the model. 
    train_gen = GraphSAGELinkGenerator(G, batch_size, num_samples) 
    val_gen = train_gen
    train_flow = train_gen.flow(gcn_edges, gcn_labels)
    val_flow = val_gen.flow(val_edges, val_labels)

    print(G.info())

    graphsage = GraphSAGE(
        layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.2
    )
    x_inp, x_out = graphsage.in_out_tensors()
    # The predictor layer at the end of the GraphSAGE
    predictions = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method=combine_method
    )(x_out)

    embedding_model = Model(inputs=x_inp, outputs=predictions)
    embedding_model.compile(
        optimizer=optimizers.Adam(lr=1e-3),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
    )
    # Train the model
    history = embedding_model.fit(
        train_flow, epochs=epochs, validation_data=val_flow, verbose=1, shuffle=True, workers = 8
    )
    # Revize the input and the output layer and generate a new model by using the weights of the trained model
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
#     Model.save("GSAGE.model")
    return Model(inputs=x_inp_src, outputs=x_out_src)

# Generating node embeddings, training binary classifier and testing the classifier
def GCN_embeddings(Model, g_nx_all):
    batch_size = 50
    
    G_all = StellarGraph.from_networkx(g_nx_all, node_features="feature")
    tra_node_gen = GraphSAGENodeGenerator(G_all, batch_size, [10, 5]) # This generator passes the nodes to the model
    tra_node_gen = tra_node_gen.flow(g_nx_all.nodes())
    emb_tra = Model.predict(tra_node_gen, workers=8, verbose=1)
    emb_dict = {} # The dict contains embeddings for all the proteins in the benchmark and holdout dataset 
    node_list = list(g_nx_all.nodes())
    for i in range(len(node_list)):
        emb_dict[node_list[i]] = emb_tra[i]
    return emb_dict