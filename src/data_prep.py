#Loading the training graph
def load_base_graph(training_graph):
    host_prots = [] # List of host proteins in training dataset
    pathogen_prots = [] # List of pathogen proteins in training dataset
    normal_seq={} # The dict of whole protein sequences in the training dataset
    neg_edges = []
    
    
    df_pos = pd.read_excel('../Data/' + training_graph)
    g_nx_pos = nx.from_pandas_edgelist(df_pos,"HOST","VIRUS") # The positive interaction graph

    for node_id, node_data in g_nx_pos.nodes(data=True): # Declare the node type. It is needed by the GraphSage. "default" is a arbitrary type.
        node_data["label"] = "default"

    host_prots.extend(list(df_pos["HOST"]))
    pathogen_prots.extend(list(df_pos["VIRUS"]))
    
    host_prots = list(set(host_prots))
    pathogen_prots = list(set(pathogen_prots))
    
    for h_prot in host_prots:
        for p_prot in pathogen_prots:
            if g_nx_pos.has_edge(*(h_prot, p_prot)) == False: 
                neg_edges.append([h_prot, p_prot])

    for index,row in df_pos.iterrows():
        normal_seq[row["HOST"]] = row["HOST_SEQ"]
        normal_seq[row["VIRUS"]] = row["VIRUS_SEQ"]


    print("# of positive edges for training: ", g_nx_pos.number_of_edges())

    return g_nx_pos, normal_seq, host_prots, pathogen_prots, neg_edges

# Loading the holdout Graph

def load_human_prots():
  human_prots = {}
  with open("../Data/uniprot_homo_sapien.fasta.gz", "r") as handle:
      seq_list = list(SeqIO.parse(handle, "fasta"))
  for prot in seq_list:
      human_prots[str(prot.id.split('|')[1])] = str(prot.seq)
  return human_prots

def load_holdout(dataset_name, host_prots, pathogen_prots, seq_dict):
    hold_out_prot={} # The dict of whole protein sequences in the holdout dataset
    hold_out_pathogen_prots = []
    hold_out_host_prots = []
    hold_out_edges = [] # The list of positive and negative edges for holdout graph 
    hold_out_labels = [] # The label list of holdout edges
    neg_nodes_to_append = [] # The list of human prots that sampled for negative edges
    ho_edges_for_gcn = [] # The list of all possible edges between holdout graph nodes
    extra_ho_prots = []
    human_prots = load_human_prots()
    human_prots_list = human_prots.keys()
    df_holdout = pd.read_excel('../Data/'+ dataset_name)
    g_nx_holdout = nx.from_pandas_edgelist(df_holdout,"HOST","VIRUS")
    for index,row in df_holdout.iterrows():
      hold_out_prot[row["HOST"]] = row["HOST_SEQ"]
      hold_out_prot[row["VIRUS"]] = row["VIRUS_SEQ"]
      hold_out_pathogen_prots.append(row["VIRUS"])
      hold_out_host_prots.append(row["HOST"])
    hold_out_pathogen_prots = list(set(hold_out_pathogen_prots))
    hold_out_host_prots = list(set(hold_out_host_prots))
        
    hold_out_edges = list(g_nx_holdout.edges())
    hold_out_labels = [1]* len(hold_out_edges) 
    

    ##Negative edge sampling
    sample_counter = 0
    number_of_neg_edges = g_nx_holdout.number_of_edges() * 10
    while sample_counter < number_of_neg_edges:
        new_human_prot = random.sample(human_prots_list,1)[0]
        rnd_pathogen_prot = random.sample(hold_out_pathogen_prots,1)[0]
        if (new_human_prot, rnd_pathogen_prot) not in g_nx_holdout.edges() and (rnd_pathogen_prot, new_human_prot) not in g_nx_holdout.edges():
            if new_human_prot  not in g_nx_holdout.nodes(): # Add the human prot to the holdout graph if it isn't in the graph already.
                neg_nodes_to_append.append(new_human_prot)
            hold_out_edges.append((rnd_pathogen_prot, new_human_prot))
            hold_out_labels.append(0)
            hold_out_prot[new_human_prot] = human_prots[new_human_prot]
            sample_counter += 1
    g_nx_holdout.add_nodes_from(neg_nodes_to_append)
      
    c = list(zip(hold_out_edges, hold_out_labels)) 
    random.shuffle(c) # Shuffle the list of the holdout graph edges.
    hold_out_edges, hold_out_labels = zip(*c)
    hold_out_edges = list(hold_out_edges)
    hold_out_labels = list(hold_out_labels)
    print("# of samples in test dataset: ", len(hold_out_edges))
    
    all_host_list =  list(set(host_prots + neg_nodes_to_append + hold_out_host_prots))
    ho_host_list = list(set(neg_nodes_to_append + hold_out_host_prots))
    all_pathogen_list = list(set(hold_out_pathogen_prots + pathogen_prots))
    
    ctr_1 = 0
    for prot1 in hold_out_host_prots:
        if prot1 in host_prots:
            ctr_1 += 1
    print("# of new host prot:", len(hold_out_host_prots) - ctr_1," All:", len(hold_out_host_prots))

    for prot_ho in host_prots: # All possible edges between holdout host proteins and pathogen proteins in the training graph
            for prot in hold_out_pathogen_prots:
                ho_edges_for_gcn.append([prot, prot_ho])

    for node_id, node_data in g_nx_holdout.nodes(data=True):
        node_data["label"] = "default"
    return g_nx_holdout, hold_out_prot, hold_out_edges, hold_out_labels, ho_edges_for_gcn, human_prots

def set_node_features(g, emb_dict):
    for node_id, node_data in g.nodes(data=True): # Declare the node features (biological features)
        node_data["feature"]= emb_dict[node_id]
def general_graph(g_nx_pos, g_nx_holdout):
    g_nx_all = nx.compose(g_nx_pos, g_nx_holdout) # The graph that contains both training and test nodes
    # print("# of edges in the training graph: ", len(g_nx_all.edges()), ", # of nodes in the training graph:", len(g_nx_all.nodes()))
    for edge in g_nx_holdout.edges(): # Remove edges of test graph to prevent information leak.
        g_nx_all.remove_edge(*edge)
        if edge in g_nx_pos.edges():
            g_nx_pos.remove_edge(*edge)
    return g_nx_all

# Preparing the training edge set and the required variables for training GCN and classifier

def load_edges_from_memory():
    train_edges = list(np.load('train_edges.npy',allow_pickle=True))
    train_labels = list(np.load('train_labels.npy',allow_pickle=True))
    gcn_edges = list(np.load('gcn_edges.npy',allow_pickle=True))
    gcn_labels = list(np.load('gcn_labels.npy',allow_pickle=True))
    print("The edge set has been loaded from the memory.")
    print("# of possitive edges: ",train_labels.count(1), ", # of negative edges: ", train_labels.count(0))
    return train_edges , train_labels, gcn_edges, gcn_labels

def prep_train_edges(g_nx_pos, n_edges):
    train_edges = []
    train_labels = []

    possitive_edges = list (g_nx_pos.edges())
    for i in range(len(possitive_edges)):
        train_edges.append(list(possitive_edges[i]))
        train_labels.append(1)


    negative_edges = []
    negative_edges = random.sample(n_edges, 3 * g_nx_pos.number_of_edges())
    for i in range(len(negative_edges)): 
        train_edges.append(list(negative_edges[i])) # Add negative edges to the training set
        train_labels.append(0)

    extra_negative_edges = [] # Generate 10x negative edges for training preliminary classifier
    extra_negative_edges = random.sample(n_edges, 7 * g_nx_pos.number_of_edges()) 
    

    c = list(zip(train_edges, train_labels)) # Shuffle the training edge set
    random.shuffle(c)
    train_edges, train_labels = zip(*c)

    gcn_edges = list(train_edges[:])
    gcn_labels = list(train_labels[:])

    # Append generated negative edges for training binary classifier
    train_edges = list(train_edges)
    train_labels = list(train_labels)
    train_edges.extend(extra_negative_edges)
    train_labels.extend([0]*len(extra_negative_edges))
    
    c = list(zip(train_edges, train_labels))
    random.shuffle(c)
    train_edges, train_labels = zip(*c)
    print("Preliminary Classifier: # of possitive edges: ",train_labels.count(1), ", # of negative edges: ", train_labels.count(0))
    np.save('train_edges',train_edges)
    np.save('train_labels',train_labels)
    np.save('gcn_edges',gcn_edges)
    np.save('gcn_labels',gcn_labels)
    return train_edges, train_labels, gcn_edges, gcn_labels