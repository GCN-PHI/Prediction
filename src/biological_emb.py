# Biological features  

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])  

#by using Doc2Vec
def d2v_embeddings(seq_dict, host_prots, is_train = False, bpe_type = "multi"):
    sequence_embeddings = {}
    tagged_seq = []
    vec_size = 128
    max_epochs = 10
    alpha = 0.025
    if bpe_type == "unified":
        sp_h = spm.SentencePieceProcessor(model_file='../Data/a.model')
        sp_p = spm.SentencePieceProcessor(model_file='../Data/a.model')
    elif bpe_type == "multi":
        sp_h = spm.SentencePieceProcessor(model_file='../Data/h.model')
        sp_p = spm.SentencePieceProcessor(model_file='../Data/p.model')
    elif bpe_type == "regular":
        for prot in seq_dict:
          chunks, chunk_size = len(seq_dict[prot]), 5
          tagged_seq.append(TaggedDocument([seq_dict[prot][i:i+chunk_size] for i in range(0, chunks, chunk_size)],prot))
    else:
        raise ValueError('Given type is not valid for the Doc2Vec method.')
    
    if is_train:
      if bpe_type != "regular":
        for prot in seq_dict:
            if prot in host_prots:
                word_list = sp_h.encode(seq_dict[prot], out_type=str)
            else:
                word_list = sp_p.encode(seq_dict[prot], out_type=str)
            tagged_seq.append(TaggedDocument(word_list, prot))

      PROT_EMB_MODEL = Doc2Vec(vector_size=vec_size,
                      min_count=2,
                      dm =1, 
                      epochs = max_epochs)
      PROT_EMB_MODEL.build_vocab(tagged_seq)
      PROT_EMB_MODEL.train(tagged_seq,
                  total_examples=PROT_EMB_MODEL.corpus_count,
                  epochs=PROT_EMB_MODEL.epochs)
      PROT_EMB_MODEL.save("../Data/D2V_"+bpe_type+"_("+str(vec_size)+")_model")
    else:
      PROT_EMB_MODEL = Doc2Vec.load("../Data/D2V_"+bpe_type+"_("+str(vec_size)+")_model")

    if bpe_type != "regular":
      for prot in seq_dict:
          if prot in host_prots:
              word_list = sp_h.encode(seq_dict[prot], out_type=str)
          else:
              word_list = sp_p.encode(seq_dict[prot], out_type=str)
          sequence_embeddings[prot] = PROT_EMB_MODEL.infer_vector(word_list)
    else:
      for prot in seq_dict:
        chunks, chunk_size = len(seq_dict[prot]), 5
        chunk_set = [seq_dict[prot][i:i+chunk_size] for i in range(0, chunks, chunk_size)]
        sequence_embeddings[prot] = PROT_EMB_MODEL.infer_vector(chunk_set)
    return sequence_embeddings

def unirep_embeddings(seq_dict):
    import_or_install("tape_proteins")
    from Bio import SeqIO
    from Bio.Seq import Seq
    unirep_emb = []
    ctr = 0
    i = 0
    unique_prots = []
    sequence_embeddings = {}
    for prot in seq_dict:
        unirep_emb.append(SeqIO.SeqRecord(Seq(seq_dict[prot][:250]), prot))
    with open("../Data/all_prots.fasta", "w") as output_handle:
        SeqIO.write(unirep_emb, output_handle, "fasta")
    ! tape-embed unirep all_prots.fasta unirep_emb.npz babbler-1900 --tokenizer unirep
    arrays = np.load('../Data/unirep_emb.npz', allow_pickle=True)
    for prot in arrays:
        sequence_embeddings[prot] = arrays[prot].item(0)["avg"]
    return sequence_embeddings

def ctd_embeddings(seq_dict):
    import_or_install("protlearn")
    from protlearn.features import ctd, aac, ngram
    sequence_embeddings = {}
    prots = []
    sequences = []
    for prot in seq_dict:
        prots.append(prot)
        sequences.append(seq_dict[prot].replace('U','').replace('X',''))
    ctd_arr, ctd_desc = ctd(sequences)
    comp, aa = aac(sequences)
    di, ngrams = ngram(sequences, n=2)
    tri, tri_ngrams = ngram(sequences, n=3)
    tri[np.isnan(tri)] = 0
    totals= [0]*len(tri_ngrams)
    for i in range(len(tri_ngrams)):
        for seq in tri:
            totals[i] += seq[i]
    top_trimers = np.argsort(totals)[::-1][:2500]
    for i in range(len(prots)):
        sequence_embeddings[prots[i]] = list(ctd_arr[i])+ list(comp[i])+ list(di[i])
    return sequence_embeddings
def pssm_embeddings(seq_dict):
    fixed_dict = seq_dict.copy()
    prot_count = len(fixed_dict)
    len_seq = 128
    aa_positions={'A':0,'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'G':6, 'E':7, 'H':8, 'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, 'U':20, 'X':21}
    for prot in fixed_dict:
      if len(fixed_dict[prot]) > len_seq:
        fixed_dict[prot] = fixed_dict[prot][:len_seq]
    count_matrix = [[0] * len(aa_positions)] * len_seq
    aa_probabilities = [0] * len(aa_positions)
    for prot in fixed_dict:
      for i in range(len(fixed_dict[prot])):
        count_matrix[i][aa_positions[fixed_dict[prot][i]]] += 1.0 / len_seq
        aa_probabilities[aa_positions[fixed_dict[prot][i]]] += 1.0 / (len_seq * prot_count)
    for i in range(len_seq):
      for j in range(len(aa_positions)):
        divider = aa_probabilities[j]
        if divider == 0:
          divider = 1
        if count_matrix[i][j] != 0:
          count_matrix[i][j] = math.log(count_matrix[i][j] / divider)

    sequence_embeddings = {}
    for prot in fixed_dict:
      prot_emb = [0] * len_seq
      for i in range(len(fixed_dict[prot])):
        prot_emb[i] = count_matrix[i][aa_positions[fixed_dict[prot][i]]]
      if len(fixed_dict[prot]) < 128:
        for i in range(len(fixed_dict[prot]), 128 - len(fixed_dict[prot])):
          prot_emb[i] = 0
      sequence_embeddings[prot] = prot_emb
    return sequence_embeddings



def protein_embedding(emb_model, host_prots, aa_dict, load_emb_model):
    PROT_EMB_MODEL_TYPE = emb_model

    if PROT_EMB_MODEL_TYPE == "Doc2Vec-unified" or PROT_EMB_MODEL_TYPE == "Doc2Vec-multi" or PROT_EMB_MODEL_TYPE == "Doc2Vec-regular":
        return d2v_embeddings(aa_dict, host_prots, load_emb_model, PROT_EMB_MODEL_TYPE.split("-")[1])
    elif PROT_EMB_MODEL_TYPE == "CTD":
        return ctd_embeddings(aa_dict)
    elif PROT_EMB_MODEL_TYPE == "Unirep":
        return unirep_embeddings(aa_dict)
    elif PROT_EMB_MODEL_TYPE == "PSSM":
        return pssm_embeddings(aa_dict)