import numpy as np
from tqdm import tqdm
from collections import Counter


class InvertedIndexSparseMatrix:
    def __init__(self,tokenized_data:"pd.DataFrame",tokens:Counter,tokens_doc_freq:Counter)->None:
        self.matrix=[]
        self.shape=(len(tokenized_data),len(tokens))
        _tokens=list(sorted(tokens.keys()))
        tokens_index={_tokens[index]:index for index in range(len(_tokens))}

        for index in tqdm(tokenized_data.index):
            ins_token=tokenized_data.loc[index].tokens
            ins_tfidfs=self.tfidf(ins_token,tokens,tokens_doc_freq,len(tokenized_data))
            ins_token=list(set(ins_token))

            matrix_row=[]

            for ind in range(len(ins_token)):
                ins_token_index=tokens_index[ins_token[ind]]
                matrix_row.append([ins_token_index,ins_tfidfs[_tokens[ins_token_index]]])

            self.matrix.append(matrix_row)

    @staticmethod
    def tfidf(ins_token:list,tokens:Counter,tokens_doc_freq:Counter,corpus_size:int)->dict:
        ins_token_counter=Counter(ins_token)
        max_freq=max(ins_token_counter.items())[1]
        tfidfs={}

        for token in ins_token_counter.keys():
            tfidfs[token]=(0.5+0.5*(ins_token_counter[token]/max_freq))*\
                np.log(corpus_size/tokens_doc_freq[token])

        return tfidfs

    @staticmethod
    def distance(vector1:"SparseVector",vector2:"SparseVector")->float:
        dist=0
        indices={}

        for index in range(len(vector1)):
            if vector1[index][0] not in indices:
                indices[vector1[index][0]]=vector1[index][1]

        for index in range(len(vector2)):
            if vector2[index][0] not in indices:
                indices[vector2[index][0]]=vector2[index][1]
            else:
                indices[vector2[index][0]]-=vector2[index][1]

        for key in indices.keys():
            dist+=indices[key]**2

        return dist**0.5

    @staticmethod
    def vector_length(vector:"SparseVector")->float:
        return sum((item[1]**2 for item in vector))**0.5

    def retrieve(self,vector:"SparseVector",k=5)->np.array:
        cosine_sims=[]

        len_vector=InvertedIndexSparseMatrix.vector_length(vector)

        for index in range(len(self.matrix)):
            len_vector_row=InvertedIndexSparseMatrix.vector_length(self.matrix[index])
            cosine_sim=InvertedIndexSparseMatrix.distance(vector,self.matrix[index])/  \
                       (len_vector*len_vector_row)
            cosine_sims.append(cosine_sim)

        return np.argsort(cosine_sims)[:k]



