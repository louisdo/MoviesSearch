from .inverted_index import InvertedIndexSparseMatrix
from .token import get_token_doc_counter, get_token_counter
import pandas as pd
import nltk

class MoviesSearchEngine:
    def __init__(self,tokenized_data_path,meta_data_path):
        self.tokenized_data=pd.read_csv(tokenized_data_path)
        self.tokenized_data.tokens = self.tokenized_data.tokens.str.split(",")
        self.meta_data=pd.read_csv(meta_data_path)
        self.tokens=get_token_counter(self.tokenized_data)
        self.tokens_doc_freq=get_token_doc_counter((self.tokenized_data))
        self.inverted_index=InvertedIndexSparseMatrix(self.tokenized_data,
                                                 self.tokens,self.tokens_doc_freq)
        self._tokens=list(sorted(self.tokens.keys()))
        self.tokens_index={self._tokens[index]:index for index in range(len(self._tokens))}

    def search(self,query:str):
        tokenized_query=nltk.word_tokenize(query)
        tokenized_query=[token.lower() for token in tokenized_query \
                         if token.isalnum()]

        tfidfs=InvertedIndexSparseMatrix.tfidf(tokenized_query,
                                               self.tokens,
                                               self.tokens_doc_freq,
                                               len(self.tokenized_data))

        vectorized_query=[]
        for index in range(len(tokenized_query)):
            token_index=self.tokens_index[tokenized_query[index]]
            vectorized_query.append([token_index,tfidfs[self._tokens[token_index]]])

        search_index=self.inverted_index.retrieve(vectorized_query,5)
        res=[]
        rank=1
        for index in search_index:
            res.append((rank,self.meta_data.loc[index].title_x))
            rank+=1
        return res


