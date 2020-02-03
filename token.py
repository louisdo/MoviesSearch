from collections import Counter
from tqdm import tqdm

def get_token_counter(tokenized_data:pd.DataFrame)->Counter:
    tokens=Counter()
    for index in tqdm(tokenized_data.index):
        for token in tokenized_data.loc[index].tokens:
            tokens[token]+=1

    return tokens


def get_token_doc_counter(tokenized_data:pd.DataFrame)->Counter:
    tokens_doc=Counter()
    for index in tqdm(tokenized_data.index):
        for token in list(set(tokenized_data.loc[index].tokens)):
            tokens_doc[token]+=1

    return tokens_doc
