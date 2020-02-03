import pandas as pd
import numpy as np
import nltk
import ast
from tqdm import tqdm
from argparse import ArgumentParser




def tokenize_genres(genres_list):
    return [genre["name"].lower() for genre in genres_list]


def tokenize_cast(cast_list):
    #characters_list=[cast["character"].lower().replace("(voice)","").strip() for cast in cast_list]
    characters_list=[]
    for cast in cast_list[:10]:
        characters_list.extend([item.lower() for item in nltk.word_tokenize(cast["character"]) if item.isalnum()])
        characters_list.extend([item.lower() for item in nltk.word_tokenize(cast["name"]) if item.isalnum()])

    return characters_list


def tokenize_keywords(keyword_list):
    token_keyword=[]
    for kw in keyword_list:
        token_keyword.extend([item.lower() for item in nltk.word_tokenize(kw["name"]) if item.isalnum()])

    return token_keyword


def tokenize_sentence(sentence):
    """
    This function can be used for the overview, tagline or any
    column that is of string representation
    """
    tokens=nltk.word_tokenize(sentence)
    return [token.lower() for token in tokens if token.isalnum()]


def create_tokenize_data(path):
    df=pd.read_csv(path)
    FUNCTIONS_2_APPLY={"genres":tokenize_genres,"cast":tokenize_cast,"keywords":tokenize_keywords}

    for col in ["cast","keywords"]:
        df[col]=df[col].fillna("[]")
        df[col]=df[col].apply(ast.literal_eval)
        df[col]=df[col].apply(FUNCTIONS_2_APPLY[col])

    for col in ["overview","tagline"]:
        df[col]=df[col].fillna("")
        df[col]=df[col].apply(tokenize_sentence)

    tokens_list=[]
    for index in tqdm(df.index):
        tokens_list.append(",".join(df.loc[index].cast+df.loc[index].overview+\
                                    df.loc[index].keywords+\
                                    nltk.word_tokenize(df.loc[index].title_x.lower())))

    tokens_col=pd.Series(tokens_list)

    tokenize_data=df[["title_x","id"]]
    tokenize_data["tokens"]=tokens_col
    return tokenize_data

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv-in",default="",
                        help="path the general data")
    parser.add_argument("--csv-out",default="",
                        help="path to where to save the processed data")
    args=parser.parse_args()

    assert (len(args.csv_in)==0 and len(args.csv_out)==0) or \
           (len(args.csv_in)>0 and len(args.csv_out)>0)

    if args.csv_in:
        tokenized_data=create_tokenize_data(args.csv_in)
        tokenized_data.to_csv(args.csv_out,index=False)
