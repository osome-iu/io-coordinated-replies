import torch 
from transformers import BertTokenizerFast, BertModel
import pandas as pd

import logging
import matplotlib.pyplot as plt
import numpy as np

def get_embedding(text):
    '''
    Gets embedding of text
    :param text: Test to get embedding
    
    :return embedding: the embedding vector
    '''

    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    model = BertModel.from_pretrained("setu4993/LaBSE")
    model = model.eval()

    inputs = tokenizer(text,
                       return_tensors="pt",
                       padding=True)

    with torch.no_grad():
        output = model(**inputs)

    embeddings = output.pooler_output
    embeddings = embeddings[:, :100]
        
    return embeddings[0]



def remove_diagonal_elements(matrix):
    '''
    Removes the diagonal elements from matrix
    :param matrix: the matrix in which the diagonal has to be removed
    
    :return new matrix
    '''
    
    rows, cols = matrix.shape

    # Create an index tensor to exclude diagonal elements
    indices = torch.arange(rows)

    # Remove diagonal elements by creating a mask
    mask = ~indices.unsqueeze(1).eq(indices)

    # Apply the mask to the matrix
    return matrix[mask].view(rows, cols - 1)


def get_cosine_similarity(df, column):
    '''
    Get pairwise cosine similarity for column
    :param df: Dataframe
    :param column: column to get the cosine similarity of
    
    :return None
    '''
    x = torch.stack(df[column].tolist())

    rounded_similarity = torch.nn.functional.cosine_similarity(
        x[None,:,:], 
        x[:,None,:], 
        dim=-1)

    rounded_similarity = torch.round(remove_diagonal_elements(rounded_similarity),
                             decimals=2)        

    return rounded_similarity

def get_emb_and_cosine(df, text_column):
    '''
    Gets the embedding and get the pairwise cosine similarity
    :param df: Dataframe
    :param text_column: Name of text column to get embedding
    
    :return Dataframe
    '''
    df['embedding'] = df[text_column].apply(
            lambda x: get_embedding(x)
    )
        
    cosines_pair = get_cosine_similarity(df, 'embedding')

    return cosines_pair


def get_emb_cosine_multiple_dataframe(df, 
                                      column_to_grp_by,
                                      text_column,
                                      upper_triangle_flag=False
                                     ):
    '''
    Gets the embedding and pairwise cosine similarity for multiple dataframe
    :param df: DataFrame
    :param column_to_grp_by: Column to groupby
    :param text_column: Name of text column
    
    :return DataFrame
    '''
    df_grp = df.groupby([column_to_grp_by]).last().reset_index()
    
    df_grp[column_to_grp_by] = df_grp[column_to_grp_by].astype(int)
    df[column_to_grp_by] = df[column_to_grp_by].astype(int)

    df_all = pd.DataFrame()

    for index, row in df_grp.iterrows():
        print(f'***** Start index: {index} ******')
        flag_1 = df[column_to_grp_by] == row[column_to_grp_by]
        
        df_sample = df.loc[flag_1]
        
        rounded_similarity = get_emb_and_cosine(df_sample, text_column)
        
        if upper_triangle_flag == True:
            upper_triangular = rounded_similarity[np.triu_indices(
                rounded_similarity.shape[0],
                k=1)]
        
            df_grp['cosine_similarity'] = upper_triangular
            df_all = pd.concat([df_all, df_grp[[column_to_grp_by,
                                                'cosine_similarity']]
                               ])
        else:
            df_sample['cosine_similarity'] = rounded_similarity.tolist()
            df_all = pd.concat([df_all, df_sample])
        
    return df_all

    
def get_sentence_embedding(sentences,
                           save_path=None):
    '''
    Get sentence embeddings
    :param sentences: sentences to get embedding
    '''
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states = True, # Whether the model returns all hidden-states.
                                      )
    model.eval()

    all_data = []
    batch_size = 100
    i = 0
    for idx in range(0, len(sentences), batch_size):
        batch = sentences[idx : min(len(sentences), idx+batch_size)]

        # encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        encoded = tokenizer.batch_encode_plus(batch,
                                              max_length=50, 
                                              padding='max_length', 
                                              truncation=True)

        # print(encoded)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        # print([key for key, value in encoded.items])
        with torch.no_grad():
            outputs = model(**encoded)
        lhs = outputs[2] #all embedding in 13 layer

        #[# layers, # batches, # tokens, # features]
        # token_vecs = lhs[-2]
        # print(len(token_vecs))
        # print(lhs[0].size())

        token_embeddings = torch.stack(lhs, dim=0)
        # print(token_embeddings.size())

        # # token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # # print(token_embeddings.size())
        token_embeddings = token_embeddings.permute(1, 0, 2, 3)
        # print(token_embeddings.size())
        for embedding in token_embeddings:
            #layer token features
            # embedding = embedding.permute(1, 0, 2)
            embedding = embedding[-4:] #

            # print(embedding.size())
            sentence_embedding = torch.sum(embedding, dim=0)


            sentence_embedding = torch.mean(sentence_embedding, dim=0)

            all_data.append(sentence_embedding.tolist())

    if save_path == None:
        return all_data

    df_emb = pd.DataFrame(columns=['embeddings'])
    df_emb['embeddings'] = all_data

    df_emb.to_pickle(f'{save_path}', index=False)

    return df_emb
