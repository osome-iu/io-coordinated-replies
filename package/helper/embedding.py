import torch 
from transformers import BertTokenizerFast, BertModel
import pandas as pd

import logging
import matplotlib.pyplot as plt

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
