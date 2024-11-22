import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

#Load data
corpus = pd.read_csv('v2_preprocessed_data.csv')

#Create target words to be extracted from the corpus.
#gender_words = ['she', 'he', 'her', 'him', 'his', 'hers','woman', 'man', 'women', 'men', 'boy', 'girl', 'lady', 'guy']
gender_words = ['she', 'he']
adjectives = pd.read_csv('adjectives_from_corpus_filtered.csv')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def create_embeddings(word):
    input_ids = tokenizer.encode(word, return_tensors='pt', padding=True, truncation=True, max_length=512)
    attention_mask = (input_ids != 0).long()  
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    last_hidden_state = outputs.last_hidden_state

    word_embeddings = []

    for i, token_id in enumerate(input_ids[0]): 
        word = tokenizer.decode([token_id]) 
        embedding = last_hidden_state[0, i].numpy()  
        
        word_embeddings.append((word, embedding))
    
    return word_embeddings

corpus_embeddings = []
descriptors_embeddings = []

'''for text in corpus['text']:
    embeddings = create_embeddings(text)
    corpus_embeddings.extend(embeddings)'''

'''embedding_df = pd.DataFrame(corpus_embeddings, columns=['Word', 'Embedding'])
embedding_df['Embedding'] = embedding_df['Embedding'].apply(lambda x: np.array2string(x, separator=',').strip('[]'))
embedding_df.to_csv('corpus_embeddings.csv', index=False)'''



#Husk at korrigere path
embedding_df = pd.read_csv('/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/3. Semester/Research Project/corpus_embeddings.csv')
embedding_df['Embedding'] = embedding_df['Embedding'].apply(lambda x: np.fromstring(x, sep=','))

print(embedding_df.head())


def extractEmbeddings(target_words):
    extracted_embeddings = {}

    for word in target_words:
        word_embeddings = embedding_df[embedding_df['Word'] == word]['Embedding'].values
        
        if word_embeddings.size > 0:
            extracted_embeddings[word] = word_embeddings
        else:
            print(f"Word '{word}' not found in the corpus embeddings CSV")

    return extracted_embeddings


def calculateAverageEmbedding(embedding):
    return np.mean(embedding, axis=0)

def calculateCosineSimilarity(gendered_word, target_embeddings):
    all_results = []

    for target_word, embedding in target_embeddings.items():
        sim = cosine_similarity(gendered_word.reshape(1, -1), embedding.reshape(1, -1))[0][0]
        all_results.append((sim, target_word))
    
    return all_results


def displayResults(gendered_word, cosine_similarities):
    df = pd.DataFrame(cosine_similarities, columns=[f"Target Words ({gendered_word})", f"Cosine Similarity ({gendered_word})"])
    return df
