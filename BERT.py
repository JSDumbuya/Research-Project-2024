import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import csv 

#Load data
corpus = pd.read_csv('v3_preprocessed_data.csv')
corpus = corpus['body'].tolist()

#gender_words = ['she', 'he', 'her', 'him', 'his', 'hers','woman', 'man', 'women', 'men', 'boy', 'girl', 'lady', 'guy']
gender_words = ['she', 'he']
adjectives = pd.read_csv('adjectives_from_corpus.csv')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def tokenize_chunks(chunk):
    # Tokenize the chunk into tokens, returning token IDs and attention masks
    encoding = tokenizer(chunk, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    return encoding
    

def create_embeddings(encoding):
    with torch.no_grad():
        outputs = model(**encoding)
        embeddings = outputs.last_hidden_state

    input_ids = encoding['input_ids'][0] 

    word_embeddings = []
    for i, token_id in enumerate(input_ids):
        word = tokenizer.decode([token_id]).strip()  
        embedding = embeddings[0, i].numpy()  
        word_embeddings.append((word, embedding))  

    return word_embeddings

#obsolete 
'''def extractEmbeddings(target_words):
    extracted_embeddings = {}

    for word in target_words:
        word_embeddings = embedding_df[embedding_df['Word'] == word]['Embedding'].values
        
        if word_embeddings.size > 0:
            extracted_embeddings[word] = word_embeddings
        else:
            print(f"Word '{word}' not found in the corpus embeddings CSV")

    return extracted_embeddings'''

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


#***General***
tokenized_chunks = []
corpus_embeddings = []
adjective_embeddings = []
gendered_word_embeddings = []

for row in corpus:
    encoding = tokenize_chunks(row)
    tokenized_chunks.append(encoding)

#***Corpus***
for encoding in tokenized_chunks:
    embeddings = create_embeddings(encoding)
    corpus_embeddings.extend(embeddings)

df_embeddings = pd.DataFrame(corpus_embeddings, columns=['token', 'embedding'])
df_embeddings.to_csv('corpus_embeddings.csv', index=False)

#***Adjectives***



'''output_csv = "adjective_embeddings.csv"
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Adjective", "Embedding"])  
    for word, embedding in adjective_embeddings:
        writer.writerow([word, embedding.tolist()])'''

#***Gendered words***

#***Cosine similarity***

#***Clustering***