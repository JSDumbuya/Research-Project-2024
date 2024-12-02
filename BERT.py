import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

#Load data
corpus = pd.read_csv('v3_preprocessed_data.csv')
corpus = corpus['body'].tolist()

#gender_words = ['she', 'he', 'her', 'him', 'his', 'hers','woman', 'man', 'women', 'men', 'boy', 'girl', 'lady', 'guy']
gender_words = ['she', 'he']
adjectives = pd.read_csv('adjectives_from_corpus.csv')['adjective'].tolist()


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


def extractEmbeddings(target_words, embedding_df):
    extracted_embeddings = []

    for word in target_words:
        word_embeddings = embedding_df[embedding_df['token'] == word]['embedding'].values
        
        if word_embeddings.size > 0:
            for embedding in word_embeddings:  
                extracted_embeddings.append({
                    'word': word,
                    'embedding': embedding
                })
        else:
            print(f"Word '{word}' not found in the corpus embeddings")

    return extracted_embeddings

def calculateAverageEmbedding(gendered_word_embeddings, target_word):
    filtered_df = gendered_word_embeddings[gendered_word_embeddings['word'] == target_word]
    embeddings = filtered_df['embedding'].values
    avg_embedding = np.mean(embeddings, axis=0)
    return target_word, avg_embedding

def calculateCosineSimilarity(gendered_embedding, target_embeddings):
    all_results = []

    for _, row in target_embeddings.iterrows():
        target_word = row['word']
        embedding = row['embedding']
        sim = cosine_similarity(gendered_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
        all_results.append((sim, target_word))

        all_results.sort(reverse=True, key=lambda x: x[0])
    
    return all_results

#***General***
#tokenized_chunks = []
#corpus_embeddings = []
adjective_embeddings = []
gendered_word_embeddings = []

'''for row in corpus:
    encoding = tokenize_chunks(row)
    tokenized_chunks.append(encoding)'''

#***Create corpus and store result***
'''for encoding in tokenized_chunks:
    embeddings = create_embeddings(encoding)
    corpus_embeddings.extend(embeddings)

df_embeddings = pd.DataFrame(corpus_embeddings, columns=['token', 'embedding'])
df_embeddings.to_csv('corpus_embeddings.csv', index=False)'''

#Husk at skifte sti
corpus = pd.read_csv('/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/3. Semester/Research Project/corpus_embeddings.csv')


#***Extract adjective embeddings and store result***
'''extracted_adj = extractEmbeddings(adjectives, corpus)
adjective_embedding_df = pd.DataFrame(extracted_adj)
adjective_embedding_df.to_csv('adjectives_embeddings.csv', index=False)'''

'''Word 'u.s.' not found in the corpus embeddings
Word 'u.k.' not found in the corpus embeddings
Word 'subscribe' not found in the corpus embeddings
Word 'uphill' not found in the corpus embeddings
Word 'last-minute' not found in the corpus embeddings
Word 'two-time' not found in the corpus embeddings'''

adjective_embeddings = pd.read_csv('/Users/jariasallydumbuya/Library/CloudStorage/OneDrive-ITU/Computer Science/3. Semester/Research Project/adjectives_embeddings.csv')
adjective_embeddings['embedding'] = adjective_embeddings['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ')
)


#***Create gendered embeddings***
'''extract_gendered_words = extractEmbeddings(gender_words, corpus)
gendered_word_embedding_df = pd.DataFrame(extract_gendered_words)
gendered_word_embedding_df.to_csv('gendered_word_embeddings.csv', index=False)'''
gendered_word_embeddings = pd.read_csv('gendered_word_embeddings.csv')
gendered_word_embeddings['embedding'] = gendered_word_embeddings['embedding'].apply(
    lambda x: np.fromstring(x.strip("[]"), sep=' ')
)

she_embedding = calculateAverageEmbedding(gendered_word_embeddings, 'she')
he_embedding = calculateAverageEmbedding(gendered_word_embeddings, 'he')


#***Calculate and store cosine similarity***

she_similarities = calculateCosineSimilarity(she_embedding[1], adjective_embeddings)
he_similarities = calculateCosineSimilarity(he_embedding[1], adjective_embeddings)

df_she_similarities = pd.DataFrame(she_similarities, columns=['cosine similarity', 'target word'])
df_she_similarities.to_csv('she_cosine_similarities.csv', index=False)

df_he_similarities = pd.DataFrame(he_similarities, columns=['cosine similarity', 'target word'])
df_he_similarities.to_csv('he_cosine_similarities.csv', index=False)

#***Clustering***