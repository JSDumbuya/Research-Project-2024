import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

#Load data
corpus = pd.read_csv('v2_preprocessed_data.csv')

#Create target words to be extracted from the corpus.
#See word2vec for more gender pairs.
gender_words = ['she', 'he', 'her', 'him', 'his', 'her','woman', 'man', 'women', 'men']
#performance
#personal_life
#aeasthetics

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

for text in corpus['text']:
    embeddings = create_embeddings(text)
    corpus_embeddings.extend(embeddings)

embedding_df = pd.DataFrame(corpus_embeddings, columns=['Word', 'Embedding'])

print(embedding_df.head())


def extractCategoryEmbeddings():
    return 0

def calculateCosineSimilarity():
    return 0

def displayResults():
    return 0

'''cosine_similarity_results = []
for gender_word, gender_embedding in gendered_embeddings.items():
    for target_word, target_embedding in target_word_embeddings.items():
        sim = cosine_similarity(gender_embedding.reshape(1, -1), target_embedding.reshape(1, -1))[0][0]
        cosine_similarity_results.append((gender_word, target_word, sim))'''


#Visualize results
'''similarity_df = pd.DataFrame(cosine_similarity_results, columns=['Gendered Word', 'Target Word', 'Cosine Similarity'])

gendered_word_pairs = [('she', 'he'), ('woman', 'man'), ('female', 'male')]


for female, male in gendered_word_pairs:
    # Filter the DataFrame for both gendered words
    female_df = similarity_df[similarity_df['Gendered Word'] == female].nlargest(10, 'Cosine Similarity')
    male_df = similarity_df[similarity_df['Gendered Word'] == male].nlargest(10, 'Cosine Similarity')
    
    # Merge the results to create the desired table structure
    merged_df = pd.DataFrame({
        f'Target Word ({female})': female_df['Target Word'].reset_index(drop=True),
        f'Cosine Similarity ({female})': female_df['Cosine Similarity'].reset_index(drop=True),
        f'Target Word ({male})': male_df['Target Word'].reset_index(drop=True),
        f'Cosine Similarity ({male})': male_df['Cosine Similarity'].reset_index(drop=True)
    })

    merged_df.to_csv(f'v1_contextual_model_similarity_{female}_{male}.csv', index=False)'''