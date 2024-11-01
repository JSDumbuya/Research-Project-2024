import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

#Load data
corpus = pd.read_csv('v2_preprocessed_data.csv')
adjectives = pd.read_csv('adjectives_from_corpus.csv')
#verbs = pd.read_csv('verbs.csv')
nouns = pd.read_csv('nouns_from_corpus.csv')

#Notice verbs have been left out
target_words = pd.concat([adjectives['adjective']]).tolist()

gendered_words = ['she', 'he', 'woman', 'man', 'women', 'men', 'female', 'male']

#maybe filter out more words
#filter_out = [gendered_words, ['human', at 'person', 'people', 'runner', 'runners', 'athlete', 'athletes', 'marathoner', 'marathoners', 'participant', 'participants', 'individual', 'individuals', 'person', 'persons]]

#Filter out target words.
filtered_target_words = [word for word in target_words if word not in gendered_words]


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def getEmbeddings(word):
    input_ids = tokenizer.encode(word, return_tensors='pt', padding=True, truncation=True, max_length=512)
    attention_mask = (input_ids != 0).long()  
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state.mean(dim=1).numpy()[0]

#word_embeddings = {word: getEmbeddings(word) for word in target_words['word']}

corpus_text = ' '.join(corpus['text']) 
corpus_embedding = getEmbeddings(corpus_text)
gendered_embeddings = {word: getEmbeddings(word) for word in gendered_words}
target_word_embeddings = {word: getEmbeddings(word) for word in filtered_target_words}

cosine_similarity_results = []
cosine_similarity_results = []
for gender_word, gender_embedding in gendered_embeddings.items():
    for target_word, target_embedding in target_word_embeddings.items():
        sim = cosine_similarity(gender_embedding.reshape(1, -1), target_embedding.reshape(1, -1))[0][0]
        cosine_similarity_results.append((gender_word, target_word, sim))


#Visualize results

similarity_df = pd.DataFrame(cosine_similarity_results, columns=['Gendered Word', 'Target Word', 'Cosine Similarity'])

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

    merged_df.to_csv(f'v1_contextual_model_similarity_{female}_{male}.csv', index=False)