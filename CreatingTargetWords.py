import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

#Combine preprocessed data
irunfar = pd.read_csv('preprocessed_irunfar copy.csv')
tg_rw = pd.read_csv('preprocessed_tg_rw.csv')
tg_rw.rename(columns={'text': 'preprocessed_content'}, inplace=True)

#Create Corpus - The corpus contains 528 rows of content.
corpus = pd.concat([irunfar, tg_rw], ignore_index=True)

#Create target words from corpus
def extractTargetWords(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    adjectives = [word for word, pos in tagged if pos.startswith('JJ')]
    nouns = [word for word, pos in tagged if pos.startswith('NN')]
    verbs = [word for word, pos in tagged if pos.startswith('VB')]
    return adjectives, nouns, verbs

corpus['adjectives'] = corpus['preprocessed_content'].apply(lambda x: extractTargetWords(x)[0])
corpus['nouns'] = corpus['preprocessed_content'].apply(lambda x: extractTargetWords(x)[1])
corpus['verbs'] = corpus['preprocessed_content'].apply(lambda x: extractTargetWords(x)[2])

# Remove repeats
unique_adjectives = set()
unique_nouns = set()
unique_verbs = set()


for index, row in corpus.iterrows():
    unique_adjectives.update(row['adjectives'])
    unique_nouns.update(row['nouns'])
    unique_verbs.update(row['verbs'])

# Save the target words to CSV files
adjectives_df = pd.DataFrame(unique_adjectives, columns=['adjective'])
nouns_df = pd.DataFrame(unique_nouns, columns=['noun'])
verbs_df = pd.DataFrame(unique_verbs, columns=['verb'])

adjectives_df.to_csv('adjectives_from_corpus.csv', index=False)
nouns_df.to_csv('nouns_from_corpus.csv', index=False)
verbs_df.to_csv('verbs_from_corpus.csv', index=False)



