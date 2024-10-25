import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import wordnet as wn
import ast
from collections import Counter

# Load preprocessed data
preprocessed_data = pd.read_csv('preprocessed_data.csv')

#Create Corpus - The corpus contains 726 rows of content.
corpus = preprocessed_data

#print(corpus)

#Create target words from corpus
def extractTargetWords(words):
    tagged = nltk.pos_tag(words)
    adjectives = [word for word, pos in tagged if pos.startswith('JJ') and len(word) > 2]
    nouns = [word for word, pos in tagged if pos.startswith('NN') and len(word) > 2]
    verbs = [word for word, pos in tagged if pos.startswith('VB') and len(word) > 2]
    return adjectives, nouns, verbs

'''corpus['adjectives'] = corpus['text'].apply(lambda x: extractTargetWords(x)[0])
corpus['nouns'] = corpus['text'].apply(lambda x: extractTargetWords(x)[1])
corpus['verbs'] = corpus['text'].apply(lambda x: extractTargetWords(x)[2])'''


all_adjectives = []
all_nouns = []
all_verbs = []


for index, row in corpus.iterrows():
    words = ast.literal_eval(row['text'])  

    adjectives, nouns, verbs = extractTargetWords(words)

    # Update unique sets with filtered words
    all_adjectives.extend(word for word in adjectives if wn.synsets(word))
    all_nouns.extend(word for word in nouns if wn.synsets(word))
    all_verbs.extend(word for word in verbs if wn.synsets(word))

adjective_count = Counter(all_adjectives)
noun_count = Counter(all_nouns)
verb_count = Counter(all_verbs)


words_to_include = 200  

# Get the most common words
adjectives_to_include = [word for word, count in adjective_count.most_common(words_to_include)]
nouns_to_include = [word for word, count in noun_count.most_common(words_to_include)]
verbs_to_include = [word for word, count in verb_count.most_common(words_to_include)]


# Save the target words to CSV files
adjectives_df = pd.DataFrame(adjectives_to_include, columns=['adjective'])
nouns_df = pd.DataFrame(nouns_to_include, columns=['noun'])
verbs_df = pd.DataFrame(verbs_to_include, columns=['verb'])


adjectives_df.to_csv('adjectives_from_corpus.csv', index=False)
nouns_df.to_csv('nouns_from_corpus.csv', index=False)
verbs_df.to_csv('verbs_from_corpus.csv', index=False)

'''
First iteration: 
    * Remove duplicates.
    * Extracting the words that can be looked up in WordNet (exists in the dictionary)
    * Nouns: 7154
    * Adjectives: 5231
    * Verbs: 5086
Second iteration: finding a way to put a contraint on the amount of words e.g. the nouns alone were 5086 words:
    * Extracting the most frequently used words.
    * remove words with less than 2 chars because they are not useful.
'''