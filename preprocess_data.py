import pandas as pd
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

all_filenames = ['runnersworld.csv', 'thegaurdian.csv', 'irunfar copy.csv']
#preprocessedData = []
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
#tempPreprocessing = []

def preprocessData():
    data = pd.concat((pd.read_csv(file).rename(columns=str.lower) for file in all_filenames), ignore_index=True)
    
    for column in ['header', 'body']:
        # Insert space between camel case words
        data[column] = data[column].apply(lambda text: re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text))
        # Convert to lowercase and remove punctuation, non-alphabetic characters
        data[column] = data[column].apply(lambda text: re.sub(r'[^a-z\s]', ' ', text.lower()) if isinstance(text, str) else '')
        # Tokenize, remove stopwords, and lemmatize
        data[column] = data[column].apply(lambda text: [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words])


    #data_single_column = pd.concat([data['header'], data['body']], ignore_index=True).to_frame(name='text')
    data_single_column = pd.DataFrame({'text': data['header'].apply(lambda tokens: ' '.join(tokens)) + ' ' + data['body'].apply(lambda tokens: ' '.join(tokens))})


    # Return the DataFrame with a single 'text' column
    data_single_column.to_csv('v2_preprocessed_data.csv', index=False)

preprocessData()