import pandas as pd
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

all_filenames = ['runnersworld.csv', 'thegaurdian.csv', 'irunfar copy.csv']

#To do:
#Lemmatization does not seem to work.
# remove stop words

def preprocessData():
    data = pd.concat((pd.read_csv(file).rename(columns=str.lower) for file in all_filenames), ignore_index=True)

    # Normalize text (convert to lowercase)
    data['body'] = data['body'].str.lower()
    # Insert space between camel case words
    data['body'] = data['body'].apply(lambda text: re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text))
    # Remove unnecessary whitespace (leading/trailing spaces, multiple spaces)
    data['body'] = data['body'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    # Tokenize the body text into sentences
    data['body'] = data['body'].apply(lambda x: sent_tokenize(x))

    # Flatten the list of sentences into chunks, ensuring each chunk is within the token limit
    max_tokens = 512  
    chunked_data = []

    for _, row in data.iterrows():
        sentences = row['body']
        chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Estimate token count (rough approximation: 1 word = 1 token)
            token_count = len(sentence.split())
            
            if current_length + token_count > max_tokens:
                # If adding this sentence exceeds max tokens, create a new chunk
                chunked_data.append(' '.join(chunk))
                chunk = [sentence]  # Start a new chunk with the current sentence
                current_length = token_count  # Reset current token count
            else:
                # Add sentence to current chunk
                chunk.append(sentence)
                current_length += token_count
        
        # Add the last chunk if it contains sentences
        if chunk:
            chunked_data.append(' '.join(chunk))
    
    # Create a new DataFrame for the chunked data
    chunked_df = pd.DataFrame(chunked_data, columns=['body'])
    
    chunked_df.to_csv('preprocessed_data_BERT.csv', index=False)

preprocessData()
