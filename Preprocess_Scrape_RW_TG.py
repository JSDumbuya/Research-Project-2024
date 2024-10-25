import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin
import pandas as pd
import re
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#check .../robots.txt to see if we have permission to scrape site.

def ensureBalancedDataSet():
    return 0

corpus = []
all_filenames = ['runnersworld.csv', 'thegaurdian.csv']
preprocessedData = []
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
tempPreprocessing = []

def preprocessData():
    for file in all_filenames:
        df = pd.read_csv(file)
        df.columns = [col.lower() for col in df.columns]  # Standardize column names to lowercase
        tempPreprocessing.append(df)
    data = pd.concat(tempPreprocessing, ignore_index=True)

    for column in ['header', 'body']:
        # Convert to lowercase
        data[column] = data[column].apply(lambda text: text.lower() if isinstance(text, str) else '')
        # Remove punctuation and non-alphabetic characters
        data[column] = data[column].apply(lambda text: re.sub(r'[^a-z\s]', '', text))
        # Tokenize the text
        data[column] = data[column].apply(lambda text: word_tokenize(text))
        # Remove stopwords and lemmatize
        data[column] = data[column].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words])

        data_single_column = pd.concat([data['header'], data['body']], ignore_index=True).to_frame(name='text')

        # Return the DataFrame with a single 'text' column
        data_single_column.to_csv('preprocessed_tg_rw.csv', index=False)


#Related to Runners world scraping
KEYWORDS = [
    "Backyard Ultra", "Ultramarathon", "UTMB", "Barkley marathons", "Ultrarunner", "Ultrarunning"
]

def contains_keyword(title):
    title_lower = title.lower() 
    return any(keyword.lower() in title_lower for keyword in KEYWORDS)

def runnersWorldScraping():
    base_url = 'https://www.runnersworld.com'
    visited_urls = set()
    article_urls = set()

    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to retrieve page: {base_url}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    for div in soup.find_all('div', attrs={'data-vars-ga-outbound-link': True}):
        article_url = div['data-vars-ga-outbound-link']
        if article_url.startswith('http'):
            article_urls.add(article_url)

    for link in soup.find_all('a', href=True):
        # Construct absolute URL
        next_url = urljoin(base_url, link['href'])  
        # Ensure the link is part of the same site and not already visited
        if next_url.startswith(base_url) and next_url not in visited_urls:
            # Mark this URL as visited
            visited_urls.add(next_url)  
            # Make a request to each collected URL
            next_response = requests.get(next_url)
            if next_response.status_code == 200:
                next_soup = BeautifulSoup(next_response.content, 'html.parser')
                for div in next_soup.find_all('div', attrs={'data-vars-ga-outbound-link': True}):
                    article_url = div['data-vars-ga-outbound-link']
                    if article_url.startswith('http'):
                        article_urls.add(article_url)

    # Filter URLs based on their keywords in their titles
    filtered_article_urls = []
    for article_url in article_urls:
        article_response = requests.get(article_url)
        if article_response.status_code == 200:
            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            header = article_soup.find('h1').get_text(strip=True)  # Extract the header
            paragraphs = article_soup.find_all('p')  # Extract all paragraphs
            body = ' '.join([para.get_text(strip=True) for para in paragraphs])  # Join all paragraphs
            
            # Check if the header or body contains any of the keywords
            if contains_keyword(header) or contains_keyword(body):
                filtered_article_urls.append(article_url)
        else:
            print(f"Failed to retrieve article: {article_url}")


    # Write to CSV file
    with open('runnersworld.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Header', 'Body'])

        # Scrape each filtered article
        for article_url in filtered_article_urls:
            article_response = requests.get(article_url)
            if article_response.status_code == 200:
                article_soup = BeautifulSoup(article_response.content, 'html.parser')
                header = article_soup.find('h1').get_text(strip=True)  
                paragraphs = article_soup.find_all('p')  
                # Join extracted paragraphs
                body = ' '.join([para.get_text(strip=True) for para in paragraphs])  
                
                writer.writerow([header, body])
            else:
                print(f"Failed to retrieve article: {article_url}")

#EOF: code related to Runners world scraping


#Related to The Gaurdian scraping

def theGuardianScraping():
    base_url = "https://www.theguardian.com"
    article_base_url = "https://www.theguardian.com/lifeandstyle/ultrarunning"
    all_articles = []

    # Loop through pages 1 to 5 
    for page in range(1, 6):
        if page > 1:
            url = f"{article_base_url}?page={page}"
        else:
            url = article_base_url
        
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}. Status code: {response.status_code}")
            continue
        
        soup = BeautifulSoup(response.content, 'html.parser')

        articles = soup.find_all('a', href=True, attrs={'data-link-name': True})

        for article in articles:
            href = article['href']

            if not (article['data-link-name'].startswith('feature') or article['data-link-name'].startswith('news')):
                continue
            
            if href.startswith('/'):
                href = base_url + href
            
            # Fetch the article page
            article_response = requests.get(href)
            if article_response.status_code != 200:
                print(f"Failed to retrieve article: {href}. Status code: {article_response.status_code}")
                continue
            
            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            
            header = article_soup.find('h1').get_text(strip=True) if article_soup.find('h1') else 'No Title'
            paragraphs = article_soup.find_all('p')
            body = ' '.join([para.get_text(strip=True) for para in paragraphs])

            all_articles.append({'header': header, 'body': body})

    keys = all_articles[0].keys() if all_articles else []
    with open('thegaurdian.csv', 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_articles)

#EOF: code related to The Gaurdian scraping 
