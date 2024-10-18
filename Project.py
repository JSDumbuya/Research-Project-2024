import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin

#check .../robots.txt to see if we have permission to scrape site.

corpus = []

def ensureBalancedDataSet():
    return 0

def preprocess():
    return 0

def createCorpus():
    return 0

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
    with open('runners_stories_corpus.csv', mode='w', newline='', encoding='utf-8') as file:
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

#Commented out to ensure that new csv file is not created again.
"""if __name__ == "__main__":
    runnersWorldScraping()"""