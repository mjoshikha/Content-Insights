import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat
import re  
import nltk  

# Download nltk data for sentence tokenization
nltk.download('punkt')

def check_relevance_score_with_keywords(meta_description, meta_title, meta_keywords, page_content):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([meta_description, meta_title, meta_keywords, page_content])
    meta_description_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0][3]
    meta_title_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0][2]
    meta_keywords_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0][1]
    weights = {'meta_description': 0.1, 'meta_title': 0.1, 'meta_keywords': 0.1, 'page_content': 0.7}
    relevance_score = (
            weights['meta_description'] * meta_description_similarity +
            weights['meta_title'] * meta_title_similarity +
            weights['meta_keywords'] * meta_keywords_similarity +
            weights['page_content'] * 1
    )
    relevance_score = (relevance_score * 10).round(2)

    readability_score = textstat.flesch_reading_ease(page_content)

    feature_names = vectorizer.get_feature_names_out()
    word_scores = list(feature_names)
    sorted_word_scores = sorted(word_scores)[:20]

    return relevance_score, sorted_word_scores, readability_score

def extract_content_with_headers(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        meta_title_tag = soup.find('title')
        meta_title = meta_title_tag.text.strip() if meta_title_tag else None
        
        meta_description_tag = soup.find('meta', attrs={'name': 'description'})
        meta_description = meta_description_tag['content'].strip() if meta_description_tag else None
        
        meta_keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        meta_keywords = meta_keywords_tag['content'].strip() if meta_keywords_tag else None
        
        relevant_items = []
        skip = False
        for tag in soup.find_all(['h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'table']):  # Include 'table' tag
            if tag.name == 'h2' and tag.get('class') == ['related-videos__title']:
                skip = True
            if not skip:
                if tag.name == 'ul':
                    list_items = tag.find_all('li')
                    for li in list_items:
                        relevant_items.append(li.text.strip())
                elif tag.name == 'table':  # Handle table content
                    table_rows = tag.find_all('tr')
                    for tr in table_rows:
                        table_data = tr.find_all('td')
                        row_text = '. '.join([td.text.strip() for td in table_data])+'. '
                        relevant_items.append(row_text)
                else:
                    relevant_items.append(tag.text.strip())
                    relevant_items.append('. ')
        
        content_text = '\n\n'.join(relevant_items) + '\n\n'

        return meta_title, meta_description, meta_keywords, content_text
    else:
        print("Failed to fetch URL:", url)
        return None, None, None, None


def detect_hard_to_read_sentences(page_content):
    page_content = re.sub(r'\s+', ' ', page_content)
    sentences = nltk.sent_tokenize(page_content)
    
    hard_to_read_sentences = []
    
    for sentence in sentences:
        readability_score = textstat.flesch_reading_ease(sentence)
        if readability_score < 50:
            hard_to_read_sentences.append(sentence)
    
    return hard_to_read_sentences

def check_keyword_density(meta_keywords, page_content):
    meta_keywords = [keyword.strip().lower() for keyword in meta_keywords.split(',')]
    page_content_lower = page_content.lower()
    keyword_counts = {keyword: page_content_lower.count(keyword) for keyword in meta_keywords}
    total_words = len(page_content_lower.split())
    keyword_density = {}
    for keyword, count in keyword_counts.items():
        density = (count / total_words) * 100
        keyword_density[keyword] = round(density, 2)
    sorted_keyword_density = sorted(keyword_density.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate overall score out of 10
    overall_score = sum(keyword_density.values()) / (len(keyword_density))
    return sorted_keyword_density, overall_score

def check_word_presence(meta_keywords, meta_title, meta_description, page_content):
    meta_keywords1 = [phrase.strip().lower() for phrase in meta_keywords.split(',')]
    meta_keywords2 = meta_keywords.lower().split()
    meta_title = meta_title.lower().split()
    meta_description = meta_description.lower().split()
    page_content = page_content.lower()
    missing_in_meta_desc = [word for word in meta_title if word not in meta_description]
    num_missing_in_meta_desc = len(missing_in_meta_desc)
    missing_in_meta_keywords = [word for word in meta_description if word not in meta_keywords2]
    num_missing_in_meta_keywords = len(missing_in_meta_keywords)
    missing_in_page_content = [phrase for phrase in meta_keywords1 if phrase not in page_content]
    num_missing_in_page_content = len(missing_in_page_content)
    return (missing_in_meta_desc, num_missing_in_meta_desc), (missing_in_meta_keywords, num_missing_in_meta_keywords), (
        missing_in_page_content, num_missing_in_page_content)

def calculate_content_quality_score(avg_keyword_density, relevance_score, readability_score):
    keyword_density_score = 10 if avg_keyword_density > 2 else 0
    relevance_score_points = 10 if relevance_score > 8 else 0
    readability_score_points = 10 if readability_score > 50 else 0
    content_quality_score = keyword_density_score + relevance_score_points + readability_score_points
    return content_quality_score

def main():
    st.title("Webpage Content Analysis")

    # Get input URLs from the user
    urls_input = st.text_area("Enter URLs (one per line):", height=150)

    # Convert input to list of URLs
    urls = urls_input.strip().split('\n')

    if st.button("Analyze"):
        data = {'URL': [], 'Meta Title': [], 'Meta Description': [], 'Meta Keywords': [], 'Content': [], 'Content Quality Score': [],'Relevance Score': [],
                'Word Recommendation for Meta Title': [], 'Readability Score': [], 'Hard-to-read Sentences': [],
                'Avg. Keyword Density Score': [], 'Each Keyword Density Score': [], 'Word Presence Check Meta Title': [],
                'Word Presence Check Meta Description': [], 'Word Presence Check Content': []}

        # Iterate over each URL
        for url in urls:
            meta_title, meta_description, meta_keywords, content = extract_content_with_headers(url)
            if meta_title is None and meta_description is None and meta_keywords is None and content is None:
                st.warning(f"Failed to fetch URL: {url}")
                continue

            relevance_score, sorted_word_scores, readability_score = check_relevance_score_with_keywords(meta_description, meta_title, meta_keywords, content)
            hard_to_read_sentences = detect_hard_to_read_sentences(content)
            keyword_density, overall_keyword_density_score = check_keyword_density(meta_keywords, content)
            word_presence_check = check_word_presence(meta_keywords, meta_title, meta_description, content)
            content_quality_score = calculate_content_quality_score(overall_keyword_density_score, relevance_score, readability_score)

            # Append data to dictionary
            data['URL'].append(url)
            data['Meta Title'].append(meta_title)
            data['Meta Description'].append(meta_description)
            data['Meta Keywords'].append(meta_keywords)
            data['Content'].append(content)
            data['Content Quality Score'].append(content_quality_score)
            data['Relevance Score'].append(relevance_score)
            data['Word Recommendation for Meta Title'].append(sorted_word_scores)
            data['Readability Score'].append(readability_score)
            data['Hard-to-read Sentences'].append(hard_to_read_sentences)
            data['Avg. Keyword Density Score'].append(overall_keyword_density_score)
            data['Each Keyword Density Score'].append(keyword_density)
            data['Word Presence Check Meta Title'].append(word_presence_check[0])
            data['Word Presence Check Meta Description'].append(word_presence_check[1])
            data['Word Presence Check Content'].append(word_presence_check[2])

        # Create DataFrame from the collected data
        df = pd.DataFrame(data)
        
        # Display DataFrame
        st.write(df)

if __name__ == "__main__":
    main()