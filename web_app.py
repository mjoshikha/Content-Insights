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

    feature_names = vectorizer.get_feature_names()
    word_scores = list(feature_names)
    sorted_word_scores = sorted(word_scores)[:20]

    return relevance_score, sorted_word_scores, readability_score

def extract_content_with_headers(url):
    
    response = requests.get(url,verify=False)

    
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

        # Return default values if any of the metadata is None
        return meta_title or "", meta_description or "", meta_keywords or "", content_text
    else:
        print("Failed to fetch URL:", url)
        return "NIL", "NIL", "NIL", "NIL"




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

def format_dataframe(df):
    def apply_color(val):
        if val >= 30:
            color = 'green'
        elif val >= 10:
            color = 'yellow'
        else:
            color = 'red'
        return f'background-color: {color}'

    formatted_df = df.style.applymap(apply_color, subset=['Content Quality Score'])
    formatted_df = formatted_df.applymap(lambda x: 'background-color: green' if x >= 8 else ('background-color: yellow' if x >= 6 else 'background-color: red'), subset=['Relevance Score'])
    formatted_df = formatted_df.applymap(lambda x: 'background-color: green' if x >= 60 else ('background-color: yellow' if x >= 50 else 'background-color: red'), subset=['Readability Score'])
    formatted_df = formatted_df.applymap(lambda x: 'background-color: green' if x >= 2 else ('background-color: yellow' if x >= 1 else 'background-color: red'), subset=['Avg. Keyword Density Score'])

    return formatted_df

# Function to analyze content
def analyze_content(urls):
    data = {'URL': [], 'Meta Title': [], 'Meta Description': [], 'Meta Keywords': [], 'Content': [], 'Content Quality Score': [],'Relevance Score': [],
            'Word Recommendation for Meta Title': [], 'Readability Score': [], 'Hard-to-read Sentences': [],
            'Avg. Keyword Density Score': [], 'Each Keyword Density Score': [], 'Word Presence Check Meta Title': [],
            'Word Presence Check Meta Description': [], 'Word Presence Check Content': []}

    for url in urls:
        meta_title, meta_description, meta_keywords, content = extract_content_with_headers(url)

        # Check if any of the metadata variables are None
        if meta_title is None or meta_description is None or meta_keywords is None:
            continue
        
        relevance_score, sorted_word_scores, readability_score = check_relevance_score_with_keywords(meta_description, meta_title, meta_keywords, content)
        hard_to_read_sentences = detect_hard_to_read_sentences(content)
        keyword_density, overall_keyword_density_score = check_keyword_density(meta_keywords, content)
        word_presence_check = check_word_presence(meta_keywords, meta_title, meta_description, content)

        # Convert 'Each Keyword Density Score', 'Word Presence Check Meta Title', 'Word Presence Check Meta Description',
        # and 'Word Presence Check Content' to string representation
        keyword_density_str = [str(item) for item in keyword_density]
        word_presence_check_meta_title_str = [str(item) for item in word_presence_check[0]]
        word_presence_check_meta_description_str = [str(item) for item in word_presence_check[1]]
        word_presence_check_content_str = [str(item) for item in word_presence_check[2]]

        content_quality_score = calculate_content_quality_score(overall_keyword_density_score, relevance_score, readability_score)

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
        data['Each Keyword Density Score'].append(keyword_density_str)
        data['Word Presence Check Meta Title'].append(word_presence_check_meta_title_str)
        data['Word Presence Check Meta Description'].append(word_presence_check_meta_description_str)
        data['Word Presence Check Content'].append(word_presence_check_content_str)

    return pd.DataFrame(data)


features_data = {
    "Feature": [
        "Content Quality Score",
        "Relevance Score",
        "Word Recommendation for Meta Title",
        "Readability Score",
        "Hard-to-read Sentences",
        "Avg. Keyword Density Score",
        "Each Keyword Density Score",
        "Word Presence Check Meta Title",
        "Word Presence Check Meta Description",
        "Word Presence Check Content"
    ],
    "Description": [
        "This score reflects the overall quality of the content on the webpage. It's calculated based on three factors:<br> \
        - Keyword Density Score: A score out of 10, indicating how well the keywords are distributed throughout the content. **If the average keyword density is greater than 2%, it gets a score of 10; otherwise, it gets 0.**<br> \
        - Relevance Score: A score out of 10, indicating how relevant the content is to the provided meta information. **If the relevance score is greater than 8, it gets a score of 10; otherwise, it gets 0.**<br> \
        - Readability Score: A score out of 10, indicating the readability of the content. **If the Flesch Reading Ease score is greater than 50, it gets a score of 10; otherwise, it gets 0. These scores are then aggregated to get the content quality score.**",
        
        "The relevance score reflects how closely the content matches the provided keywords. It's calculated using TF-IDF cosine similarity between the provided meta information (meta description, meta title, meta keywords) and the actual content of the webpage. **A score above 8 is acceptable.**",
        
        "This column provides a list of suggested words that are present in the content and could potentially be included in the meta title to improve relevance.",
        
        "The Flesch Reading Ease score indicates how easy or difficult it is to read the content. Higher scores indicate easier readability. **A score above 60 is acceptable.**",
        
        "This column contains sentences from the content that have a readability score below 50, indicating that they are hard to read.",
        
        "The average keyword density score across all keywords present in the content. **A score above 2 is acceptable.**",
        
        "This column provides keyword density scores for each keyword present in the meta keywords.",
        
        "This column indicates words that are present in the meta title but not in the meta description or content, along with the count.",
        
        "This column indicates words that are present in the meta description but not in the meta keywords or content, along with the count.",
        
        "This column indicates keywords present in the content but not in the meta keywords or meta title, along with the count."
    ]
}

# Create a DataFrame
features_df = pd.DataFrame(features_data)

# Streamlit app
def main():
    st.set_page_config(
        page_title="SEO NEXUS - Content Insights",
        page_icon=":bar_chart:",
        layout="wide"
    )
    
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        /* Add custom CSS styles here */
        body {
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            border: 1px solid #ced4da;
        }
        .stDataFrame {
            border: 1px solid #ced4da;
        }
        .stSelectbox>div>div>div {
            background-color: #e9ecef;
            color: #343a40;
        }
        .stSelectbox>div>div>div:hover {
            background-color: #adb5bd;
            color: #343a40;
        }
        .css-1mcyjrj {
            background-color: #28a745 !important;
            border-color: #28a745 !important;
            color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
        
    # Header
    st.header("Welcome to SEO NEXUS - Content Insights")
    st.write("Get insights into your content's SEO performance")

    urls_input = st.text_area("Enter URLs (one per line)", height=150)

    if st.button("Analyze", help="Click to analyze the provided URLs"):
        urls = urls_input.split('\n')
        urls = [url.strip() for url in urls if url.strip()]  # Remove empty lines and strip whitespace
        if not urls:
            st.warning("Please enter at least one valid URL.")
            return

        with st.spinner("Analyzing content..."):
            df = analyze_content(urls)
            formatted_df = format_dataframe(df)
        
        st.subheader("Analysis Results")
        st.dataframe(formatted_df, height=400)

    # Sidebar (Hamburger Menu)
    sidebar_option = st.sidebar.selectbox(
        "Menu",
        ["Feature Explanation", "Score Explanation"]
    )

    if sidebar_option == "Feature Explanation":
        st.sidebar.subheader("Feature Explanation")
        for index, row in features_df.iterrows():
            st.sidebar.markdown(f"**{row['Feature']}**: {row['Description']}", unsafe_allow_html=True)
            st.sidebar.write("---")  # Add a horizontal line between descriptions for better readability
    
    elif sidebar_option == "Score Explanation":
        st.sidebar.subheader("Score Explanation")
        score_explanation_df = pd.DataFrame({
            "Aspect": ["Content Quality Score", "Relevance Score", "Readability Score", "Keyword Density Score"],
            "Green (Favorable)": ["Score is 30", "Score >= 8", "Score >= 60", "Score >= 2"],
            "Yellow (Moderate)": ["Score is 20 or 10", "6 <= Score < 8", "50 <= Score < 60", "1 <= Score < 2"],
            "Red (Poor)": ["Score is 0", "Score < 6", "Score < 50", "Score < 1"]
        })

        # Format dataframe with colors
        score_explanation_df = score_explanation_df.style.applymap(
            lambda x: 'background-color: #28a745; color: white' if "Green" in x else (
                'background-color: #ffc107; color: black' if "Yellow" in x else (
                    'background-color: #dc3545; color: white' if "Red" in x else ''
                )
            )
        ).hide_index()

        st.sidebar.write(score_explanation_df)

if __name__ == "__main__":
    main()
