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

def calculate_readability_score(page_content):
    return textstat.flesch_reading_ease(page_content)

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

    feature_names = vectorizer.get_feature_names()
    word_scores = list(feature_names)
    sorted_word_scores = sorted(word_scores)[:20]

    return relevance_score, sorted_word_scores

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
        # Check if the sentence has at least 10 words
        if len(sentence.split()) >= 10:
            readability_score = textstat.flesch_reading_ease(sentence)
            hard_to_read_sentences.append((sentence, readability_score))
    
    # Sort the hard-to-read sentences based on their readability scores in ascending order
    hard_to_read_sentences.sort(key=lambda x: x[1])
    
    # Extract only the sentences without scores
    sorted_sentences = [sentence for sentence, _ in hard_to_read_sentences]
    
    return sorted_sentences



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
    overall_score = round(sum(keyword_density.values()) / len(keyword_density), 2)
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
        readability_score = calculate_readability_score(content)
        relevance_score, sorted_word_scores = check_relevance_score_with_keywords(meta_description, meta_title, meta_keywords, content)
        
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
        **-** Keyword Density Score: A score out of 10, indicating how well the keywords are distributed throughout the content. **If the average keyword density is greater than 2%, it gets a score of 10; otherwise, it gets 0.**<br> \
        **-** Relevance Score: A score out of 10, indicating how relevant the content is to the provided meta information. **If the relevance score is greater than 8, it gets a score of 10; otherwise, it gets 0.**<br> \
        **-** Readability Score: A score out of 10, indicating the readability of the content. **If the Flesch Reading Ease score is greater than 50, it gets a score of 10; otherwise, it gets 0. These scores are then aggregated to get the content quality score.**",
        
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
        if len(df) == 1:  # Check if only one URL is analyzed
            row = df.iloc[0]  # Get the first row of the DataFrame
            
            st.write(f"**Content Quality Score:** {row['Content Quality Score']}/30")
            st.write("The Content Quality Score reflects the overall quality of the content on the webpage. It takes into account the distribution of keywords, relevance to provided meta information, and readability of the content. 30 score indicate better quality.")
            
            st.write(f"**Relevance Score:** {row['Relevance Score']}/10")
            st.write("The Relevance Score indicates how closely the content matches the provided keywords in the meta information (meta description, meta title, meta keywords). Scores above 8 suggests better relevance.")
            
            st.write(f"**Word Recommendation for Meta Title:**")
            st.write("These are the suggested words that are present in the content and could potentially be included in the meta title to improve relevance.")
            st.write(", ".join(row['Word Recommendation for Meta Title']))
            
            st.write(f"**Readability Score:** {row['Readability Score']}/100")
            st.write("The Readability Score reflects how easy or difficult it is to read the content. Scores above 60 indicate easier readability.")
            
            st.write("**Top 20 Hard-to-read Sentences:**")
            st.write("Below are the top 20 sentences from the content that are identified as hard to read. Improving the readability of these sentences can enhance user experience.")
            for i, sentence in enumerate(row['Hard-to-read Sentences'][:20]):  # Display only top 20 hard-to-read sentences
                st.write(f"{i+1}. {sentence}")
            
            st.write(f"**Avg. Keyword Density Score:** {row['Avg. Keyword Density Score']}")
            st.write("The Average Keyword Density Score indicates the average density of keywords across all keywords present in the content. Score above 2 suggest better keyword distribution.")
            
            st.write("**Each Keyword Density Score:**")
            st.write("The following are the keyword density scores for each keyword present in the meta keywords.")
            for keyword_density_score in row['Each Keyword Density Score']:
                st.write(keyword_density_score)
            
            st.write("**Word Presence Check Meta Title:**")
            st.write("The following are the words present in the meta title but not in the meta description or content, along with the count. Consider adding them to the Meta Description.")
            for word_presence in row['Word Presence Check Meta Title']:
                st.write(f"- {word_presence}")
            
            st.write("**Word Presence Check Meta Description:**")
            st.write("The following are the words present in the meta description but not in the meta keywords or content, along with the count. Consider adding them to the Meta Keywords.")
            for word_presence in row['Word Presence Check Meta Description']:
                st.write(f"- {word_presence}")
            
            st.write("**Word Presence Check Content:**")
            st.write("The following are the keywords present in the content but not in the Page Content, along with the count. Consider adding them to the Page Content ")
            for word_presence in row['Word Presence Check Content']:
                st.write(f"- {word_presence}")
    
        


    

    # Sidebar (Hamburger Menu)
    sidebar_option = st.sidebar.selectbox(
        "Menu",
        ["Feature Explanation", "Score Explanation"]
    )

    if sidebar_option == "Feature Explanation":
        st.sidebar.subheader("Feature Explanation")
        st.sidebar.write("---")  # Add a horizontal line between descriptions for better readability

        for index, row in features_df.iterrows():
            st.sidebar.markdown(f"**{row['Feature']}**: {row['Description']}", unsafe_allow_html=True)
            st.sidebar.write("---")  # Add a horizontal line between descriptions for better readability
    
    elif sidebar_option == "Score Explanation":
        st.sidebar.subheader("Score Explanation")
        st.sidebar.table({
            "Aspect": ["Content Quality Score","Relevance Score", "Keyword Density Score", "Readability Score"],
            "Green (Favorable)": ["Score is 30","Score >= 8", "Score >= 2", "Score >= 60"],
            "Yellow (Moderate)": ["Score is 20 or 10","6 <= Score < 8", "1 <= Score < 2 ", "50 <= Score < 60"],
            "Red (Poor)": ["Score is 0","Score < 6", "Score < 1", "Score < 50"]
        })


if __name__ == "__main__":
    main()
