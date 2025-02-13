import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CamembertModel, CamembertTokenizer
import torch
from fuzzywuzzy import fuzz
import re

# Title and Description

st.title("Keyword Classifier")
st.write("Upload brand terms and keywords to classify keywords using embeddings.")

# Instructions Section
with st.expander("Read More: How to Use the App"):
    st.markdown("""
    ### Instructions for Using the App
    1. **Enter your OpenAI API Key**:
       - Obtain your API key from [OpenAI](https://platform.openai.com/signup).
       - Log in or sign up for an account.
       - Navigate to the API Keys section under your profile settings.
       - Create a new key and copy it for use in this app.

    2. **Upload CSV Files**:
       - Upload a CSV file with a column named `brand_terms` containing your brand-related terms.
       - Upload another CSV file with a column named `keywords` containing the keywords you want to classify.

    3. **Set Cosine Similarity Threshold**:
       - Use the slider to adjust the threshold for classification.
       - Higher thresholds (e.g., 0.9) result in stricter matching, while lower thresholds (e.g., 0.7) allow more flexibility.

    4. **Click 'Classify Keywords'**:
       - The app will generate embeddings using OpenAI's API and classify the keywords as **branded** or **non-branded**.

    5. **View Results**:
       - A sample of 15 branded and 5 non-branded keywords will be displayed for quick QA.
       - You can download the full results as a CSV file.

    ### Notes
    - Ensure your CSV files are formatted correctly with the appropriate column names (`brand_terms` and `keywords`).
    - The app does not store your API key or data. Your key is used only during the session for embedding generation.

    ### Troubleshooting
    - If you encounter issues, verify:
        - The API key is entered correctly.
        - Your CSV files have the correct structure.
        - Your OpenAI account has sufficient credits to use the API.

    For further help, visit [OpenAI API Documentation](https://platform.openai.com/docs/).
    """)


# Input OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# File Uploads
brand_file = st.file_uploader("Upload a CSV file with brand terms (column: 'brand_terms')", type=["csv"])
keyword_file = st.file_uploader("Upload a CSV file with keywords (column: 'keywords')", type=["csv"])

# Threshold Slider
threshold = st.slider("Cosine Similarity Threshold", min_value=0.0, max_value=1.0, value=0.8)

# Add model selection
embedding_model = st.radio(
    "Select Embedding Model",
    ["OpenAI (English)", "CamemBERT (French)"],
    help="Choose OpenAI for English keywords or CamemBERT for French keywords"
)

# Add a checkbox for the new matching method
use_substring_matching = st.checkbox("Use Substring and Fuzzy Matching", value=False, help="Check to use substring and fuzzy matching for classification.")

# Define the get_embeddings function before the button click logic
def get_embeddings(texts):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [item['embedding'] for item in response['data']]

# Ensure this function is defined before the button click logic
def get_camembert_embeddings(texts):
    @st.cache_resource
    def load_camembert_model():
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        model = CamembertModel.from_pretrained('camembert-base')
        model.eval()
        return tokenizer, model
    
    tokenizer, model = load_camembert_model()
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize and prepare input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Get model output
            outputs = model(**inputs)
            
            # Use [CLS] token embedding (first token) as sentence embedding
            embedding = outputs.last_hidden_state[0, 0, :].numpy()
            embeddings.append(embedding)
    
    return np.array(embeddings)

# Function to generate regex patterns
def generate_regex_patterns(terms):
    patterns = []
    for term in terms:
        # Create a basic pattern that accounts for common variations
        pattern = re.escape(term).replace(r'\ ', r'[-\s]?')
        patterns.append(pattern)
    return patterns

# Processing
if st.button("Classify Keywords"):
    if embedding_model == "OpenAI (English)" and not api_key:
        st.error("Please enter your OpenAI API key for OpenAI embeddings.")
    elif not brand_file or not keyword_file:
        st.error("Please upload both brand terms and keywords.")
    else:
        try:
            # Load files
            brand_terms = pd.read_csv(brand_file)['brand_terms'].tolist()
            keywords = pd.read_csv(keyword_file)['keywords'].tolist()
            
            if use_substring_matching:
                # Substring and fuzzy matching
                classifications = []
                for keyword in keywords:
                    is_branded = False
                    closest_brand_term = None
                    max_similarity = 0
                    for brand_term in brand_terms:
                        # Check for exact substring match
                        if brand_term.lower() in keyword.lower():
                            is_branded = True
                            closest_brand_term = brand_term
                            break
                        # Check for fuzzy match
                        similarity = fuzz.partial_ratio(brand_term.lower(), keyword.lower())
                        if similarity > max_similarity:
                            max_similarity = similarity
                            closest_brand_term = brand_term
                        if similarity >= 85:  # You can adjust this threshold
                            is_branded = True
                            break
                    classification = "branded" if is_branded else "non-branded"
                    classifications.append({
                        "keyword": keyword,
                        "classification": classification,
                        "max_similarity": max_similarity,
                        "closest_brand_term": closest_brand_term
                    })
            else:
                # Embedding-based matching
                if embedding_model == "OpenAI (English)":
                    # Set OpenAI API key
                    openai.api_key = api_key
                    
                    st.info("Generating embeddings for brand terms...")
                    brand_embeddings = np.array(get_embeddings(brand_terms))
                    
                    st.info("Generating embeddings for keywords...")
                    keyword_embeddings = np.array(get_embeddings(keywords))
                else:  # CamemBERT (French)
                    st.info("Generating embeddings for brand terms using CamemBERT...")
                    brand_embeddings = get_camembert_embeddings(brand_terms)
                    
                    st.info("Generating embeddings for keywords using CamemBERT...")
                    keyword_embeddings = get_camembert_embeddings(keywords)
                
                # Compute cosine similarities
                st.info("Calculating cosine similarities...")
                similarities = cosine_similarity(keyword_embeddings, brand_embeddings)
                
                # Classify keywords
                classifications = []
                for i, sim in enumerate(similarities):
                    max_similarity = max(sim)
                    max_index = sim.argmax()  # Get the index of the brand term with the highest similarity
                    closest_brand_term = brand_terms[max_index]  # Get the corresponding brand term
                    classification = "branded" if max_similarity >= threshold else "non-branded"
                    classifications.append({
                        "keyword": keywords[i],
                        "classification": classification,
                        "max_similarity": max_similarity,
                        "closest_brand_term": closest_brand_term  # Include the closest brand term
                    })
            
            # Convert to DataFrame
            results = pd.DataFrame(classifications)
            st.success("Classification complete!")
            
            # Extract branded keywords
            branded_keywords = results[results['classification'] == 'branded']['keyword'].tolist()
            
            # Generate regex patterns
            regex_patterns = generate_regex_patterns(branded_keywords)
            
            # Combine patterns into a single regex
            combined_pattern = '|'.join(set(regex_patterns))  # Use set to deduplicate
            
            # Display the regex pattern
            st.subheader("Generated Regex Pattern for Branded Keywords")
            st.text_area("Regex Pattern", value=combined_pattern, height=100)
            
            # Quick QA Sample
            st.subheader("Quick QA Sample")
            branded_sample = results[results['classification'] == 'branded'].head(15)
            non_branded_sample = results[results['classification'] == 'non-branded'].head(5)

            st.write("### Sample of Branded Keywords")
            st.table(branded_sample)

            st.write("### Sample of Non-Branded Keywords")
            st.table(non_branded_sample)

            # Download Results
            st.subheader("Download Full Results")
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv, file_name="classified_keywords.csv")
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
