import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Noindex meta tag
st.markdown(
    """
    <head>
        <meta name="robots" content="noindex, nofollow">
    </head>
    """,
    unsafe_allow_html=True

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

# Processing
if st.button("Classify Keywords"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    elif not brand_file or not keyword_file:
        st.error("Please upload both brand terms and keywords.")
    else:
        try:
            # Load files
            brand_terms = pd.read_csv(brand_file)['brand_terms'].tolist()
            keywords = pd.read_csv(keyword_file)['keywords'].tolist()
            
            # Set OpenAI API key
            openai.api_key = api_key
            
            # Generate embeddings
            def get_embeddings(texts):
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=texts
                )
                return [item['embedding'] for item in response['data']]
            
            st.info("Generating embeddings for brand terms...")
            brand_embeddings = np.array(get_embeddings(brand_terms))
            
            st.info("Generating embeddings for keywords...")
            keyword_embeddings = np.array(get_embeddings(keywords))
            
            # Compute cosine similarities
            st.info("Calculating cosine similarities...")
            similarities = cosine_similarity(keyword_embeddings, brand_embeddings)
            
            # Classify keywords
            classifications = []
            for i, sim in enumerate(similarities):
                max_similarity = max(sim)
                classification = "branded" if max_similarity >= threshold else "non-branded"
                classifications.append({
                    "keyword": keywords[i],
                    "classification": classification,
                    "max_similarity": max_similarity
                })
            
            # Convert to DataFrame
            results = pd.DataFrame(classifications)
            st.success("Classification complete!")
            
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
