# ML-NLP-for-Drug-Discovery-in-Under_served-Diseases
Python program of an AI strategy for a biopharmaceutical company, with a focus on using Natural Language Processing (NLP) to identify underserved diseases by mining biomedical literature (e.g., PubMed abstracts)


## AI Strategy Tool for Drug Discovery Using NLP + ML created for L.E.K. Consulting 
This Streamlit-based Python application helps biopharmaceutical company use AI and NLP for drug discovery and research related to underserved diseases. The result of running the code is a Streamlit App.

## Defintions 
- Streamlit: An open-source Python framework that allows you to quickly build interactive web apps for data science and machine learning — using pure Python.

## What the Python Script Does
- Uses SciSpaCy, a biomedical NLP model, to parse abstracts.
- Includes a Scikit-learn text classifier using a Naive Bayes model (ML model) to classify abstracts with Scikit-learn and displays performance.
- Extracts disease-related terms using custom logic and entity recognition.
- Displays a word cloud to visually highlight frequently mentioned conditions, useful for identifying research gaps in underserved diseases.
- The Python script creates a Streamlit app that allows users to do the following steps:
    - Lets you input search terms.
    - Fetches relevant abstracts from PubMed using the Entrez API.
    - Applies SciSpaCy to extract disease-related and keyword information.
    - Visualizes the results using a word cloud.
    - Lays the foundation for drug-gene relationship exploration and AI prioritization.

---

## The app created by the Python program when the user runs it does the below features:

## 🚀 Features

### 🔍 1. Abstract Retrieval
- Fetches biomedical literature abstracts using **NCBI Entrez API**.
- User specifies search terms and number of results.

### 🧠 2. NLP Keyword Extraction
- Uses **SciSpaCy**, a biomedical NLP model, to extract meaningful terms from abstracts.
- Highlights disease terms, gene mentions, and key medical concepts.
- Displays top keywords and frequencies in a data table.

### ☁️ 3. Word Cloud Visualization
- Generates a visual representation of the most frequent terms.
- Easy identification of thematic relevance in the biomedical space.

### 🧬 4. Drug–Gene–Disease Network
- Shows relationships between extracted terms, genes, and drugs.
- Interactive graph built with **NetworkX + Plotly**.

### 🤖 5. Text Classification with Scikit-learn
- Demonstrates classification of abstracts into `Disease` and `Non-Disease` using **Naive Bayes**.
- Includes vectorization (`TfidfVectorizer`) and pipeline integration.
- Outputs a **classification report** (precision, recall, F1-score).

### 📤 6. Data Export
- Downloadable CSV with extracted terms and their frequencies.
- Prepares data for external tools like Power BI.

---

## 🧱 Tech Stack
- **Streamlit** for web interface
- **BioPython / Entrez** for PubMed access
- **SciSpaCy** for biomedical NLP
- **Matplotlib / WordCloud / Plotly / NetworkX** for visualizations
- **Scikit-learn** for machine learning

---

## 📦 How to Run
```bash
# Install required packages
pip install streamlit biopython scispacy spacy matplotlib wordcloud plotly networkx scikit-learn
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz

# Run the app
streamlit run app.py
```

---

## 🌐 Use Case
Built for strategic AI consulting in biopharma (L.E.K. Consulting), this tool showcases:
- How to operationalize NLP in R&D
- Identifying trends in neglected or rare diseases
- Building an AI pipeline from text ingestion to model-based insight

---

## ✨ Future Enhancements
- Integrate real drug–gene–disease databases (e.g., DisGeNET, DrugBank)
- Add timeline filters and MeSH term drilldowns
- Deploy via Docker or Streamlit Cloud

---

## 📫 Contact
For improvements, reach out to your AI strategy or analytics lead.
