
import streamlit as st
import pandas as pd
import spacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from Bio import Entrez
from collections import Counter
import re
import json
import networkx as nx
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# NCBI Entrez email
Entrez.email = "your_email@example.com"

# Load biomedical NLP model
try:
    nlp = spacy.load("en_core_sci_sm")
except:
    st.error("SciSpaCy model not found. Please install en_core_sci_sm.")
    st.stop()

# Fetch abstracts from PubMed
def fetch_abstracts(search_term, max_results=10):
    handle = Entrez.esearch(db="pubmed", term=search_term, retmax=max_results)
    record = Entrez.read(handle)
    id_list = record["IdList"]
    handle = Entrez.efetch(db="pubmed", id=','.join(id_list), rettype="abstract", retmode="text")
    abstracts_raw = handle.read()
    abstracts = re.split(r'\n\n', abstracts_raw)
    return abstracts

# NLP keyword extraction
def extract_keywords(abstracts):
    disease_terms = []
    for text in abstracts:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_.lower() in ["disease", "condition", "disorder"]:
                disease_terms.append(ent.text.lower())
        for token in doc:
            if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "PROPN"]:
                disease_terms.append(token.lemma_.lower())
    return Counter(disease_terms).most_common(30)

# Word cloud display
def visualize_terms(term_freq):
    if not term_freq:
        st.warning("No terms to display in word cloud.")
        return
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud.generate_from_frequencies(dict(term_freq))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Simulated drug-gene-disease relationships
def get_mock_relationships(key_terms):
    relationships = []
    for i, term in enumerate(key_terms[:5]):
        relationships.append((term[0], f"Gene{i}", "gene"))
        relationships.append((f"Gene{i}", f"Drug{i}", "drug"))
    return relationships

# Network graph visualization
def draw_relationship_network(relationships):
    G = nx.Graph()
    for src, tgt, rel in relationships:
        G.add_node(src, type=rel if rel == "gene" else "term")
        G.add_node(tgt, type="drug" if rel == "gene" else "gene")
        G.add_edge(src, tgt)
    pos = nx.spring_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'),
                            hoverinfo='none', mode='lines')
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            text=node_text, textposition="top center",
                            hoverinfo='text', marker=dict(
                                showscale=False, color=[], size=10, line_width=2))
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(showlegend=False, hovermode='closest',
                                    margin=dict(b=20, l=5, r=5, t=40)))
    st.plotly_chart(fig)

# Text classification using scikit-learn
def run_text_classifier(abstracts):
    if not abstracts or len(abstracts) < 2:
        st.warning("Not enough data to run classifier.")
        return
    labels = ["Disease"] * (len(abstracts) // 2) + ["Non-Disease"] * (len(abstracts) - len(abstracts) // 2)
    X_train, X_test, y_train, y_test = train_test_split(abstracts, labels, test_size=0.4, random_state=42)
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    st.subheader("Text Classification Report (Naive Bayes)")
    st.text(classification_report(y_test, preds))

# Streamlit layout
st.set_page_config(page_title="AI for Drug Discovery", layout="wide")
st.title("AI Strategy for Underserved Disease Discovery")

search_term = st.text_input("Enter a biomedical search term (e.g., 'rare disease', 'orphan drugs', etc.):", "rare diseases")
max_results = st.slider("Number of abstracts to fetch:", 5, 100, 10)

if st.button("Analyze Abstracts"):
    with st.spinner("Fetching abstracts and analyzing..."):
        abstracts = fetch_abstracts(search_term, max_results)
        if not abstracts:
            st.error("No abstracts found.")
        else:
            st.subheader("Sample Abstracts")
            for i, abs_text in enumerate(abstracts[:3]):
                st.markdown(f"**Abstract {i+1}:**\n{abs_text}")

            term_freq = extract_keywords(abstracts)
            st.subheader("Top Terms Extracted")
            df_terms = pd.DataFrame(term_freq, columns=["Term", "Frequency"])
            st.dataframe(df_terms)

            st.subheader("Keyword Frequency Word Cloud")
            visualize_terms(term_freq)

            st.subheader("Drug-Gene-Disease Network (Simulated)")
            relationships = get_mock_relationships(term_freq)
            draw_relationship_network(relationships)

            st.download_button(
                label="Download Keyword Table as CSV",
                data=df_terms.to_csv(index=False).encode('utf-8'),
                file_name='top_keywords.csv',
                mime='text/csv')

            run_text_classifier(abstracts)

st.markdown("---")
st.markdown("Developed as part of L.E.K.'s AI strategy recommendation for biopharma clients.")
