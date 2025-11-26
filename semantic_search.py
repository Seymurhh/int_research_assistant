import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = self._load_model()

    @st.cache_resource
    def _load_model(_self):
        """
        Loads the SentenceTransformer model. Cached by Streamlit.
        """
        return SentenceTransformer(_self.model_name)

    @st.cache_data
    def encode_papers(_self, papers_text):
        """
        Encodes a list of paper texts (e.g., abstracts) into embeddings.
        """
        return _self.model.encode(papers_text, convert_to_tensor=True)

    def search(self, query, paper_embeddings, papers_df, top_k=5):
        """
        Searches for the most relevant papers given a query.
        """
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarity
        cos_scores = util.cos_sim(query_embedding, paper_embeddings)[0]
        
        # Find top_k results
        top_results = torch.topk(cos_scores, k=min(top_k, len(papers_df)))
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            paper = papers_df.iloc[idx.item()]
            results.append({
                'score': score.item(),
                'title': paper['title'],
                'abstract': paper['abstract'],
                'published': paper['published'],
                'pdf_url': paper['pdf_url']
            })
            
        return results
