import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Import local modules
import data_loader
import topic_model
import visualize
from semantic_search import SemanticSearch

# Page Config
st.set_page_config(
    page_title="ArXiv Topic Analyzer",
    page_icon="📚",
    layout="wide"
)

# Title and Description
st.title("🧠 Intelligent Research Assistant")
st.markdown("""
This application allows you to explore academic papers from ArXiv using **Topic Modeling (LDA)** and **Semantic Search**.
It demonstrates advanced NLP techniques including:
- **Latent Dirichlet Allocation (LDA)** for discovering hidden topics.
- **Sentence Transformers** for semantic understanding and search.
""")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Data Fetching Config
st.sidebar.subheader("Data Fetching")
query = st.sidebar.text_input("ArXiv Query", value="cat:cs.RO OR cat:cs.AI")
max_results = st.sidebar.slider("Max Papers", 100, 2000, 500)

# Model Config
st.sidebar.subheader("Topic Modeling")
num_topics = st.sidebar.slider("Number of Topics", 2, 20, 5)

# Load Data
with st.spinner("Fetching and processing papers..."):
    df = data_loader.fetch_arxiv_papers(query=query, max_results=max_results)
    
    if df.empty:
        st.error("No papers found. Please try a different query.")
        st.stop()
        
    # Preprocess
    processed_docs = [data_loader.preprocess_text(doc) for doc in df['abstract']]
    
    # Bigrams
    processed_docs = data_loader.make_bigrams(processed_docs)

# Main Content Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Data Explorer", "📊 Topic Modeling", "📈 Trend Analysis", "🚀 Research Direction", "🔍 Semantic Search"])

with tab1:
    st.header("Fetched Papers")
    st.write(f"Total Papers: {len(df)}")
    st.dataframe(df[['title', 'published', 'categories', 'abstract']])

with tab2:
    st.header("Topic Modeling (LDA)")
    
    # Train Model
    dictionary, corpus = topic_model.create_dictionary_corpus(processed_docs)
    lda_model = topic_model.train_lda_model(corpus, dictionary, num_topics=num_topics)
    
    # Show Topics
    st.subheader("Discovered Topics")
    topics = topic_model.get_topics(lda_model)
    for idx, topic in topics:
        st.write(f"**Topic {idx}:** {topic}")
        
    # Interactive Visualization
    st.subheader("Interactive Topic Map")
    try:
        vis_data = visualize.visualize_topics_interactive(lda_model, corpus, dictionary, filepath="lda_visualization.html")
        # Read the saved html file and display it
        with open("lda_visualization.html", 'r') as f:
            html_string = f.read()
        components.html(html_string, width=1300, height=800, scrolling=True)
    except Exception as e:
        st.error(f"Error generating interactive visualization: {e}")

    # Word Clouds
    st.subheader("Topic Word Clouds")
    fig_wc = visualize.create_wordclouds(lda_model, num_topics=num_topics)
    st.pyplot(fig_wc)

    # Topic Insights
    st.subheader("Topic Insights")
    insights_df = visualize.get_topic_insights(df, lda_model, corpus)
    st.dataframe(insights_df)

with tab3:
    st.header("Topic Trends Over Time")
    if 'published' in df.columns:
        # Use Plotly for interactive trend analysis
        fig_trend = visualize.plot_topic_trends_plotly(df, lda_model, corpus)
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Published date not available for trend analysis.")

with tab4:
    st.header("🚀 Research Direction Dashboard")
    st.markdown("""
    This dashboard helps identify **Emerging Trends** in the field. 
    It calculates the growth rate of each topic over time to tell you which areas are heating up.
    """)
    
    if 'published' in df.columns:
        growth_df = visualize.get_topic_growth_metrics(df, lda_model, corpus)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        emerging_topics = growth_df[growth_df['Trend Status'].str.contains("Emerging")]
        fastest_growing = growth_df.iloc[0] if not growth_df.empty else None
        
        with col1:
            st.metric("Emerging Topics", len(emerging_topics))
        with col2:
            if fastest_growing is not None:
                st.metric("Fastest Growing", f"Topic {fastest_growing['Topic ID']}")
            else:
                st.metric("Fastest Growing", "N/A")
        with col3:
             st.metric("Total Papers Analyzed", len(df))

        st.subheader("Topic Growth Analysis")
        st.dataframe(growth_df[['Topic Label', 'Trend Status', 'Growth Score (Slope)', 'Paper Count']], use_container_width=True)
        
        st.info("💡 **Tip**: 'Emerging' topics (Positive Slope) represent good opportunities for new research.")
    else:
        st.warning("Published date not available for growth analysis.")

with tab5:
    st.header("Semantic Search")
    st.markdown("Search for papers using natural language. The model understands the *meaning* of your query, not just keywords.")
    
    search_query = st.text_input("Enter search query", "deep learning for autonomous navigation")
    
    if search_query:
        with st.spinner("Searching..."):
            # Initialize Semantic Search
            searcher = SemanticSearch()
            
            # Encode papers (cached)
            paper_embeddings = searcher.encode_papers(df['abstract'].tolist())
            
            # Perform search
            results = searcher.search(search_query, paper_embeddings, df, top_k=5)
            
            st.subheader("Top Results")
            for i, res in enumerate(results):
                with st.expander(f"{i+1}. {res['title']} (Score: {res['score']:.4f})"):
                    st.write(f"**Published:** {res['published']}")
                    st.write(f"**Abstract:** {res['abstract']}")
                    st.markdown(f"[Read PDF]({res['pdf_url']})")
