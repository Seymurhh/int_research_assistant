from semantic_search import SemanticSearch
import pandas as pd

def test_search():
    print("Initializing Semantic Search...")
    searcher = SemanticSearch()
    
    papers = [
        {'title': 'Paper 1', 'abstract': 'Deep learning is great for image recognition.', 'published': '2023', 'pdf_url': 'http://a.com'},
        {'title': 'Paper 2', 'abstract': 'Robots use sensors to navigate the world.', 'published': '2023', 'pdf_url': 'http://b.com'},
        {'title': 'Paper 3', 'abstract': 'Quantum computing is the future of processing.', 'published': '2023', 'pdf_url': 'http://c.com'}
    ]
    df = pd.DataFrame(papers)
    
    print("Encoding papers...")
    embeddings = searcher.encode_papers(df['abstract'].tolist())
    
    query = "neural networks for vision"
    print(f"Searching for: '{query}'")
    
    results = searcher.search(query, embeddings, df, top_k=1)
    
    print("Top Result:")
    print(results[0])
    
    assert results[0]['title'] == 'Paper 1', "Search failed to find relevant paper"
    print("Test Passed!")

if __name__ == "__main__":
    test_search()
