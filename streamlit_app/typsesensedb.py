from qdrant_client import QdrantClient
from qdrantdb import query_qdrant
import streamlit as st

# Function to set up Typesense collection
def setup_typesense_collection(client, collection_name):
    # Check if collection exists and delete if it does
    try:
        client.collections[collection_name].delete()
    except:
        pass
    
    # Create collection schema
    schema = {
        'name': collection_name,
        'fields': [
            {'name': 'id', 'type': 'string'},
            {'name': 'text', 'type': 'string'},
            {'name': 'source', 'type': 'string'},
            {'name': 'type', 'type': 'string'},
            {'name': 'doc_index', 'type': 'int32', 'facet': False}
        ],
        'default_sorting_field': 'doc_index'
    }
    
    # Create collection
    client.collections.create(schema)

# Function to add documents to Typesense
def add_to_typesense(client, collection_name, text, image_analyses):
    # Add main text document
    documents = [{
        'id': '1',
        'text': text,
        'source': 'pdf_text',
        'type': 'text',
        'doc_index': 1
    }]
    
    # Add image analyses as separate documents
    for i, (filename, analysis) in enumerate(image_analyses.items(), start=2):
        documents.append({
            'id': str(i),
            'text': analysis,
            'source': filename,
            'type': 'image_analysis',
            'doc_index': i
        })
    
    # Import documents
    client.collections[collection_name].documents.import_(documents)
    return len(documents)

# Function to search both Qdrant and Typesense
def hybrid_search(query, qdrant_client, typesense_client, collection_name, query_embedding_values):
    try:
        # Semantic search with Qdrant
        qdrant_results = query_qdrant(
            qdrant_client,
            collection_name,
            query_embedding_values
        )
        
        # Keyword search with Typesense
        search_parameters = {
            'q': query,
            'query_by': 'text',
            'sort_by': '_text_match:desc'
        }
        
        try:
            typesense_results = typesense_client.collections[collection_name].documents.search(search_parameters)
            
            # Process Qdrant results
            qdrant_texts = [hit.payload["text"] for hit in qdrant_results]
            
            # Process Typesense results
            typesense_texts = []
            for hit in typesense_results['hits']:
                # Avoid duplication by checking if this text is already in our results
                if hit['document']['text'] not in qdrant_texts and hit['document']['text'] not in typesense_texts:
                    typesense_texts.append(hit['document']['text'])
            
            # Combine texts, prioritizing semantic results first
            all_texts = qdrant_texts + typesense_texts
            
        except Exception as e:
            st.warning(f"Typesense search failed, using only Qdrant results: {e}")
            all_texts = [hit.payload["text"] for hit in qdrant_results]
        
        # Return combined context
        return "\n\n".join(all_texts)
        
    except Exception as e:
        st.error(f"Error in hybrid search: {e}")
        return "Failed to retrieve information. Please try again."