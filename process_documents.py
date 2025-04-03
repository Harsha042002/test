import os
import argparse
import asyncio
import tiktoken
import chromadb
import voyageai
import numpy as np
from dotenv import load_dotenv

load_dotenv()

tokenizer = tiktoken.get_encoding("cl100k_base")
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

class DocumentProcessor:
    def __init__(self, collection_name="fresh_bus_kb"):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = None
        self.collection_name = collection_name
        self.chunk_size = 512
        self.chunk_overlap = 100
        self.initialize_collection()
    
    def initialize_collection(self):
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Collection '{self.collection_name}' initialized")
            return True
        except Exception as e:
            print(f"Error initializing collection: {e}")
            return False
    
    def chunk_text(self, text, document_id):
        tokens = tokenizer.encode(text)
        chunks = []
        chunk_ids = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunk_id = f"{document_id}_{i}"
                chunks.append(chunk_text)
                chunk_ids.append(chunk_id)
        
        return chunks, chunk_ids
    
    async def process_document(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                document_text = file.read()
            
            document_id = os.path.basename(file_path)
            chunks, chunk_ids = self.chunk_text(document_text, document_id)
            
            embeddings = []
            batch_size = 32
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                response = vo.embed(batch, model="voyage-3", input_type="document")
                embeddings.extend(response.embeddings)
            
            # Save embeddings to numpy file
            np.save(f"embeddings_{document_id}.npy", np.array(embeddings))
            print(f"Saved embeddings to embeddings_{document_id}.npy")
            
            metadatas = [{"source": document_id, "chunk_index": i} for i in range(len(chunks))]
            
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=chunk_ids,
                metadatas=metadatas
            )
            
            print(f"Added {len(chunks)} chunks from {document_id}")
            return len(chunks)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return 0

async def process_single_file(file_path, collection_name="fresh_bus_kb"):
    processor = DocumentProcessor(collection_name)
    chunks_added = await processor.process_document(file_path)
    print(f"Processed {chunks_added} chunks from {os.path.basename(file_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process document")
    parser.add_argument("--file", required=True, help="Path to document")
    parser.add_argument("--collection", default="fresh_bus_kb", help="Collection name")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found")
        exit(1)
    
    asyncio.run(process_single_file(args.file, args.collection))