#!/usr/bin/env python3
"""
Batch process PDF documents from a folder.

This script processes all PDF files in a given folder and indexes them
into Azure Cognitive Search without using the API endpoint.

Usage:
    python batch_process_documents.py /path/to/folder/with/pdfs
    
    Or with Python:
    python batch_process_documents.py /path/to/folder/with/pdfs
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.pdf_service import PDFService
from app.services.chunk_service import ChunkService
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.db.client import search_client_manager


def find_pdf_files(folder_path: str) -> list:
    """
    Find all PDF files in the given folder.
    
    Args:
        folder_path: Path to folder containing PDF files
        
    Returns:
        List of PDF file paths
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    pdf_files = list(folder.glob("*.pdf"))
    pdf_files.extend(folder.glob("*.PDF"))  # Case-insensitive
    
    return sorted([str(f) for f in pdf_files])


def process_single_document(
    pdf_path: str,
    pdf_service: PDFService,
    chunk_service: ChunkService,
    embedding_service: EmbeddingService,
    search_service: SearchService
) -> dict:
    """
    Process a single PDF document.
    
    Args:
        pdf_path: Path to PDF file
        pdf_service: PDF service instance
        chunk_service: Chunk service instance
        embedding_service: Embedding service instance
        search_service: Search service instance
        
    Returns:
        Dictionary with processing results
    """
    document_name = os.path.basename(pdf_path)
    
    try:
        print(f"\n{'='*60}")
        print(f"📄 Processing: {document_name}")
        print(f"{'='*60}")
        
        # Step 1: Extract text
        print("🔄 Step 1: Extracting text from PDF...")
        pages = pdf_service.extract_text_from_pdf(pdf_path)
        
        if not pages:
            return {
                "status": "error",
                "document_name": document_name,
                "error": "No text extracted from PDF"
            }
        print(f"✅ Step 1: SUCCESS - Extracted {len(pages)} pages")
        
        # Step 2: Chunk text
        print("🔄 Step 2: Chunking text into pieces...")
        chunks = chunk_service.chunk_pages(pages, document_name)
        
        if not chunks:
            return {
                "status": "error",
                "document_name": document_name,
                "error": "No chunks created from document"
            }
        print(f"✅ Step 2: SUCCESS - Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        print("🔄 Step 3: Generating embeddings...")
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = embedding_service.generate_embeddings(chunk_texts)
        
        valid_embeddings = 0
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                chunk.embedding = embedding
                valid_embeddings += 1
        
        print(f"✅ Step 3: SUCCESS - Generated {valid_embeddings} embeddings")
        
        # Step 4: Index chunks
        print("🔄 Step 4: Indexing chunks into Azure Cognitive Search...")
        search_service.index_chunks_batch(chunks, embeddings)
        
        print(f"✅ Step 4: SUCCESS - Indexed {len(chunks)} chunks")
        print(f"✅ {document_name} completed successfully!")
        
        return {
            "status": "success",
            "document_name": document_name,
            "pages_processed": len(pages),
            "chunks_created": len(chunks),
            "chunks_indexed": len(chunks),
            "embeddings_generated": valid_embeddings
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR processing {document_name}: {error_msg}")
        return {
            "status": "error",
            "document_name": document_name,
            "error": error_msg
        }


def main():
    """Main function to batch process documents."""
    if len(sys.argv) < 2:
        print("Usage: python batch_process_documents.py <folder_path>")
        print("\nExample:")
        print("  python batch_process_documents.py ./documents")
        print("  python batch_process_documents.py /path/to/pdf/folder")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    # Find all PDF files
    print("=" * 60)
    print("🔍 Scanning for PDF files...")
    print("=" * 60)
    
    try:
        pdf_files = find_pdf_files(folder_path)
    except ValueError as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {folder_path}")
        sys.exit(1)
    
    print(f"✅ Found {len(pdf_files)} PDF file(s)")
    print("\nFiles to process:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"  {i}. {os.path.basename(pdf_file)}")
    
    # Initialize services
    print("\n" + "=" * 60)
    print("🔧 Initializing services...")
    print("=" * 60)
    
    try:
        pdf_service = PDFService()
        chunk_service = ChunkService()
        embedding_service = EmbeddingService()
        search_client = search_client_manager.get_client()
        search_service = SearchService(search_client)
    except Exception as e:
        print(f"❌ ERROR initializing services: {e}")
        print("\nPlease check your .env file and ensure all Azure credentials are set.")
        sys.exit(1)
    
    # Process all files
    print("\n" + "=" * 60)
    print(f"🚀 Starting batch processing: {len(pdf_files)} files")
    print("=" * 60)
    
    results = []
    total_pages = 0
    total_chunks = 0
    total_embeddings = 0
    
    for file_idx, pdf_path in enumerate(pdf_files, 1):
        result = process_single_document(
            pdf_path,
            pdf_service,
            chunk_service,
            embedding_service,
            search_service
        )
        
        results.append(result)
        
        if result["status"] == "success":
            total_pages += result.get("pages_processed", 0)
            total_chunks += result.get("chunks_created", 0)
            total_embeddings += result.get("embeddings_generated", 0)
    
    # Summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    
    print("\n" + "=" * 60)
    print("🎉 Batch processing completed!")
    print("=" * 60)
    print(f"   ✅ Successful: {successful}/{len(pdf_files)}")
    print(f"   ❌ Failed: {failed}/{len(pdf_files)}")
    print(f"   📄 Total pages: {total_pages}")
    print(f"   📦 Total chunks: {total_chunks}")
    print(f"   🔢 Total embeddings: {total_embeddings}")
    print("=" * 60)
    
    # Detailed results
    print("\n📊 Detailed Results:")
    print("-" * 60)
    for i, result in enumerate(results, 1):
        if result["status"] == "success":
            print(f"✅ {i}. {result['document_name']}")
            print(f"   Pages: {result['pages_processed']}, "
                  f"Chunks: {result['chunks_created']}, "
                  f"Embeddings: {result['embeddings_generated']}")
        else:
            print(f"❌ {i}. {result['document_name']}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Close search client
    search_client_manager.close()
    
    # Exit with error code if any files failed
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All files processed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
