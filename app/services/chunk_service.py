"""Text chunking service for splitting documents into 300-500 token pieces."""
from typing import List
from app.models.chunk import Chunk

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None


class ChunkService:
    """Service for chunking text into token-sized pieces."""
    
    def __init__(self, model_name: str = "gpt-4"):
        """Initialize chunking service with tokenizer."""
        self.use_tiktoken = TIKTOKEN_AVAILABLE
        self.encoding = None
        
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model_name)
            except (KeyError, ValueError):
                try:
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self.use_tiktoken = False
        else:
            self.use_tiktoken = False
        
        self.min_tokens = 300
        self.max_tokens = 500
        self.overlap_tokens = 50
    
    def _sanitize_document_name(self, document_name: str) -> str:
        """Sanitize document name for use in Azure Cognitive Search document keys."""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_=-]', '_', document_name)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.use_tiktoken and self.encoding:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                return self._estimate_tokens(text)
        else:
            return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count when tiktoken is not available."""
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def chunk_text(self, text: str, document_name: str, page_number: int, chunk_index_offset: int = 0) -> List[Chunk]:
        """Split text into chunks of 300-500 tokens."""
        if not text.strip():
            return []
        
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = chunk_index_offset
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                chunk_text = " ".join(current_chunk)
                sanitized_doc_name = self._sanitize_document_name(document_name)
                chunk = Chunk(
                    id=f"{sanitized_doc_name}_page{page_number}_chunk{chunk_index}",
                    content=chunk_text,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    token_count=self.count_tokens(chunk_text)
                )
                chunks.append(chunk)
                
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
                chunk_index += 1
            
            elif sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    sanitized_doc_name = self._sanitize_document_name(document_name)
                    chunk = Chunk(
                        id=f"{sanitized_doc_name}_page{page_number}_chunk{chunk_index}",
                        content=chunk_text,
                        document_name=document_name,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        token_count=self.count_tokens(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                
                words = sentence.split()
                word_chunk = []
                word_tokens = 0
                
                for word in words:
                    word_token_count = self.count_tokens(word + " ")
                    if word_tokens + word_token_count > self.max_tokens and word_chunk:
                        chunk_text = " ".join(word_chunk)
                        sanitized_doc_name = self._sanitize_document_name(document_name)
                        chunk = Chunk(
                            id=f"{sanitized_doc_name}_page{page_number}_chunk{chunk_index}",
                            content=chunk_text,
                            document_name=document_name,
                            page_number=page_number,
                            chunk_index=chunk_index,
                            token_count=self.count_tokens(chunk_text)
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        word_chunk = [word]
                        word_tokens = word_token_count
                    else:
                        word_chunk.append(word)
                        word_tokens += word_token_count
                
                if word_chunk:
                    current_chunk = word_chunk
                    current_tokens = word_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            token_count = self.count_tokens(chunk_text)
            if token_count >= self.min_tokens:
                sanitized_doc_name = self._sanitize_document_name(document_name)
                chunk = Chunk(
                    id=f"{sanitized_doc_name}_page{page_number}_chunk{chunk_index}",
                    content=chunk_text,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    token_count=token_count
                )
                chunks.append(chunk)
        
        return chunks
    
    def chunk_pages(self, pages: List[dict], document_name: str) -> List[Chunk]:
        """Chunk multiple pages of text."""
        all_chunks = []
        chunk_index = 0
        
        for page in pages:
            page_number = page.get("page_number", 1)
            text = page.get("text", "")
            
            page_chunks = self.chunk_text(
                text=text,
                document_name=document_name,
                page_number=page_number,
                chunk_index_offset=chunk_index
            )
            
            all_chunks.extend(page_chunks)
            chunk_index += len(page_chunks)
        
        return all_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str], max_overlap: int = 3) -> List[str]:
        """Get last few sentences for overlap."""
        return sentences[-max_overlap:] if len(sentences) > max_overlap else sentences
