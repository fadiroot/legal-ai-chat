"""Answer generation service using Azure OpenAI GPT models."""
import os
from typing import List, Dict, Optional
from openai import AzureOpenAI


class AnswerService:
    """Service for generating natural language answers from retrieved chunks using Azure OpenAI."""
    
    def __init__(self):
        self.aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.aoai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.aoai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        self.aoai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        
        if not self.aoai_endpoint or not self.aoai_api_key:
            raise ValueError(
                "Azure OpenAI credentials not found. "
                "Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env"
            )
        
        if not self.aoai_deployment:
            raise ValueError(
                "AZURE_OPENAI_DEPLOYMENT_NAME is required. "
                "Please set it in your .env file with the name of your GPT deployment (e.g., gpt-4o)."
            )
        
        self._client: Optional[AzureOpenAI] = None
    
    def _get_client(self) -> AzureOpenAI:
        """Get or create Azure OpenAI client."""
        if self._client is None:
            self._client = AzureOpenAI(
                api_key=self.aoai_api_key,
                api_version=self.aoai_api_version,
                azure_endpoint=self.aoai_endpoint
            )
        return self._client
    
    def _build_context_from_chunks(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk.get("document_name", "Unknown")
            page_num = chunk.get("page_number", 0)
            content = chunk.get("content", "")
            
            context_parts.append(
                f"[المصدر {i}: {doc_name}, الصفحة {page_num}]\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, chunks: List[Dict], language: Optional[str] = None) -> str:
        """Generate a natural language answer from retrieved chunks."""
        if not chunks:
            return "المعلومة غير موجودة في الوثائق." if language == "ar" else "No relevant information found in the documents."
        
        client = self._get_client()
        context = self._build_context_from_chunks(chunks)
        
        if language is None:
            has_arabic = any('\u0600' <= char <= '\u06FF' for char in question)
            language = "ar" if has_arabic else "en"
        system_prompt = """أنت مساعد قانوني ذكي. مهمتك هي الإجابة على الأسئلة بناءً على السياق المقدم فقط.

القواعد المهمة:
1. استخدم فقط المعلومات الموجودة في السياق المقدم
2. إذا كان السؤال يتعلق بموضوع محدد (مثل "عقد") ولكن السياق يتحدث عن موضوع مختلف (مثل "لائحة الأجور"):
   - اذكر أن المعلومات المطلوبة غير موجودة
   - لكن قدم ملخصاً مفيداً عما وجدته في السياق إذا كان ذا صلة
   - اشرح نوع الوثائق الموجودة
3. إذا لم تجد أي معلومات ذات صلة على الإطلاق، قل: "المعلومة غير موجودة في الوثائق"
4. أجب بالعربية إذا كان السؤال بالعربية، وبالإنجليزية إذا كان السؤال بالإنجليزية
5. قدم إجابات واضحة ومفصلة
6. أشر إلى المصادر عند الإمكان (اسم الوثيقة ورقم الصفحة)

You are an intelligent legal assistant. Your task is to answer questions based only on the provided context.

Important rules:
1. Use only information found in the provided context
2. If the question asks about a specific topic (e.g., "contract") but the context discusses a different topic (e.g., "wage regulations"):
   - State that the requested information is not found
   - But provide a helpful summary of what you found in the context if it's related
   - Explain what type of documents are available
3. If you find no relevant information at all, state: "The information is not found in the documents"
4. Answer in Arabic if the question is in Arabic, and in English if the question is in English
5. Provide clear and detailed answers
6. Reference sources when possible (document name and page number)"""
        
        user_prompt = f"""السياق (Context):
{context}

السؤال (Question):
{question}

يرجى الإجابة بناءً على السياق أعلاه. إذا كان السؤال يتعلق بموضوع محدد ولكن السياق يتحدث عن موضوع مختلف:
- اذكر أن المعلومات المطلوبة غير موجودة
- لكن قدم ملخصاً عما وجدته في الوثائق المتاحة

Please answer based on the context above. If the question asks about a specific topic but the context discusses a different topic:
- State that the requested information is not found
- But provide a summary of what information is available in the documents"""
        
        try:
            response = client.chat.completions.create(
                model=self.aoai_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception:
            if chunks:
                fallback_answer = chunks[0].get("content", "")
                return f"[Note: LLM generation failed, showing raw chunk]\n\n{fallback_answer}"
            
            return "المعلومة غير موجودة في الوثائق." if language == "ar" else "No relevant information found in the documents."
