"""
PDF document ingestion service using Azure Document Intelligence via Azure AI Foundry.
"""
import os
import requests
from typing import List, Dict


class PDFService:
    """Service for extracting text from PDF documents using Azure Document Intelligence."""
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY") or os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
        
        if not self.endpoint or not self.api_key:
            self.endpoint = os.getenv("AZURE_PROJECT_ENDPOINT")
            self.api_key = os.getenv("AZURE_PROJECT_API_KEY")
            
            if not self.endpoint or not self.api_key:
                raise ValueError(
                    "Azure Document Intelligence credentials not found. "
                    "Please set either:\n"
                    "1. AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY, or\n"
                    "2. AZURE_PROJECT_ENDPOINT and AZURE_PROJECT_API_KEY in .env"
                )
        
        if "cognitiveservices.azure.com" in self.endpoint:
            base_endpoint = self.endpoint.rstrip('/')
            self.doc_intel_endpoint = f"{base_endpoint}/documentintelligence/documentModels/prebuilt-read:analyze"
        else:
            base_endpoint = self.endpoint.rstrip('/')
            self.doc_intel_endpoint = f"{base_endpoint}/documentintelligence/documentModels/prebuilt-read:analyze"
        
        self.headers = {
            "Content-Type": "application/pdf",
            "Ocp-Apim-Subscription-Key": self.api_key
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Extract text from PDF file with page-level information.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing:
            - page_number: Page number (1-indexed)
            - text: Extracted text from the page
        """
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        return self._extract_with_azure_doc_intel(pdf_bytes)
    
    def extract_text_from_bytes(self, pdf_bytes: bytes, filename: str) -> List[Dict[str, str]]:
        """
        Extract text from PDF bytes (for uploaded files).
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename
            
        Returns:
            List of dictionaries with page_number and text
        """
        return self._extract_with_azure_doc_intel(pdf_bytes)
    
    def _extract_with_azure_doc_intel(self, pdf_bytes: bytes) -> List[Dict[str, str]]:
        """Extract text using Azure Document Intelligence REST API."""
        try:
            response = requests.post(
                self.doc_intel_endpoint,
                headers=self.headers,
                data=pdf_bytes,
                params={"api-version": "2024-11-30"}
            )
            
            if response.status_code != 202:
                raise Exception(f"Failed to submit document for analysis: {response.status_code} - {response.text}")
            
            operation_location = response.headers.get("Operation-Location")
            if not operation_location:
                raise Exception("No operation location returned from Azure")
            
            import time
            max_attempts = 30
            attempt = 0
            
            result_headers = {
                "Ocp-Apim-Subscription-Key": self.api_key
            }
            
            while attempt < max_attempts:
                result_response = requests.get(operation_location, headers=result_headers)
                
                if result_response.status_code != 200:
                    raise Exception(f"Failed to get analysis results: {result_response.status_code} - {result_response.text}")
                
                result_data = result_response.json()
                status = result_data.get("status")
                
                if status == "succeeded":
                    return self._parse_document_intelligence_result(result_data)
                elif status == "failed":
                    error_msg = result_data.get("error", {}).get("message", "Unknown error")
                    raise Exception(f"Document analysis failed: {error_msg}")
                elif status in ["running", "notStarted"]:
                    time.sleep(2)
                    attempt += 1
                else:
                    raise Exception(f"Unexpected status: {status}")
            
            raise Exception("Document analysis timed out")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error during document analysis: {str(e)}")
        except Exception as e:
            raise Exception(f"Error extracting text from PDF with Azure Document Intelligence: {str(e)}")
    
    def _parse_document_intelligence_result(self, result_data: dict) -> List[Dict[str, str]]:
        """Parse the Document Intelligence API result into page-level text."""
        try:
            analyze_result = result_data.get("analyzeResult", {})
            pages_data = analyze_result.get("pages", [])
            content = analyze_result.get("content", "")
            
            pages = []
            
            if pages_data:
                for page_data in pages_data:
                    page_number = page_data.get("pageNumber", 1)
                    lines = page_data.get("lines", [])
                    page_text = "\n".join([line.get("content", "") for line in lines])
                    
                    pages.append({
                        "page_number": page_number,
                        "text": page_text
                    })
            else:
                pages.append({
                    "page_number": 1,
                    "text": content
                })
            
            return pages if pages else [{"page_number": 1, "text": ""}]
            
        except Exception as e:
            raise Exception(f"Error parsing Document Intelligence result: {str(e)}")
