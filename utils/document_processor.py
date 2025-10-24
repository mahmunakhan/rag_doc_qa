import os
import tempfile
from typing import List, Dict, Any
from pypdf import PdfReader
import markdown
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return self._split_text(text, os.path.basename(file_path))
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def process_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return self._split_text(text, os.path.basename(file_path))
        except Exception as e:
            raise Exception(f"Error processing TXT file: {str(e)}")
    
    def process_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_text = file.read()
            
            # Convert markdown to plain text
            html = markdown.markdown(md_text)
            text = self._html_to_text(html)
            
            return self._split_text(text, os.path.basename(file_path))
        except Exception as e:
            raise Exception(f"Error processing Markdown file: {str(e)}")
    
    def _html_to_text(self, html: str) -> str:
        """Simple HTML to text conversion"""
        import re
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def _split_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Split text into chunks and prepare documents"""
        if not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            })
        
        return documents
    
    def process_uploaded_file(self, uploaded_file) -> List[Dict[str, Any]]:
        """Process uploaded file based on its type"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                return self.process_pdf(tmp_path)
            elif file_extension == 'txt':
                return self.process_txt(tmp_path)
            elif file_extension in ['md', 'markdown']:
                return self.process_markdown(tmp_path)
            else:
                raise Exception(f"Unsupported file type: {file_extension}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)