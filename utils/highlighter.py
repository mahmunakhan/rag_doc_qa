"""
Document Highlighter for RAG System
Highlights relevant text in PDF, DOCX, TXT, and MD files
"""

import fitz  # PyMuPDF for PDFs
from PIL import Image
import os
from docx import Document
import re
import markdown
from typing import List, Tuple, Union


class DocumentHighlighter:
    """Highlights answers in various document formats"""
    
    def __init__(self, vertical_padding: int = 75, proximity_threshold: int = 50):
        """
        Initialize highlighter
        
        Args:
            vertical_padding: Pixels to add above/below highlighted region
            proximity_threshold: Max distance for search term and answer to be considered related
        """
        self.vertical_padding = vertical_padding
        self.proximity_threshold = proximity_threshold
    
    def highlight_document(
        self,
        file_path: str,
        search_sentence: str,
        answer: str
    ) -> Union[List[Tuple[Image.Image, int]], str]:
        """
        Highlights occurrences of the answer near the search sentence
        
        Args:
            file_path: Path to the document (PDF, DOCX, TXT, or MD)
            search_sentence: Sentence or phrase to locate near the answer
            answer: The text to highlight
        
        Returns:
            For PDFs → List of (PIL.Image, page_number) tuples
            For DOCX, TXT, MD → HTML string with highlighted text
        """
        if not os.path.exists(file_path):
            return f"<p style='color: red;'>File not found: {file_path}</p>"
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == ".pdf":
                return self._highlight_pdf(file_path, search_sentence, answer)
            elif file_ext == ".docx":
                return self._highlight_docx(file_path, answer)
            elif file_ext == ".txt":
                return self._highlight_text(file_path, answer)
            elif file_ext in [".md", ".markdown"]:
                return self._highlight_markdown(file_path, answer)
            else:
                return f"<p style='color: orange;'>Unsupported file format: {file_ext}</p>"
        
        except Exception as e:
            return f"<p style='color: red;'>Error highlighting: {str(e)}</p>"
    
    def _highlight_pdf(
        self,
        pdf_path: str,
        search_sentence: str,
        answer: str
    ) -> List[Tuple[Image.Image, int]]:
        """Highlight text in PDF and return images of highlighted regions"""
        
        pdf_document = fitz.open(pdf_path)
        images = []
        
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            
            # Search for both the search term and the answer
            rects_search_term = page.search_for(search_sentence)
            rects_answer = page.search_for(answer)
            
            highlighted_rects = []
            
            # Check if search term and answer are near each other
            for rect_search in rects_search_term:
                for rect_answer in rects_answer:
                    # If they're close vertically, highlight the answer
                    if abs(rect_search.y0 - rect_answer.y0) <= self.proximity_threshold:
                        page.add_highlight_annot(rect_answer)
                        highlighted_rects.append(rect_answer)
                        break
            
            # If we found highlights on this page, create an image
            if highlighted_rects:
                # Calculate bounding box for all highlights
                combined_rect = fitz.Rect(
                    page.rect.x0,
                    page.rect.y1,
                    page.rect.x1,
                    page.rect.y0
                )
                
                for rect in highlighted_rects:
                    combined_rect.y0 = min(combined_rect.y0, rect.y0 - self.vertical_padding)
                    combined_rect.y1 = max(combined_rect.y1, rect.y1 + self.vertical_padding)
                
                # Render the highlighted region
                mat = fitz.Matrix(4.0, 4.0)  # High resolution
                pix = page.get_pixmap(clip=combined_rect, matrix=mat, dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append((img, page_number + 1))
        
        pdf_document.close()
        return images
    
    def _highlight_docx(self, file_path: str, answer: str) -> str:
        """Highlight text in DOCX and return HTML"""
        
        doc = Document(file_path)
        highlighted_html = "<div style='background: white; padding: 20px; border-radius: 5px; color: black;'>"
        
        for para in doc.paragraphs:
            text = para.text
            
            if answer.lower() in text.lower():
                # Highlight the answer with a yellow background
                new_text = re.sub(
                    rf"({re.escape(answer)})",
                    r"<mark style='background-color: yellow; padding: 2px; font-weight: bold;'>\1</mark>",
                    text,
                    flags=re.IGNORECASE
                )
                highlighted_html += f"<p>{new_text}</p>"
            else:
                highlighted_html += f"<p>{text}</p>"
        
        highlighted_html += "</div>"
        return highlighted_html
    
    def _highlight_text(self, file_path: str, answer: str) -> str:
        """Highlight text in TXT file and return HTML"""
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Highlight the answer
        highlighted = re.sub(
            rf"({re.escape(answer)})",
            r"<mark style='background-color: yellow; padding: 2px; font-weight: bold;'>\1</mark>",
            content,
            flags=re.IGNORECASE
        )
        
        # Convert to HTML with preserved formatting
        html_content = f"""
        <div style='background: white; padding: 20px; border-radius: 5px; color: black; white-space: pre-wrap; font-family: monospace;'>
            {highlighted}
        </div>
        """
        
        return html_content
    
    def _highlight_markdown(self, file_path: str, answer: str) -> str:
        """Highlight text in Markdown file and return HTML"""
        
        with open(file_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        # Highlight the answer in markdown
        highlighted_md = re.sub(
            rf"({re.escape(answer)})",
            r"<mark style='background-color: yellow; padding: 2px; font-weight: bold;'>\1</mark>",
            md_content,
            flags=re.IGNORECASE
        )
        
        # Convert Markdown to HTML
        html_content = markdown.markdown(highlighted_md, extensions=['extra', 'codehilite'])
        
        # Wrap in styled container
        styled_html = f"""
        <div style='background: white; padding: 20px; border-radius: 5px; color: black;'>
            {html_content}
        </div>
        """
        
        return styled_html


def highlight_in_chunk(chunk_content: str, answer: str) -> str:
    """
    Intelligently highlights relevant information in chunk based on answer
    Extracts key entities, numbers, and phrases for matching
    
    Args:
        chunk_content: The text chunk to highlight in
        answer: The full answer text
    
    Returns:
        HTML string with selective highlighting of matching content
    """
    import re
    
    chunk_lower = chunk_content.lower()
    answer_lower = answer.lower()
    
    # Extract all numbers from answer (very important for factual answers)
    numbers_in_answer = re.findall(r'\d+\.?\d*', answer)
    
    # Extract key phrases (noun phrases, important terms)
    # Look for quoted text, capitalized terms, or words after ":" 
    key_terms = set()
    
    # Add numbers as key terms
    for num in numbers_in_answer:
        key_terms.add(num)
    
    # Extract multi-word phrases (2-6 words) from answer
    answer_words = answer_lower.split()
    for length in [6, 5, 4, 3, 2]:
        for i in range(len(answer_words) - length + 1):
            phrase = ' '.join(answer_words[i:i+length])
            # Only consider phrases that appear in chunk
            if phrase in chunk_lower and len(phrase) > 8:  # At least 8 chars
                key_terms.add(phrase)
    
    # Extract important single words (capitalized, specific terms)
    important_words = ['maternity', 'paternity', 'leave', 'vacation', 'sick', 'days', 'weeks', 
                      'months', 'hours', 'paid', 'unpaid', 'annual', 'monthly', 'weekly',
                      'salary', 'bonus', 'insurance', 'benefits', 'policy', 'eligible',
                      'accrual', 'notice', 'required', 'provided', 'entitled', 'employee']
    
    for word in answer_words:
        if len(word) > 3 and (word in important_words or word.isdigit()):
            if word in chunk_lower:
                key_terms.add(word)
    
    # Apply highlighting to ALL matching terms/phrases
    highlighted = chunk_content
    
    # Sort by length (longest first) to avoid partial replacements
    sorted_terms = sorted(key_terms, key=len, reverse=True)
    
    for term in sorted_terms:
        if term and len(term) > 1:  # Skip empty or single char
            # Create case-insensitive pattern
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            
            # Check if term exists before highlighting
            if pattern.search(highlighted):
                highlighted = pattern.sub(
                    lambda m: f"<mark style='background-color: yellow; padding: 1px 3px; font-weight: 500;'>{m.group(0)}</mark>",
                    highlighted
                )
    
    return highlighted