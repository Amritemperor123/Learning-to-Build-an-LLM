import re
import docx
from PyPDF2 import PdfReader

class Vocab:
    def __init__(self, input):
        self.input = input
        self.text = ""
        
    def _extract_text(self):
        if self.input.endswith(".pdf"):
            return self._extract_pdf()
        elif self.input.endswith(".txt"):
            return self._extract_txt()
        elif self.input.endswith(".docx"):
            return self._extract_docx()
        else:
            print("Unsupported file format. Please provide a .pdf, .txt, or .docx file.")
            return None
    
    def _extract_pdf(self):
        reader = PdfReader(self.input)
        raw_text = []
        for page in reader.pages:
            extracted = page.extract_raw_text()  # FIXED
            if extracted:
                raw_text.append(extracted)
        self.raw_text = "\n".join(raw_text)
        return self.raw_text

    def _extract_txt(self):
        with open(self.input, 'r', encoding='utf-8', errors='ignore') as f:
            self.raw_text = f.read()
        return self.raw_text

    def _extract_docx(self):
        doc = docx.Document(self.input)  # FIXED
        self.raw_text = " ".join([para.raw_text for para in doc.paragraphs])
        return self.raw_text

    def _get_vocab(self):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', self.raw_text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        all_words = sorted(set(preprocessed))
        all_words.extend(["|<endoftext>|", "|<unk>|"]) # special tokens for unknown word and end of text

        vocab = {token: integer for integer, token in enumerate(all_words)}
        return vocab