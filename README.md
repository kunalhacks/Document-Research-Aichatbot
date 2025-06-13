# AI Document Research Assistant

A powerful AI-powered document research assistant that helps you analyze and extract insights from multiple documents through natural language queries. This project is part of the Wasserstoff AI Intern Task.

## 🚀 Features

- 📄 **Multi-format Support**: Upload and process PDFs, DOCX, and text documents
- 🔍 **Advanced Search**: Semantic search across multiple documents
- 🤖 **AI-Powered Q&A**: Get accurate answers with source citations
- 🎨 **Theme Analysis**: Automatically identify key themes and topics
- 🖥️ **Streamlit UI**: Clean and intuitive user interface
- 🛠️ **Easy Integration**: Simple setup and configuration

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **AI/ML**: OpenAI GPT, LangChain
- **Vector Database**: FAISS
- **Document Processing**: PyPDF2, python-docx
- **NLP**: spaCy, NLTK

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Tesseract OCR (for image-based PDFs)
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kunalhacks/kunal-sengar-wasserstoff-AiInternTask.git
   cd kunal-sengar-wasserstoff-AiInternTask
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR**
   - Windows: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

5. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🏃 Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface** at `http://localhost:8501`

3. **Upload documents** and start asking questions!

## 🏗️ Project Structure

```
AiInternTask/
├── app/                    # Main application package
│   ├── __init__.py         # Package initialization
│   ├── models/             # Data models
│   │   └── document.py     # Document processing models
│   └── services/           # Business logic
│       ├── document_processor.py  # Document processing logic
│       ├── theme_extractor.py     # Theme extraction logic
│       └── vector_store.py        # Vector storage and search
├── data/                   # Document storage
├── .env.example            # Example environment variables
├── app.py                  # Main Streamlit application
├── main.py                 # CLI entry point
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing UI framework
- [LangChain](https://langchain.com/) for LLM integration
- [OpenAI](https://openai.com/) for their powerful language models
