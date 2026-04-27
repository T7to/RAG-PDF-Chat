# Mini Project: Local PDF Chat System with RAG

## Goal

Build a system that allows users to:

* Upload a PDF document
* Ask natural language questions
* Receive accurate, context-based answers

The system should:

* Retrieve relevant parts of the document
* Generate answers grounded in the content
* Minimize hallucinations

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system for intelligent question-answering over PDF documents using fully local and open-source tools. It utilizes Ollama to run the LLaMA 3 model locally, enabling efficient, privacy-preserving and cost-free inference without relying on external APIs.

To run this, you need to follow these steps:

1. Install the packages listed in requirements.txt via:
```python
pip install -r requirements.txt
```

 I suggest you to create virtual environment and install these packages in your virtual environment.

 2. If you don't have ollama, go to the official page and install it. Then you can download llama3 model via:

```pythhon
ollama pull llama3
```
3. Upload the pdf file that you prefer in /data. Don't forget to change the name in app.py to match your file.

4. In 1st terminal, start local LLM by,

```python
ollama run llama3
```
5. In 2nd terminal, activate your virtual environemnt (if you created for this project), then run the application by,

```python
python app.py
```