## FinGPT - Financial Report Analysis
This project provides a tool enabling users to analyze financial reports, explicitly focusing on annual reports (10-K). The tool incorporates advanced language models like GPT-4 or other locally deployed Large Language Models (LLM) to help users generate detailed analysis reports in PDF format. These reports provide insights into a company's financial health and performance over the fiscal year.

#### Features
- **Customizable Analysis:** the analysis scope can be modified by choosing different company symbols.
- **PDF Report Generation:**: generate detailed analysis reports in PDF format.
- **RAG Support:**: the power of RAG is used in the construction of the report, and the base created can be used for question-answering and summary tasks.

#### Requirements
Before starting, ensure you have the following installed:

- Python 3.11 or later
- SEC-API, which is used to grab the 10-K report: https://sec-api.io/profile
- (Optional) [FMP API](https://site.financialmodelingprep.com/developer/docs/dashboard) for target price
- [Qdrant cloud account](https://qdrant.to/cloud)

#### Setup

Create a new environment:
```bash
conda create --prefix ./env python=3.11
```
Activate the new environment:
```bash
conda activate ./env
```
Install  the requirements:
```bash
pip install -r requirements.txt
```

Prepare credentials:
```bash
cp .env.example .env
```
and complete the `.env`file with your external services credentials.

Run:
```bash
streamlit run  main.py
```