import os
import streamlit as st
from report_analysis import ReportAnalysis
from pdf_generator import generate_pdf
from utils.list_reports import list_existing_reports

tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'INTC', 'AMD', 'ORCL']
st.title('Financial Report Analysis')

ticker_symbol = st.selectbox("Choose the company ticker:", tickers)
if st.button('Generate the report'):
    ra = ReportAnalysis(ticker_symbol)
    pdf_path = generate_pdf(ra, ticker_symbol)
    st.write("Reports:", pdf_path)
    with open(pdf_path, "rb") as file:
        st.download_button(label="Download PDF", data=file, file_name=f"{ticker_symbol}_report.pdf")

st.subheader("All Reports")
report_files = list_existing_reports(tickers)
for ticker, full_path in report_files:
    with open(full_path, "rb") as file:
        st.download_button(label=f"Download {ticker}", data=file, file_name=os.path.basename(full_path))
