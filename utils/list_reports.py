import os

def list_existing_reports(tickers):
    report_files = []
    base_dir = "projects/"
    for ticker in tickers:
        project_dir = os.path.join(base_dir, ticker)
        if os.path.exists(project_dir):
            for file in os.listdir(project_dir):
                if file.endswith(".pdf"):
                    report_files.append((ticker, os.path.join(project_dir, file)))
    return report_files
