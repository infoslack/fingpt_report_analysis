import os
import numpy as np
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib import pagesizes
from reportlab.platypus import SimpleDocTemplate, Frame, Paragraph, Image, PageTemplate, FrameBreak, Spacer, Table, TableStyle, NextPageTemplate, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

load_dotenv()

def generate_pdf(ra, ticker_symbol):
    answer = ra.financial_summarization()

    # 2. Create PDF and insert images
    # page settings
    page_width, page_height = pagesizes.A4
    left_column_width = page_width * 2/3
    right_column_width = page_width - left_column_width
    margin = 4

    # Create PDF document path
    pdf_path = os.path.join(ra.project_dir, f"{ticker_symbol}_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=pagesizes.A4)

    # Define a Frame with two fields
    frame_left = Frame(margin, margin, left_column_width-margin*2, page_height-margin*2, id='left')
    frame_right = Frame(left_column_width, margin, right_column_width-margin*2, page_height-margin*2, id='right')

    left_column_width_p2 = (page_width-margin*3) // 2
    right_column_width_p2 = left_column_width_p2
    frame_left_p2 = Frame(margin, margin, left_column_width_p2-margin*2, page_height-margin*2, id='left')
    frame_right_p2 = Frame(left_column_width_p2, margin, right_column_width_p2-margin*2, page_height-margin*2, id='right')

    # Create a PageTemplate and add it to the document
    page_template = PageTemplate(id='TwoColumns', frames=[frame_left, frame_right])
    page_template_p2 = PageTemplate(id='TwoColumns_p2', frames=[frame_left_p2, frame_right_p2])
    doc.addPageTemplates([page_template, page_template_p2])

    styles = getSampleStyleSheet()

    # Custom style
    custom_style = ParagraphStyle(
        name="Custom",
        parent=styles['Normal'],
        fontName="Helvetica",
        fontSize=10,
        # leading=15,
        alignment=TA_JUSTIFY,
    )

    title_style = ParagraphStyle(
        name="TitleCustom",
        parent=styles['Title'],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        alignment=TA_LEFT,
        spaceAfter=10,
    )

    subtitle_style = ParagraphStyle(
        name="Subtitle",
        parent=styles['Heading2'],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=12,
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    # Prepare left and right column content
    content = []
    # title
    content.append(Paragraph(f"Equity Research Report: {ra.get_company_info()['Company Name']}", title_style))

    # subtitle
    content.append(Paragraph("Income Statement Analysis", subtitle_style))
    content.append(Paragraph(answer['Income Statement Analysis'], custom_style))

    content.append(Paragraph("Balance Sheet Analysis", subtitle_style))
    content.append(Paragraph(answer['Balance Sheet Analysis'], custom_style))

    content.append(Paragraph("Cashflow Analysis", subtitle_style))
    content.append(Paragraph(answer['Cash Flow Analysis'], custom_style))

    content.append(Paragraph("Summarization", subtitle_style))
    content.append(Paragraph(answer['Financial Summary'], custom_style))


    content.append(FrameBreak())  # Used to jump from left column to right column

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 8),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 12),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # First column left aligned
        ('ALIGN', (0,1), (0,-1), 'LEFT'),
        # Second column right aligned
        ('ALIGN', (1,1), (1,-1), 'RIGHT'),
        # Add a horizontal line below the title bar
        ('LINEBELOW', (0,0), (-1,0), 2, colors.black),
    ])
    full_length = right_column_width-2*margin

    rating, _ = ra.get_analyst_recommendations()
    target_price = ra.get_target_price()
    if target_price is not None:
        data = [
            ["Rating:", rating.upper()],
            ["Target Price:", f"{target_price:.2f}"]
        ]
    else:
        data = [["Rating:", rating.upper()]]
    col_widths = [full_length//3*2, full_length//3]
    table = Table(data, colWidths=col_widths)
    table.setStyle(table_style)
    content.append(table)

    # content.append(Paragraph("", custom_style))
    content.append(Spacer(1, 0.15*inch))
    key_data = ra.get_key_data()
    # tabular data
    data = [["Key data", ""]]
    data += [
        [k, v] for k, v in key_data.items()
    ]
    col_widths = [full_length//3*2, full_length//3]
    table = Table(data, colWidths=col_widths)
    table.setStyle(table_style)
    content.append(table)


    # Add Matplotlib image to right column

    # historical stock price
    data = [["Share Performance"]]
    col_widths = [full_length]
    table = Table(data, colWidths=col_widths)
    table.setStyle(table_style)
    content.append(table)

    plot_path = ra.get_stock_performance()
    width = right_column_width
    height = width//2
    content.append(Image(plot_path, width=width, height=height))

    # Historical PE and EPS
    data = [["PE & EPS"]]
    col_widths = [full_length]
    table = Table(data, colWidths=col_widths)
    table.setStyle(table_style)
    content.append(table)

    plot_path = ra.get_pe_eps_performance()
    width = right_column_width
    height = width//2
    content.append(Image(plot_path, width=width, height=height))


    # Start a new page
    content.append(NextPageTemplate('TwoColumns_p2'))
    content.append(PageBreak())

    table_style2 = TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('BACKGROUND', (0, 0), (-1, 0), colors.white),
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 6),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        # First column left aligned
        ('ALIGN', (0,1), (0,-1), 'LEFT'),
        # Second column right aligned
        ('ALIGN', (1,1), (1,-1), 'RIGHT'),
        # Add a horizontal line below the title bar
        ('LINEBELOW', (0,0), (-1,0), 2, colors.black),
        # Add a horizontal line at the bottom of the table
        ('LINEBELOW', (0,-1), (-1,-1), 2, colors.black),
    ])

    # For content on the second page and beyond, use a single-column layout.
    df = ra.get_income_stmt()
    df = df[df.columns[:3]]
    def convert_if_money(value):
        if np.abs(value) >= 1000000:
            return value / 1000000
        else:
            return value

    # Apply transformation function to each column of DataFrame
    df = df.applymap(convert_if_money)

    df.columns = [col.strftime('%Y') for col in df.columns]
    df.reset_index(inplace=True)
    currency = ra.info['currency']
    df.rename(columns={'index': f'FY ({currency} mn)'}, inplace=True)  # Optional: Rename the index column to "serial number"
    table_data = [["Income Statement"]]
    table_data += [df.columns.to_list()] + df.values.tolist()

    table = Table(table_data)
    table.setStyle(table_style2)
    content.append(table)

    content.append(FrameBreak())  # Used to jump from left column to right column

    df = ra.get_cash_flow()
    df = df[df.columns[:3]]

    df = df.applymap(convert_if_money)

    df.columns = [col.strftime('%Y') for col in df.columns]
    df.reset_index(inplace=True)
    currency = ra.info['currency']
    df.rename(columns={'index': f'FY ({currency} mn)'}, inplace=True)  # Optional: Rename the index column to "serial number"
    table_data = [["Cash Flow Sheet"]]
    table_data += [df.columns.to_list()] + df.values.tolist()

    table = Table(table_data)
    table.setStyle(table_style2)
    content.append(table)

    # Build PDF documents
    doc.build(content)
    return pdf_path
