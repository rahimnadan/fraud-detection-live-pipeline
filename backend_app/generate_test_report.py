import os
import markdown
from weasyprint import HTML
from datetime import datetime

def convert_markdown_to_pdf(md_path, output_path):
    """Convert a markdown file to PDF"""
    print(f"Converting {md_path} to PDF...")
    
    # Read the markdown file
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    
    # Add basic styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
            h2 {{ color: #34495e; }}
            code {{ background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
            pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    HTML(string=styled_html).write_pdf(output_path)
    print(f"Successfully converted {md_path} to PDF")

def main():
    # List of markdown files to include in the report
    md_files = [
        'TEST_DOCUMENTATION_BEFORE_FIXES.md',
        'TEST_DOCUMENTATION_AFTER_FIXES.md',
        'TEST_DOCUMENTATION_FINAL.md'
    ]
    
    # Create output directory if it doesn't exist
    output_dir = 'reports'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert each markdown file to PDF
    pdf_files = []
    for md_file in md_files:
        md_path = os.path.join('tests', md_file)
        if os.path.exists(md_path):
            pdf_path = os.path.join(output_dir, f"{os.path.splitext(md_file)[0]}.pdf")
            convert_markdown_to_pdf(md_path, pdf_path)
            pdf_files.append(pdf_path)
    
    # Merge all PDFs into one report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_report = os.path.join(output_dir, f'test_documentation_report_{timestamp}.pdf')
    
    print(f"\nReport generation complete! Final report saved as: {final_report}")

if __name__ == "__main__":
    main() 