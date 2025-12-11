import os
import nbformat
from nbconvert import PDFExporter
from nbconvert.preprocessors import ExecutePreprocessor
import PyPDF2
from datetime import datetime

def convert_notebook_to_pdf(notebook_path, output_path):
    """Convert a Jupyter notebook to PDF"""
    print(f"Converting {notebook_path} to PDF...")
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Execute the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
    
    # Convert to PDF
    pdf_exporter = PDFExporter()
    pdf_exporter.template_file = 'report'
    pdf_data, _ = pdf_exporter.from_notebook_node(nb)
    
    # Save the PDF
    with open(output_path, 'wb') as f:
        f.write(pdf_data)
    
    print(f"Successfully converted {notebook_path} to PDF")

def merge_pdfs(pdf_files, output_path):
    """Merge multiple PDF files into one"""
    merger = PyPDF2.PdfMerger()
    
    for pdf in pdf_files:
        merger.append(pdf)
    
    merger.write(output_path)
    merger.close()
    print(f"Successfully merged PDFs into {output_path}")

def main():
    # List of notebooks to include in the report
    notebooks = [
        'pre_transaction.ipynb',
        'feature_selection.ipynb',
        'classical_models.ipynb',
        'fit_model.ipynb',
        'further_ann_optimizations.ipynb',
        'training_pipline.ipynb',
        'fraud_detection.ipynb'
    ]
    
    # Create output directory if it doesn't exist
    output_dir = 'reports'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert each notebook to PDF
    pdf_files = []
    for notebook in notebooks:
        notebook_path = os.path.join('jupyter_codes', notebook)
        if os.path.exists(notebook_path):
            pdf_path = os.path.join(output_dir, f"{os.path.splitext(notebook)[0]}.pdf")
            convert_notebook_to_pdf(notebook_path, pdf_path)
            pdf_files.append(pdf_path)
    
    # Merge all PDFs into one report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_report = os.path.join(output_dir, f'fraud_detection_report_{timestamp}.pdf')
    merge_pdfs(pdf_files, final_report)
    
    print(f"\nReport generation complete! Final report saved as: {final_report}")

if __name__ == "__main__":
    main() 