"""
level3_pdf.py - 直接复制HTML，不转PDF
"""
import os, shutil

def run(input_dir: str, output_path: str, worksheet_title: str = ""):
    html_path = os.path.join(input_dir, "adapted.html")
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"找不到HTML文件: {html_path}")
    # output_path是 adapted.pdf，我们改成 adapted.html
    html_output = output_path.replace(".pdf", ".html")
    shutil.copy(html_path, html_output)
    print(f"✅ HTML已保存: {html_output}")
