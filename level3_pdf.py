"""
level3_pdf.py
================
第三层：读取adapted/adapted.html → 用weasyprint转成PDF

比之前的ReportLab版本简单得多，格式由Claude的HTML决定。
"""

import os
import argparse
from pathlib import Path


def run(input_dir: str, output_path: str, worksheet_title: str = ""):
    """
    input_dir:   adapted/ 目录，里面有 adapted.html
    output_path: 输出PDF路径
    """
    html_path = os.path.join(input_dir, "adapted.html")

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"找不到HTML文件: {html_path}")

    try:
        from weasyprint import HTML, CSS
        print(f"用weasyprint转换: {html_path} → {output_path}")
        HTML(filename=html_path).write_pdf(
            output_path,
            stylesheets=[CSS(string="@page { size: A4; margin: 15mm; }")]
        )
        print(f"✅ PDF生成成功: {output_path}")

    except ImportError:
        # weasyprint不可用时fallback：直接把HTML改名为pdf（前端可以serve HTML）
        print("⚠️  weasyprint不可用，使用HTML作为输出")
        import shutil
        shutil.copy(html_path, output_path.replace(".pdf", ".html"))
        # 创建一个简单的PDF重定向文件
        with open(output_path, "wb") as f:
            # 写一个最小PDF，内容是跳转说明
            f.write(b"%PDF-1.4\n")
        raise RuntimeError("weasyprint not available")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--title",       default="")
    args = parser.parse_args()
    run(input_dir=args.input_dir, output_path=args.output, worksheet_title=args.title)
