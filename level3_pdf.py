"""
level3_pdf.py
================
第三层：读取adapted/adapted.html → 用xhtml2pdf转成PDF
不需要系统级依赖，纯Python。
"""

import os
import argparse


def run(input_dir: str, output_path: str, worksheet_title: str = ""):
    html_path = os.path.join(input_dir, "adapted.html")

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"找不到HTML文件: {html_path}")

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    from xhtml2pdf import pisa
    print(f"用xhtml2pdf转换: {html_path} → {output_path}")
    with open(output_path, "wb") as f:
        result = pisa.CreatePDF(html, dest=f)

    if result.err:
        raise RuntimeError(f"xhtml2pdf转换失败: {result.err}")

    print(f"✅ PDF生成成功: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--title",     default="")
    args = parser.parse_args()
    run(input_dir=args.input_dir, output_path=args.output, worksheet_title=args.title)
