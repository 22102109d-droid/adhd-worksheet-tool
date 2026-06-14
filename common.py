"""
公共模块：模型定义、特征提取、上下文构建等。
训练和推理脚本共用。

标签体系（三分类）:
    I = 0  当前 section 的后续内容
    B = 1  新指令开始
    J = 2  Others（页眉、页脚、标题、装饰性文字等杂项）
"""

import re
import torch
import torch.nn as nn
from transformers import AutoModel


NUM_EXTRA_FEATURES = 6
NUM_CLASSES = 3  # I=0, B=1, J=2

LABEL_MAP = {"I": 0, "B": 1, "J": 2, "S": 0}  # S 向后兼容，归入 I
LABEL_NAMES = ["I-SECTION", "B-SECTION", "J-OTHERS"]

DEFAULT_MODEL = "microsoft/deberta-v3-base"


def build_context_text(lines, idx, window=2):
    """拼接上下文窗口：前 window 行 [SEP] 当前行 [SEP] 后 window 行。"""
    parts = []
    for i in range(max(0, idx - window), idx):
        parts.append(lines[i]["text"])
    parts.append("[CUR] " + lines[idx]["text"] + " [/CUR]")
    for i in range(idx + 1, min(len(lines), idx + window + 1)):
        parts.append(lines[i]["text"])
    return " [SEP] ".join(parts)


def extract_extra_features(line):
    """提取手工特征向量。"""
    f = line.get("features", {})
    return [
        float(f.get("has_number_prefix", False)),
        float(f.get("has_roman_prefix", False)),
        float(f.get("has_letter_prefix", False)),
        float(f.get("has_imperative_verb", False)),
        min(f.get("line_length", 0) / 200.0, 1.0),
        min(f.get("indent_level", 0) / 20.0, 1.0),
    ]


def extract_lines_from_pdf(pdf_path: str) -> list[dict]:
    """从 PDF 提取文本行，保留排版特征。"""
    import pdfplumber

    lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue
            for raw_line in text.split("\n"):
                stripped = raw_line.strip()
                if not stripped:
                    continue

                # 检测答案区域：命中后丢弃当前行及后续所有内容
                if re.match(r"^(Answer\s*Key|Answers?)\s*:?\s*$", stripped, re.IGNORECASE):
                    return lines

                features = {
                    "has_number_prefix": bool(re.match(r"^\d+[\.\)\s]", stripped)),
                    "has_roman_prefix": bool(re.match(r"^(I{1,3}|IV|V|VI{0,3}|IX|X)[\.\)\s]", stripped)),
                    "has_letter_prefix": bool(re.match(r"^[A-Z][\.\)\s]", stripped)),
                    "has_imperative_verb": bool(re.match(
                        r"^(Read|Write|Fill|Match|Circle|Choose|Complete|Listen|Look|"
                        r"Answer|Underline|Check|Put|Make|Find|Draw|Color|Colour|"
                        r"Cross|Tick|Sort|Order|Number|Label|Rewrite|Correct|"
                        r"Translate|Copy|Say|Repeat|Unscramble|Rearrange|Select|"
                        r"Identify|Highlight|Mark|Discuss|Describe|Compare|Explain|"
                        r"Use|Give|List|Name|Solve|Calculate|Count|Trace|Cut|Paste|"
                        r"Glue|Fold|Connect|Join|Link|Group|Classify|Categorize)",
                        stripped, re.IGNORECASE
                    )),
                    "line_length": len(stripped),
                    "indent_level": len(raw_line) - len(raw_line.lstrip()),
                    "page": page_num,
                }
                lines.append({
                    "line_id": len(lines),
                    "text": stripped,
                    "features": features,
                })
    return lines


def extract_lines_with_rich(pdf_path: str) -> list[dict]:
    """
    同时提取纯文本行（供BERT分类）和富文本行（带格式标记，供content字段）。
    返回列表，每个元素包含:
      text      - 纯文本（BERT输入）
      rich      - 富文本（带<b><u><size:><color:>[IMG:]标记）
      features  - 手工特征dict
      page      - 页码
      is_image  - 是否为图片行
    """
    import fitz
    import os
    import re as _re

    doc = fitz.open(pdf_path)
    result = []
    line_id = 0

    for page_num, page in enumerate(doc):
        d = page.get_text("dict")
        page_area = page.rect.width * page.rect.height
        prev_size = None

        for block in d["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                plain_parts = []
                rich_parts = []
                for span in line["spans"]:
                    text = span["text"]
                    if not text.strip():
                        plain_parts.append(text)
                        rich_parts.append(text)
                        continue

                    plain_parts.append(text)

                    cur_size = round(span["size"], 1)
                    if cur_size != prev_size:
                        rich_parts.append(f"<size:{cur_size}>")
                        prev_size = cur_size

                    color = span.get("color", 0)
                    if color != 0:
                        hex_color = f"#{color:06x}" if isinstance(color, int) else str(color)
                        rich_parts.append(f"<color:{hex_color}>")

                    is_bold = bool(span["flags"] & 16) or bool(
                        _re.search(r"bold|black|heavy", span["font"], _re.I))
                    is_underline = bool(span["flags"] & 4)

                    if is_bold and is_underline:
                        rich_parts.append(f"<b><u>{text}</u></b>")
                    elif is_bold:
                        rich_parts.append(f"<b>{text}</b>")
                    elif is_underline:
                        rich_parts.append(f"<u>{text}</u>")
                    else:
                        rich_parts.append(text)

                plain = "".join(plain_parts).strip()
                rich = "".join(rich_parts).strip()

                if not plain:
                    continue

                if _re.match(r"^(Answer\s*Key|Answers?)\s*:?\s*$", plain, _re.IGNORECASE):
                    doc.close()
                    return result

                features = {
                    "has_number_prefix": bool(_re.match(r"^\d+[\.\)\s]", plain)),
                    "has_roman_prefix": bool(_re.match(r"^(I{1,3}|IV|V|VI{0,3}|IX|X)[\.\)\s]", plain)),
                    "has_letter_prefix": bool(_re.match(r"^[A-Z][\.\)\s]", plain)),
                    "has_imperative_verb": bool(_re.match(
                        r"^(Read|Write|Fill|Match|Circle|Choose|Complete|Listen|Look|"
                        r"Answer|Underline|Check|Put|Make|Find|Draw|Color|Colour|"
                        r"Cross|Tick|Sort|Order|Number|Label|Rewrite|Correct|"
                        r"Translate|Copy|Say|Repeat|Unscramble|Rearrange|Select|"
                        r"Identify|Highlight|Mark|Discuss|Describe|Compare|Explain|"
                        r"Use|Give|List|Name|Solve|Calculate|Count|Trace|Cut|Paste|"
                        r"Glue|Fold|Connect|Join|Link|Group|Classify|Categorize)",
                        plain, _re.IGNORECASE
                    )),
                    "line_length": len(plain),
                    "indent_level": 0,
                    "page": page_num,
                }

                result.append({
                    "line_id": line_id,
                    "text": plain,
                    "rich": rich,
                    "features": features,
                    "page": page_num,
                    "is_image": False,
                })
                line_id += 1

        # 图片行
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    r = img_rects[0]
                    w, h = round(r.width, 1), round(r.height, 1)
                    area_ratio = round((r.width * r.height) / page_area * 100) if page_area else 0
                    img_filename = f"page{page_num+1}_img{img_index+1}.png"
                    marker = f"[IMG: {img_filename}, {w}x{h}pt, 占页面{area_ratio}%]"
                else:
                    img_filename = f"page{page_num+1}_img{img_index+1}.png"
                    marker = f"[IMG: {img_filename}]"

                result.append({
                    "line_id": line_id,
                    "text": "",
                    "rich": marker,
                    "features": {},
                    "page": page_num,
                    "is_image": True,
                })
                line_id += 1
            except Exception:
                pass

    doc.close()
    return result


class WorksheetSegmenter(nn.Module):
    """预训练模型 + 手工特征 → B/I/J 三分类。
    支持 BERT、DeBERTa、XLM-RoBERTa 等任意 HuggingFace 模型。"""

    def __init__(self, bert_name=DEFAULT_MODEL,
                 num_extra=NUM_EXTRA_FEATURES, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(bert_name)
        hidden = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden + num_extra, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, attention_mask, features):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_output, features], dim=1)
        logits = self.classifier(combined)
        return logits
