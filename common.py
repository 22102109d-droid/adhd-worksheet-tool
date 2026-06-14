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
