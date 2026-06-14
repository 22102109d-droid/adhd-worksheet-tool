"""
level3_pdf.py
================
第三层：读取第二层输出的 adapted_chunk_*.json → 用ReportLab渲染ADHD友好版PDF

设计原则：
  - 流式布局 (continuous flow)，不强制每个chunk一页，避免空白
  - KeepTogether 保证 task标题+指令+前几行答题区 不被孤立切断
  - 渐进式 Task N/Total 进度指示
  - 各类型chunk有对应的渲染规则（fill_in_the_blank / multiple_choice / discussion / source_text等）
  - key_words_box / task_steps / image_suggestion 按第二层标注渲染
  - [IMG:...] / <b> <color:#xxx> <size:x> <u> 等富文本标记转换为ReportLab样式

用法：
  python level3_pdf.py --input_dir ~/Desktop/output_A1/worksheet_name/adapted \
                        --output ~/Desktop/output_A1/worksheet_name/adapted.pdf \
                        --title "Born, Bred and Buttered in London"
"""

import json
import os
import re
import argparse
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, HRFlowable, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 注册支持Unicode符号(▶ ☐ 等)的字体，Helvetica内置编码无法渲染这些字符
import os
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", os.path.join(FONT_DIR, "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans-BoldOblique", os.path.join(FONT_DIR, "DejaVuSans-BoldOblique.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
                    italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-BoldOblique")


# ================================================================
# 颜色与样式常量
# ================================================================
C_AMBER      = HexColor("#D98E2B")
C_AMBER_LT   = HexColor("#FBF3E6")
C_GREEN      = HexColor("#1D9E75")
C_GREEN_LT   = HexColor("#E7F5EF")
C_BLUE       = HexColor("#3B6FA8")
C_BLUE_LT    = HexColor("#EAF1F8")
C_TEXT       = HexColor("#3A3530")
C_TEXT_LIGHT = HexColor("#7A736A")
C_BORDER     = HexColor("#E5DCC8")
C_LINE       = HexColor("#C9BFA8")

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm


def build_styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle("WSTitle", parent=ss["Title"],
        fontName="DejaVuSans-Bold", fontSize=20, textColor=C_TEXT,
        spaceAfter=2, leading=24))

    ss.add(ParagraphStyle("WSSubtitle", parent=ss["Normal"],
        fontName="DejaVuSans-Oblique", fontSize=10, textColor=C_TEXT_LIGHT,
        spaceAfter=10))

    ss.add(ParagraphStyle("TaskHeader", parent=ss["Heading2"],
        fontName="DejaVuSans-Bold", fontSize=13, textColor=C_TEXT,
        spaceBefore=4, spaceAfter=4, leading=16))

    ss.add(ParagraphStyle("TaskProgress", parent=ss["Normal"],
        fontName="DejaVuSans", fontSize=8, textColor=C_TEXT_LIGHT,
        alignment=TA_LEFT, spaceAfter=2))

    ss.add(ParagraphStyle("Instruction", parent=ss["Normal"],
        fontName="DejaVuSans-Bold", fontSize=10.5, textColor=C_TEXT,
        leading=14, spaceAfter=6))

    ss.add(ParagraphStyle("Body", parent=ss["Normal"],
        fontName="DejaVuSans", fontSize=10.5, textColor=C_TEXT,
        leading=15, spaceAfter=4))

    ss.add(ParagraphStyle("BodyTight", parent=ss["Normal"],
        fontName="DejaVuSans", fontSize=10.5, textColor=C_TEXT,
        leading=14, spaceAfter=2))

    ss.add(ParagraphStyle("KeyWordTerm", parent=ss["Normal"],
        fontName="DejaVuSans-Bold", fontSize=10, textColor=C_GREEN,
        leading=13))

    ss.add(ParagraphStyle("KeyWordDef", parent=ss["Normal"],
        fontName="DejaVuSans", fontSize=9.5, textColor=C_TEXT,
        leading=13))

    ss.add(ParagraphStyle("BoxHeader", parent=ss["Normal"],
        fontName="DejaVuSans-Bold", fontSize=9, textColor=C_TEXT_LIGHT,
        leading=12, spaceAfter=4))

    ss.add(ParagraphStyle("TipText", parent=ss["Normal"],
        fontName="DejaVuSans-Oblique", fontSize=9.5, textColor=C_AMBER,
        leading=13))

    ss.add(ParagraphStyle("StepText", parent=ss["Normal"],
        fontName="DejaVuSans", fontSize=10.5, textColor=C_TEXT,
        leading=15, spaceAfter=3, leftIndent=4))

    ss.add(ParagraphStyle("ImgPlaceholder", parent=ss["Normal"],
        fontName="DejaVuSans-Oblique", fontSize=9, textColor=C_TEXT_LIGHT,
        leading=12, alignment=TA_CENTER))

    ss.add(ParagraphStyle("CaptionText", parent=ss["Normal"],
        fontName="DejaVuSans-Oblique", fontSize=8, textColor=C_TEXT_LIGHT,
        leading=11, spaceAfter=2, spaceBefore=4))

    ss.add(ParagraphStyle("FooterText", parent=ss["Normal"],
        fontName="DejaVuSans", fontSize=8, textColor=C_TEXT_LIGHT))

    return ss


STYLES = build_styles()


def split_by_font_size(text: str, size_drop_threshold: float = 1.0):
    """
    把content按<size:x>标记分割成"主体文本"和"小字说明文本"两部分。

    逻辑:
      - 找到第一个出现的<size:x>值作为该chunk的"主体字号"基准
        (没有size标记则视为正文默认字号,不触发分离)
      - 后续出现的<size:y>，若 y <= 主体字号 - size_drop_threshold，
        则该段及其后内容(直到下一个size标记或结尾)归为"小字说明"
      - 小字说明片段中的[IMG:...]标记会被保留供后续处理，但通常这些
        图片已经在chunk的image_files里处理过，这里仅处理文字部分

    返回: (main_markup, caption_markup_list)
      main_markup: 主体部分的富文本(含原size标记，交给rich_to_reportlab处理)
      caption_markup_list: 小字说明部分的纯文本列表(已去除markup标记)
    """
    if not text:
        return "", []

    # 按<size:x>标记切分，保留分隔符
    parts = re.split(r"(<size:[\d.]+>)", text)

    main_segments = []
    caption_segments = []

    base_size = None
    current_is_caption = False
    seen_any_text = False

    for part in parts:
        m = re.match(r"<size:([\d.]+)>", part)
        if m:
            sz = float(m.group(1))
            if base_size is None:
                if seen_any_text:
                    # 已经有未标记字号的正文出现过 -> 默认正文字号视为基准(更大)
                    # 此处size标记的内容相对更小，归为caption
                    base_size = sz + size_drop_threshold  # 确保下方判断成立
                    current_is_caption = (sz <= base_size - size_drop_threshold)
                else:
                    # 整个chunk第一个内容就是size标记 -> 以此为基准
                    base_size = sz
                    current_is_caption = False
            else:
                current_is_caption = (sz <= base_size - size_drop_threshold)
            continue

        if not part:
            continue

        seen_any_text = True
        if current_is_caption:
            caption_segments.append(part)
        else:
            main_segments.append(part)

    main_markup = "".join(main_segments)

    # caption部分：去除markup标记和[IMG:...]，只保留纯文字，按空白合并
    caption_texts = []
    for seg in caption_segments:
        plain = re.sub(r"<[^>]+>", "", seg)
        plain = re.sub(r"\[IMG:[^\]]*\]", "", plain)
        plain = re.sub(r"\s+", " ", plain).strip()
        if plain:
            caption_texts.append(plain)

    return main_markup, caption_texts


def render_caption_segments(caption_texts: list) -> list:
    """把caption文本片段渲染为小字灰色斜体Paragraph列表"""
    flows = []
    for txt in caption_texts:
        flows.append(Paragraph(escape_plain(txt), STYLES["CaptionText"]))
    return flows


# ================================================================
# 富文本标记 → ReportLab markup 转换
# ================================================================
def rich_to_reportlab(text: str) -> str:
    """
    将BERT层的富文本标记转换为ReportLab Paragraph支持的markup。
    支持: <b>, <u>, <color:#xxx>, <size:x>
    去除: [IMG:...] 标记（图片单独处理）
    """
    if not text:
        return ""

    # 去掉图片标记（在chunk content里图片由 image_suggestion 单独渲染，原[IMG]标记不直接显示）
    text = re.sub(r"\[IMG:[^\]]*\]", "", text)

    # <color:#xxxxxx> ... 后续文字着色，简化处理：转换为 <font color="#xxx">
    # 先找到所有 color 标记位置，转换为成对标签较复杂，这里采用简化策略：
    # 把 <color:#xxx> 转为 <font color="#xxx"> 并在下一个标记或行尾前插入</font>
    def color_repl(m):
        return f'<font color="{m.group(1)}">'
    text = re.sub(r"<color:(#[0-9a-fA-F]+)>", color_repl, text)

    # <size:x> 转为 <font size="x">
    def size_repl(m):
        sz = float(m.group(1))
        sz = max(8, min(sz, 16))  # 限制范围，避免打印异常
        return f'<font size="{sz}">'
    text = re.sub(r"<size:[\d.]+>", "", text)  # 字号差异在打印稿中不强调，直接移除避免标签未闭合问题

    # 移除遗留的color font标签的闭合问题：简单地在每行末尾补</font>（如果有未闭合的<font）
    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        open_fonts = len(re.findall(r"<font[^>]*>", line))
        close_fonts = len(re.findall(r"</font>", line))
        if open_fonts > close_fonts:
            line += "</font>" * (open_fonts - close_fonts)
        fixed_lines.append(line)
    text = "<br/>".join(fixed_lines)

    # <b> </b> 和 <u> </u> ReportLab原生支持，保留
    # 清理多余空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(<br/>\s*){3,}", "<br/><br/>", text)

    return text.strip()


def escape_plain(text: str) -> str:
    """对纯文本做XML转义（用于不含markup的字段，如instruction）"""
    if not text:
        return ""
    return (text.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;"))


def markdown_bold_to_rl(text: str) -> str:
    """把 **bold** 转为 <b>bold</b>（Claude改编后的instruction可能含markdown）"""
    if not text:
        return ""
    text = escape_plain(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text


# ================================================================
# 通用小部件
# ================================================================
def boxed(flowables, bg=C_AMBER_LT, border=C_AMBER, pad=10, border_width=0.75):
    """把一组flowable包进一个带背景色和边框的Table，模拟卡片样式"""
    t = Table([[flowables]], colWidths=[PAGE_W - 2 * MARGIN - 2 * pad])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), border_width, border),
        ("LEFTPADDING", (0, 0), (-1, -1), pad),
        ("RIGHTPADDING", (0, 0), (-1, -1), pad),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def answer_line(width_mm=None):
    """生成一条答题横线（用Table模拟下划线，避免HRFlowable破坏流式布局）"""
    width = (width_mm * mm) if width_mm else (PAGE_W - 2 * MARGIN)
    return Table([[""]], colWidths=[width], rowHeights=[14],
                  style=TableStyle([
                      ("LINEBELOW", (0, 0), (-1, -1), 0.75, C_LINE),
                  ]))


def checkbox_row(label_text):
    """渲染 □ 选项 形式（用于multiple choice / yes-no）"""
    return Paragraph(f"☐ &nbsp;{escape_plain(label_text)}", STYLES["BodyTight"])


# ================================================================
# Key Words Box (pre_training)
# ================================================================
def render_key_words_box(key_words):
    if not key_words:
        return None

    rows = []
    header = Paragraph("\u270E &nbsp; KEY WORDS — read these before you start", STYLES["BoxHeader"])
    rows.append(header)
    rows.append(Spacer(1, 4))

    for kw in key_words:
        word = kw.get("word", "")
        definition = kw.get("definition", "")
        line = Paragraph(
            f'<b><font color="#1D9E75">{escape_plain(word)}</font></b> — {escape_plain(definition)}',
            STYLES["KeyWordDef"]
        )
        rows.append(line)
        rows.append(Spacer(1, 2))

    return boxed(rows, bg=C_GREEN_LT, border=C_GREEN, pad=10)


# ================================================================
# Task Steps Box (task_decomposition)
# ================================================================
def render_task_steps(task_steps):
    if not task_steps:
        return None

    items = []
    for step in task_steps:
        # step格式预期: "Step 1: Read the text."
        m = re.match(r"Step\s*(\d+)\s*:\s*(.+)", step, re.I)
        if m:
            num, content = m.groups()
        else:
            num, content = "•", step

        row = Table(
            [[Paragraph(f"<b>{num}</b>", STYLES["StepText"]),
              Paragraph(escape_plain(content), STYLES["StepText"])]],
            colWidths=[16, PAGE_W - 2 * MARGIN - 16 - 20],
            style=TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        items.append(row)

    return boxed(items, bg=C_BLUE_LT, border=C_BLUE, pad=10)


# ================================================================
# Image Suggestion Placeholder (multimedia)
# ================================================================
def render_image_placeholder(image_suggestion):
    if not image_suggestion:
        return None

    content = Table(
        [[Paragraph("\u2587\u2587\u2587 IMAGE \u2587\u2587\u2587",
                     ParagraphStyle("ImgIcon", fontName="DejaVuSans", fontSize=9,
                                     textColor=C_TEXT_LIGHT, alignment=TA_CENTER))],
         [Paragraph(f"Suggested image: {escape_plain(image_suggestion)}", STYLES["ImgPlaceholder"])]],
        colWidths=[PAGE_W - 2 * MARGIN - 20],
        style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    return Table([[content]], colWidths=[PAGE_W - 2 * MARGIN],
                  style=TableStyle([
                      ("BOX", (0, 0), (-1, -1), 0.75, C_LINE),
                      ("LINEBELOW", (0, 0), (-1, -1), 0, C_LINE),
                      ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FAFAF8")),
                      ("TOPPADDING", (0, 0), (-1, -1), 10),
                      ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                  ]))


# ================================================================
# 按 chunk_type 渲染答题区
# ================================================================
def render_answer_area(chunk: dict) -> list:
    """根据chunk_type生成对应的答题区flowables"""
    ctype = chunk.get("chunk_type", "other")
    raw_content = chunk.get("content", "")

    # 分离主体文本与小字说明文本(图片署名/OCR残留等)
    content, caption_texts = split_by_font_size(raw_content)

    flows = []

    if ctype == "fill_in_the_blank":
        # 按行拆分，每个空白处给一条独立答题线
        lines = [l for l in content.split("\n") if l.strip()]
        for line in lines:
            txt = rich_to_reportlab(line)
            if txt:
                flows.append(Paragraph(txt, STYLES["Body"]))
            flows.append(answer_line())
            flows.append(Spacer(1, 4))

    elif ctype == "multiple_choice":
        lines = [l for l in content.split("\n") if l.strip()]
        for line in lines:
            txt = rich_to_reportlab(line)
            # 题干本身正常显示；选项行加 checkbox
            if re.match(r"^\s*[A-Da-d][\.\)]\s", line) or re.match(r"^\s*[-•]\s", line):
                flows.append(checkbox_row(re.sub(r"<[^>]+>", "", line).strip()))
            else:
                flows.append(Paragraph(txt, STYLES["Body"]))
        flows.append(Spacer(1, 4))

    elif ctype == "discussion":
        txt = rich_to_reportlab(content)
        flows.append(Paragraph(txt, STYLES["Body"]))
        flows.append(Spacer(1, 4))
        for _ in range(3):
            flows.append(answer_line())
            flows.append(Spacer(1, 6))

    elif ctype == "writing":
        txt = rich_to_reportlab(content)
        flows.append(Paragraph(txt, STYLES["Body"]))
        flows.append(Spacer(1, 4))
        for _ in range(5):
            flows.append(answer_line())
            flows.append(Spacer(1, 6))

    elif ctype == "matching":
        txt = rich_to_reportlab(content)
        flows.append(Paragraph(txt, STYLES["Body"]))
        flows.append(Spacer(1, 4))

    elif ctype == "table_completion":
        txt = rich_to_reportlab(content)
        flows.append(Paragraph(txt, STYLES["Body"]))
        flows.append(Spacer(1, 4))

    elif ctype == "source_text":
        txt = rich_to_reportlab(content)
        flows.append(Paragraph(txt, STYLES["Body"]))

    elif ctype in ("vocabulary", "translation"):
        lines = [l for l in content.split("\n") if l.strip()]
        for line in lines:
            txt = rich_to_reportlab(line)
            flows.append(Paragraph(txt, STYLES["BodyTight"]))
            flows.append(answer_line(width_mm=60))
            flows.append(Spacer(1, 3))

    else:  # other / explanation / 默认
        txt = rich_to_reportlab(content)
        if txt:
            flows.append(Paragraph(txt, STYLES["Body"]))

    # 小字说明文本(图片署名/OCR残留等) 单独用小字灰色斜体渲染在末尾
    if caption_texts:
        flows.extend(render_caption_segments(caption_texts))

    return flows


# ================================================================
# 渲染单个chunk
# ================================================================
def render_chunk(chunk: dict, task_number: int, total_tasks: int) -> list:
    """
    返回一个flowable列表，代表这个chunk的完整渲染内容。
    使用KeepTogether包裹"头部"部分(标题+指令+key words box+steps)，
    防止头部和答题区第一行被孤立分页。
    答题区本身允许跨页流动（尤其是source_text长文本）。
    """
    flows_head = []
    flows_body = []

    ctype = chunk.get("chunk_type", "other")
    task_title = chunk.get("task_title", "")
    type_desc  = chunk.get("type_description", "")
    instruction_adapted = chunk.get("instruction_adapted") or chunk.get("instruction", "")
    strategies_applied = chunk.get("strategies_applied", [])

    # --- 进度指示 (仅非source_text/explanation的"任务型"chunk显示) ---
    is_task = ctype not in ("explanation",)
    if is_task and total_tasks > 0:
        flows_head.append(Paragraph(f"TASK {task_number} / {total_tasks}", STYLES["TaskProgress"]))

    # --- 标题 ---
    if task_title:
        label = escape_plain(task_title)
    else:
        label_map = {
            "source_text": "Reading Text",
            "explanation": "",
            "fill_in_the_blank": "Fill in the Blanks",
            "multiple_choice": "Multiple Choice",
            "matching": "Matching",
            "writing": "Writing Task",
            "discussion": "Discussion",
            "vocabulary": "Vocabulary",
            "translation": "Translation",
            "table_completion": "Complete the Table",
            "other": type_desc or "Task",
        }
        label = label_map.get(ctype, "Task")

    if label:
        flows_head.append(Paragraph(label, STYLES["TaskHeader"]))

    # --- 指令 (signaling/decomposition改编后) ---
    if instruction_adapted:
        instr_markup = markdown_bold_to_rl(instruction_adapted)
        # ▶ 符号来自signaling策略,已在Claude输出中体现，直接渲染
        flows_head.append(Paragraph(instr_markup, STYLES["Instruction"]))

    # --- Key Words Box (pre_training) ---
    kw_box = render_key_words_box(chunk.get("key_words_box"))
    if kw_box:
        flows_head.append(Spacer(1, 4))
        flows_head.append(kw_box)
        flows_head.append(Spacer(1, 6))

    # --- Task Steps Box (task_decomposition) ---
    steps_box = render_task_steps(chunk.get("task_steps"))
    if steps_box:
        flows_head.append(Spacer(1, 4))
        flows_head.append(steps_box)
        flows_head.append(Spacer(1, 6))

    # --- 答题区 ---
    flows_body.extend(render_answer_area(chunk))

    # --- Image Suggestion (multimedia) ---
    img_box = render_image_placeholder(chunk.get("image_suggestion"))
    if img_box:
        flows_body.append(Spacer(1, 6))
        flows_body.append(img_box)

    # --- 分隔线 ---
    flows_body.append(Spacer(1, 10))
    flows_body.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER,
                                   spaceBefore=2, spaceAfter=10))

    # head部分用KeepTogether，防止标题与指令分页；
    # body首个元素也纳入KeepTogether，防止"标题+指令"后紧跟空白答题区被孤立
    if flows_body:
        head_and_first = flows_head + [flows_body[0]]
        rest_body = flows_body[1:]
        return [KeepTogether(head_and_first)] + rest_body
    else:
        return [KeepTogether(flows_head)]


# ================================================================
# 页眉页脚
# ================================================================
def make_header_footer(worksheet_title: str):
    def _draw(canvas, doc):
        canvas.saveState()

        # Header
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(C_TEXT_LIGHT)
        canvas.drawString(MARGIN, PAGE_H - 12 * mm, worksheet_title)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 12 * mm, "ADHD-Friendly Adapted Version")
        canvas.setStrokeColor(C_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - 14 * mm, PAGE_W - MARGIN, PAGE_H - 14 * mm)

        # Footer
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(C_TEXT_LIGHT)
        canvas.drawCentredString(PAGE_W / 2, 10 * mm, f"Page {doc.page}")
        canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)

        canvas.restoreState()
    return _draw


# ================================================================
# 主渲染函数
# ================================================================
def build_pdf(chunks: list[dict], output_path: str, worksheet_title: str = "Adapted Worksheet"):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + 6 * mm, bottomMargin=MARGIN + 4 * mm,
        title=worksheet_title,
    )

    story = []

    # --- 文档标题区 (第一页顶部) ---
    story.append(Paragraph(worksheet_title, STYLES["WSTitle"]))
    story.append(Paragraph("Adapted for focus and clarity", STYLES["WSSubtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=C_AMBER,
                              spaceBefore=2, spaceAfter=10))

    # --- 统计任务总数 (排除explanation类型) ---
    task_chunks = [c for c in chunks if c.get("chunk_type") != "explanation"]
    total_tasks = len(task_chunks)

    task_counter = 0
    for chunk in chunks:
        if chunk.get("chunk_type") != "explanation":
            task_counter += 1
            tnum = task_counter
        else:
            tnum = 0
        story.extend(render_chunk(chunk, tnum, total_tasks))

    doc.build(story, onFirstPage=make_header_footer(worksheet_title),
              onLaterPages=make_header_footer(worksheet_title))

    print(f"\n✅ PDF已生成: {output_path}")
    print(f"   共渲染 {len(chunks)} 个chunk ({total_tasks} 个任务)")


# ================================================================
# 加载adapted chunks
# ================================================================
def load_adapted_chunks(input_dir: str) -> list[dict]:
    path = Path(input_dir)
    files = sorted(path.glob("adapted_chunk_*.json"))
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到 adapted_chunk_*.json 文件")
    chunks = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            chunks.append(json.load(fp))
    print(f"读取到 {len(chunks)} 个adapted chunk")
    return chunks


# ================================================================
# 主入口
# ================================================================
def run(input_dir: str, output_path: str, worksheet_title: str = "Adapted Worksheet"):
    chunks = load_adapted_chunks(input_dir)
    build_pdf(chunks, output_path, worksheet_title)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第三层：ReportLab生成ADHD友好PDF")
    parser.add_argument("--input_dir", required=True, help="adapted_chunk_*.json所在目录")
    parser.add_argument("--output",    required=True, help="输出PDF路径")
    parser.add_argument("--title",     default="Adapted Worksheet", help="Worksheet标题")
    args = parser.parse_args()

    run(args.input_dir, args.output, args.title)
