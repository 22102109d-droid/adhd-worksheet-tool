"""
level1_pipeline.py
================
第一层：PDF -> chunk dict列表

提供两个主入口函数，供 main.py 调用：

  process_pdf_single_column(pdf_path, image_dir, model_dir, api_key, model_endpoint)
      正常模式: fitz提取(带格式标记) -> BERT切chunk -> 豆包打标签 -> Python检测signaling/multimedia

  process_pdf_two_column(pdf_path, image_dir, api_key, model_endpoint)
      双栏模式: fitz提取全文(带格式标记) -> 豆包一次性分chunk+打标签
                -> Python检测signaling/multimedia
      (BERT按行分类在双栏布局下顺序会错乱，所以跳过BERT，整页交给豆包)

两个函数都返回 chunk dict 列表 (不写文件，由调用方决定是否落盘)。
图片会被提取到 image_dir。
"""

import json
import os
import re
import sys
from pathlib import Path

import fitz
# torch / transformers / volcenginesdkarkruntime 是重依赖，
# 只在真正调用对应函数时import，避免mock模式下也要安装这些包


# ================================================================
# Signaling / Multimedia 检测 (与原版一致，规则不变)
# ================================================================
def detect_signaling(content):
    """
    检测 chunk 是否具备 Signaling 策略。
    从 content 里的格式标记读取加粗/字号/颜色/下划线信息。
    返回: (bool, dict) — 是否具备, 详细检测结果
    """
    plain = re.sub(r"<[^>]+>", "", content)
    plain = re.sub(r"\[IMG:[^\]]*\]", "", plain)
    total_chars = len(plain.replace(" ", "").replace("\n", ""))
    if total_chars == 0:
        return False, {"reason": "无文字内容"}

    # --- 1. 加粗 ---
    bold_matches = re.findall(r"<b>(.*?)</b>", content, re.S)
    bold_chars = sum(len(m.replace(" ", "")) for m in bold_matches)
    bold_ratio = bold_chars / total_chars if total_chars > 0 else 0

    title_pattern = re.compile(
        r"^(task|part|section|activity|exercise|question)\s*\d*\s*[:.]?\s*$", re.I)
    emphasis_bolds = [m.strip() for m in bold_matches
                      if m.strip() and not title_pattern.match(m.strip())]
    has_emphasis_bold = len(emphasis_bolds) > 0

    bold_ok = has_emphasis_bold and 0.005 < bold_ratio < 0.4

    # --- 2. 颜色 ---
    color_matches = re.findall(r"<color:(#[0-9a-f]+)>", content, re.I)
    unique_colors = set(color_matches)
    color_ok = len(unique_colors) >= 1
    color_excessive = len(unique_colors) > 5

    # --- 3. 字号变化 ---
    size_matches = re.findall(r"<size:([\d.]+)>", content)
    unique_sizes = sorted(set(float(s) for s in size_matches)) if size_matches else []
    size_range = (max(unique_sizes) - min(unique_sizes)) if len(unique_sizes) > 1 else 0
    size_ok = size_range >= 3

    # --- 4. 下划线 ---
    underline_matches = re.findall(r"<u>(.*?)</u>", content, re.S)
    underline_chars = sum(len(m.replace(" ", "")) for m in underline_matches)
    underline_ratio = underline_chars / total_chars if total_chars > 0 else 0
    underline_ok = len(underline_matches) > 0 and 0.003 < underline_ratio < 0.3

    # --- 综合判断 ---
    signals_found = []
    if bold_ok:
        signals_found.append("bold_emphasis")
    if color_ok and not color_excessive:
        signals_found.append("color_highlight")
    if color_excessive:
        signals_found.append("color_excessive_warning")
    if size_ok:
        signals_found.append("font_hierarchy")
    if underline_ok:
        signals_found.append("underline")

    effective_signals = [s for s in signals_found if s != "color_excessive_warning"]
    has_signaling = len(effective_signals) > 0

    details = {
        "bold": {
            "found": len(bold_matches),
            "emphasis_count": len(emphasis_bolds),
            "ratio": round(bold_ratio, 3),
            "effective": bold_ok,
        },
        "color": {
            "found": len(color_matches),
            "unique_colors": sorted(unique_colors),
            "excessive": color_excessive,
            "effective": color_ok and not color_excessive,
        },
        "font_size": {
            "unique_sizes": unique_sizes,
            "range_pt": round(size_range, 1),
            "effective": size_ok,
        },
        "underline": {
            "found": len(underline_matches),
            "ratio": round(underline_ratio, 3),
            "effective": underline_ok,
        },
        "signals_found": signals_found,
    }

    return has_signaling, details


def detect_multimedia(content):
    """
    检测 chunk 是否具备 Multimedia 策略。
    从 content 里的 [IMG] 标记读取图片信息。
    返回: (bool, dict) — 是否具备, 详细检测结果
    """
    img_full = re.findall(
        r"\[IMG:\s*([^,\]]+?)(?:,\s*([\d.]+)x([\d.]+)pt)?(?:,\s*占页面(\d+)%)?\]",
        content)

    all_images = []
    meaningful_images = []

    for match in img_full:
        filename = match[0].strip()
        area_pct = int(match[3]) if match[3] else -1

        all_images.append(filename)

        if area_pct == -1 or area_pct >= 5:
            meaningful_images.append(filename)

    plain = re.sub(r"<[^>]+>", "", content)
    plain = re.sub(r"\[IMG:[^\]]*\]", "", plain)
    has_text = len(plain.strip()) > 20

    has_multimedia = len(meaningful_images) > 0 and has_text

    details = {
        "total_images": len(all_images),
        "meaningful_images": len(meaningful_images),
        "image_files": meaningful_images,
        "has_text_alongside": has_text,
    }

    return has_multimedia, details


def annotate_signaling_multimedia(chunks: list[dict]) -> list[dict]:
    """给每个chunk打上has_signaling / has_multimedia标签 (原第三步逻辑)"""
    for chunk in chunks:
        content = chunk.get("content", "")

        sig_result, sig_details = detect_signaling(content)
        chunk["has_signaling"] = sig_result
        chunk["signaling_details"] = sig_details

        mm_result, mm_details = detect_multimedia(content)
        image_relevance = chunk.get("image_relevance", "none")
        image_type = chunk.get("image_type", "none")
        if image_type == "logo" or image_relevance == "decorative":
            mm_result = False
        chunk["has_multimedia"] = mm_result
        chunk["multimedia_details"] = mm_details

    return chunks


# ================================================================
# fitz提取 (带格式标记的富文本 + 图片)
# ================================================================
def extract_rich_text_and_images(pdf_path: str, image_dir: str):
    """
    提取PDF的富文本(带<b><u><size:><color:>标记和[IMG:...]位置标记)和图片。
    图片保存到image_dir。

    返回:
      pdf_text: 全文富文本字符串(按页拼接，含 "--- PAGE N ---"分隔)
      image_list: 图片文件名列表
    """
    os.makedirs(image_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    pdf_text = ""
    image_list = []

    for page_num, page in enumerate(doc):
        page_text = ""
        d = page.get_text("dict")
        prev_size = None
        page_area = page.rect.width * page.rect.height

        for block in d["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    text = span["text"]
                    if not text.strip():
                        line_text += text
                        continue

                    cur_size = round(span["size"], 1)
                    if cur_size != prev_size:
                        line_text += f"<size:{cur_size}>"
                        prev_size = cur_size

                    color = span.get("color", 0)
                    if color != 0:
                        hex_color = f"#{color:06x}" if isinstance(color, int) else str(color)
                        line_text += f"<color:{hex_color}>"

                    is_bold = bool(span["flags"] & 16) or bool(
                        re.search(r"bold|black|heavy", span["font"], re.I))
                    is_underline = bool(span["flags"] & 4)

                    if is_bold and is_underline:
                        line_text += f"<b><u>{text}</u></b>"
                    elif is_bold:
                        line_text += f"<b>{text}</b>"
                    elif is_underline:
                        line_text += f"<u>{text}</u>"
                    else:
                        line_text += text

                page_text += line_text + "\n"
            page_text += "\n"

        # --- 图片提取 + 位置标记 ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                img_filename = f"page{page_num+1}_img{img_index+1}.png"
                img_path = os.path.join(image_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])
                image_list.append(img_filename)

                img_rects = page.get_image_rects(xref)
                if img_rects:
                    r = img_rects[0]
                    w, h = round(r.width, 1), round(r.height, 1)
                    area_ratio = round((r.width * r.height) / page_area * 100) if page_area else 0
                    page_text += f"\n[IMG: {img_filename}, {w}x{h}pt, 占页面{area_ratio}%]\n"
                else:
                    page_text += f"\n[IMG: {img_filename}]\n"
            except Exception as e:
                print(f"提取图片失败: {e}")

        pdf_text += f"--- PAGE {page_num+1} ---\n{page_text}\n"

    doc.close()
    return pdf_text, image_list


# ================================================================
# 豆包: 单个chunk打标签 (原第二步b逻辑，抽出复用)
# ================================================================
def label_chunk_with_doubao(client, model_endpoint, chunk_content, image_list):
    """对单个chunk内容调用豆包，返回标注dict"""
    label_prompt = f"""你是一个EFL教学材料分析助手。以下是一个worksheet的文字片段，请分析并输出JSON格式的标注结果。

只输出JSON对象，不要其他文字：
{{
  "chunk_type": "chunk类型",
  "type_description": "如果是other则描述题型，否则为空字符串",
  "task_title": "Task标题或编号，没有则为空字符串",
  "instruction": "指令句，没有则为空字符串",
  "has_image": true或false,
  "image_files": ["对应图片文件名，没有则为空列表"],
  "image_type": "none/logo/illustration/photo/diagram",
  "image_relevance": "none/decorative/task_related",
  "has_pre_training": true或false,
  "task_decomposition_needed": null,
  "linked_to": null
}}

chunk_type选项：source_text、explanation、fill_in_the_blank、multiple_choice、matching、writing、discussion、vocabulary、translation、table_completion、other

图片文件名列表：{image_list}

以下是chunk内容：
{chunk_content}"""

    response = client.chat.completions.create(
        model=model_endpoint,
        messages=[{"role": "user", "content": label_prompt}]
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}


# ================================================================
# 模式A: 单栏 — fitz + BERT + 豆包
# ================================================================
def process_pdf_single_column(
    pdf_path: str,
    image_dir: str,
    model_dir: str,
    api_key: str,
    model_endpoint: str,
) -> list[dict]:
    """
    单栏模式主入口。

    参数:
      pdf_path:       上传的PDF路径
      image_dir:      图片输出目录
      model_dir:      BERT模型目录 (包含 tokenizer/ 和 best_model.pt)
      api_key:        豆包API key
      model_endpoint: 豆包model endpoint

    返回: chunk dict 列表 (已包含 has_signaling/has_multimedia)
    """
    from volcenginesdkarkruntime import Ark
    import torch

    client = Ark(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")

    # --- Step 1: fitz提取 ---
    print("提取文字和图片(带格式标记)...")
    _pdf_text, image_list = extract_rich_text_and_images(pdf_path, image_dir)
    print(f"文字提取完成，找到{len(image_list)}张图片")

    # --- Step 2: 模型切chunk ---
    print("\n模型切chunk...")
    bert_dir = Path(model_dir)
    sys.path.insert(0, str(Path(__file__).parent))
    from common import (
        WorksheetSegmenter, NUM_CLASSES, build_context_text,
        extract_extra_features, extract_lines_with_rich,
    )
    from transformers import AutoTokenizer

    device = torch.device("cpu")
    checkpoint = torch.load(bert_dir / "best_model.pt", map_location=device, weights_only=False)
    bert_name = checkpoint.get("bert_name", "microsoft/deberta-v3-base")
    if bert_name.startswith("/"):
        bert_name = "microsoft/deberta-v3-base"
    bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_dir / "tokenizer"))
    bert_model = WorksheetSegmenter(
        bert_name=bert_name,
        num_classes=checkpoint.get("num_classes", NUM_CLASSES)
    ).to(device)
    bert_model.load_state_dict(checkpoint["model_state_dict"])
    bert_model.eval()
    context_window = checkpoint["context_window"]
    max_length = checkpoint["max_length"]
    print(f"模型加载完成 ({bert_name}, F1={checkpoint['best_f1']:.4f})")

    all_lines = extract_lines_with_rich(pdf_path)
    text_lines = [l for l in all_lines if not l.get("is_image", False) and l["text"]]
    print(f"共{len(text_lines)}行（不含图片行）")

    results = []
    with torch.no_grad():
        for i, line in enumerate(text_lines):
            text = build_context_text(text_lines, i, window=context_window)
            extra = extract_extra_features(line)
            encoding = bert_tokenizer(
                text, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            features = torch.tensor([extra], dtype=torch.float).to(device)
            logits = bert_model(input_ids, attention_mask, features)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()
            label = ["I", "B", "J"][pred_class]
            results.append({"rich": line["rich"], "label": label, "page": line["page"]})

    img_lines = [l for l in all_lines if l.get("is_image", False)]
    img_by_page = {}
    for img in img_lines:
        img_by_page.setdefault(img["page"], []).append(img["rich"])

    sections = []
    current = None
    processed_pages = set()
    for r in results:
        if r["label"] == "J":
            continue
        if r["label"] == "B":
            if current:
                sections.append(current)
            current = {"rich_lines": [r["rich"]], "pages": {r["page"]}}
        else:
            if current is None:
                current = {"rich_lines": [r["rich"]], "pages": {r["page"]}}
            else:
                current["rich_lines"].append(r["rich"])
                current["pages"].add(r["page"])

        page = r["page"]
        if page not in processed_pages and page in img_by_page and current:
            for img_marker in img_by_page[page]:
                current["rich_lines"].append(img_marker)
            processed_pages.add(page)

    if current:
        sections.append(current)

    print(f"BERT切出{len(sections)}个chunk")

    # --- Step 3: 豆包打标签 ---
    print("\n豆包标注chunk类型...")
    chunks = []
    for i, section in enumerate(sections):
        chunk_content = "\n".join(section["rich_lines"])
        print(f"  标注chunk {i+1}/{len(sections)}...")

        labels = label_chunk_with_doubao(client, model_endpoint, chunk_content, image_list)

        chunk = {
            "chunk_id": i + 1,
            "order": i + 1,
            "chunk_type": labels.get("chunk_type", "other"),
            "type_description": labels.get("type_description", ""),
            "task_title": labels.get("task_title", ""),
            "instruction": labels.get("instruction", ""),
            "content": chunk_content,
            "has_image": labels.get("has_image", False),
            "image_files": labels.get("image_files", []),
            "image_type": labels.get("image_type", "none"),
            "image_relevance": labels.get("image_relevance", "none"),
            "has_pre_training": labels.get("has_pre_training", False),
            "task_decomposition_needed": None,
            "linked_to": labels.get("linked_to", None),
        }
        chunks.append(chunk)

    print(f"豆包标注完成，共{len(chunks)}个chunk")

    # --- Step 4: signaling/multimedia检测 ---
    print("\n检测 Signaling 和 Multimedia...")
    chunks = annotate_signaling_multimedia(chunks)

    return chunks


# ================================================================
# 模式B: 双栏 — 整页交给豆包做"分chunk+打标签"
# ================================================================
def label_full_page_with_doubao(client, model_endpoint, pdf_text, image_list):
    """
    双栏模式: 把整份PDF的富文本一次性交给豆包，
    让豆包同时完成分chunk(按教学任务边界切分)和打标签。
    返回: chunk dict列表 (不含chunk_id/order/has_signaling等，后续补充)
    """
    prompt = f"""你是一个EFL教学材料分析助手。以下是一份worksheet的完整文字内容(双栏排版PDF提取，
行顺序可能因为双栏布局而交错，请你根据语义和教学逻辑重新组织，
把内容切分为若干个独立的教学任务/chunk(例如: 一段阅读文本算一个chunk，
一组练习题算一个chunk，一个讨论任务算一个chunk)。

对每个chunk输出以下字段，只输出JSON数组，不要其他文字：
[
  {{
    "chunk_type": "chunk类型",
    "type_description": "如果是other则描述题型，否则为空字符串",
    "task_title": "Task标题或编号，没有则为空字符串",
    "instruction": "指令句，没有则为空字符串",
    "content": "该chunk对应的原文内容(尽量保留原始的<b><u><size:><color:>标记和[IMG:...]标记，不要改写文字，只做切分和重新排序)",
    "has_image": true或false,
    "image_files": ["对应图片文件名，没有则为空列表"],
    "image_type": "none/logo/illustration/photo/diagram",
    "image_relevance": "none/decorative/task_related",
    "has_pre_training": true或false,
    "linked_to": null
  }}
]

chunk_type选项：source_text、explanation、fill_in_the_blank、multiple_choice、matching、writing、discussion、vocabulary、translation、table_completion、other

图片文件名列表：{image_list}

重要: content字段必须是原文的精确摘录(可重新排序/分段，但不可改写文字内容)。

以下是PDF全文内容：
{pdf_text}"""

    response = client.chat.completions.create(
        model=model_endpoint,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception as e:
        print(f"豆包双栏模式输出解析失败: {e}")
        print(raw[:500])
        return []


def process_pdf_two_column(
    pdf_path: str,
    image_dir: str,
    api_key: str,
    model_endpoint: str,
) -> list[dict]:
    """
    双栏模式主入口。跳过BERT，整页文本交给豆包分chunk+打标签。

    参数:
      pdf_path:       上传的PDF路径
      image_dir:      图片输出目录
      api_key:        豆包API key
      model_endpoint: 豆包model endpoint

    返回: chunk dict 列表 (已包含 has_signaling/has_multimedia)
    """
    from volcenginesdkarkruntime import Ark

    client = Ark(api_key=api_key, base_url="https://ark.cn-beijing.volces.com/api/v3")

    # --- Step 1: fitz提取全文 ---
    print("提取文字和图片(带格式标记, 双栏模式)...")
    pdf_text, image_list = extract_rich_text_and_images(pdf_path, image_dir)
    print(f"文字提取完成，找到{len(image_list)}张图片")
    print("注意: 双栏布局下行顺序可能交错，已整页交给豆包重新组织")

    # --- Step 2: 豆包一次性分chunk+打标签 ---
    print("\n豆包分chunk + 标注 (双栏模式)...")
    raw_chunks = label_full_page_with_doubao(client, model_endpoint, pdf_text, image_list)

    chunks = []
    for i, labels in enumerate(raw_chunks):
        chunk = {
            "chunk_id": i + 1,
            "order": i + 1,
            "chunk_type": labels.get("chunk_type", "other"),
            "type_description": labels.get("type_description", ""),
            "task_title": labels.get("task_title", ""),
            "instruction": labels.get("instruction", ""),
            "content": labels.get("content", ""),
            "has_image": labels.get("has_image", False),
            "image_files": labels.get("image_files", []),
            "image_type": labels.get("image_type", "none"),
            "image_relevance": labels.get("image_relevance", "none"),
            "has_pre_training": labels.get("has_pre_training", False),
            "task_decomposition_needed": None,
            "linked_to": labels.get("linked_to", None),
        }
        chunks.append(chunk)

    print(f"豆包分chunk完成，共{len(chunks)}个chunk")

    # --- Step 3: signaling/multimedia检测 ---
    print("\n检测 Signaling 和 Multimedia...")
    chunks = annotate_signaling_multimedia(chunks)

    return chunks


# ================================================================
# 保存chunks到磁盘 (可选，main.py也可以直接用返回的dict)
# ================================================================
def save_chunks(chunks: list[dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", 0)
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_id:03d}.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)


# ================================================================
# CLI测试入口 (可选)
# ================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Level1: PDF -> chunks")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--two_column", action="store_true")
    parser.add_argument("--model_dir", default=None, help="BERT模型目录 (单栏模式必需)")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model_endpoint", required=True)
    args = parser.parse_args()

    if args.two_column:
        chunks = process_pdf_two_column(
            pdf_path=args.pdf,
            image_dir=args.image_dir,
            api_key=args.api_key,
            model_endpoint=args.model_endpoint,
        )
    else:
        if not args.model_dir:
            raise ValueError("单栏模式需要 --model_dir")
        chunks = process_pdf_single_column(
            pdf_path=args.pdf,
            image_dir=args.image_dir,
            model_dir=args.model_dir,
            api_key=args.api_key,
            model_endpoint=args.model_endpoint,
        )

    save_chunks(chunks, args.output_dir)
    print(f"\n✅ 完成！共{len(chunks)}个chunk，输出到：{args.output_dir}")
