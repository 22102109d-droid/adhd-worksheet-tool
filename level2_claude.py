"""
level2_claude.py
================
第二层：读取第一层输出的chunk JSON → 调Claude API → 返回改编后的HTML

流程：
  1. 读取某个worksheet目录下所有 chunk_*.json
  2. 生成策略缺失报告（供前端展示，让老师勾选）
  3. 接收老师勾选的策略列表
  4. 一次性调用Claude API，完成：
       a. Task Decomposition判断（所有chunk）
       b. 对勾选策略执行改编
       c. 直接生成完整的ADHD友好HTML
  5. 输出HTML字符串，供第三层转PDF
"""

import json
import os
import re
import sys
import argparse
import anthropic
from pathlib import Path

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
CLAUDE_MODEL   = "claude-sonnet-4-20250514"

SELECTABLE_STRATEGIES = ["pre_training", "signaling", "task_decomposition", "multimedia"]

# ================================================================
# Step 1: 读取chunk JSON文件
# ================================================================
def load_chunks(input_dir: str) -> list[dict]:
    path = Path(input_dir)
    files = sorted(path.glob("chunk_*.json"))
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 中未找到chunk_*.json文件")
    chunks = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            chunks.append(json.load(fp))
    print(f"读取到 {len(chunks)} 个chunk")
    return chunks


# ================================================================
# Step 2: 生成策略缺失报告（供前端展示）
# ================================================================
MULTIMEDIA_SUITABLE_TYPES = {"source_text", "explanation", "discussion", "vocabulary"}


def generate_strategy_report(chunks: list[dict]) -> dict:
    pre_missing = [c["chunk_id"] for c in chunks
                   if not c.get("has_pre_training") and c.get("instruction")]
    sig_missing  = [c["chunk_id"] for c in chunks
                    if not c.get("has_signaling") and c.get("instruction")]
    mm_missing   = [c["chunk_id"] for c in chunks
                    if not c.get("has_multimedia")
                    and c.get("chunk_type") in MULTIMEDIA_SUITABLE_TYPES]

    return {
        "pre_training": {
            "missing_count": len(pre_missing),
            "affected_chunks": pre_missing,
            "title": "Pre-training (Key Words Box)",
            "why": "Students with ADHD often get stuck on unfamiliar vocabulary mid-task, which breaks their attention and makes it hard to return to the main task. Pre-teaching 3-5 key words before the task reduces this cognitive interruption (Mayer's Pre-training Principle).",
            "example": {
                "before": "Task starts directly with: \"Read the interview and answer the questions.\"",
                "after": "A green 'KEY WORDS' box appears first:\n  buttered — grew up completely in one place\n  dairy — a small shop selling milk\nThen the task starts as normal."
            }
        },
        "signaling": {
            "missing_count": len(sig_missing),
            "affected_chunks": sig_missing,
            "title": "Signaling (Visual Cues)",
            "why": "Plain instruction text gives no visual anchor for where to focus. Adding bold action verbs and a clear marker (▶) helps students immediately locate 'what to do', reducing the time spent re-reading instructions (Mayer's Signaling Principle).",
            "example": {
                "before": "Read the statements. Circle YES or NO.",
                "after": "▶ Circle YES or NO for each statement."
            }
        },
        "task_decomposition": {
            "missing_count": None,
            "affected_chunks": "To be determined by AI analysis",
            "title": "Task Decomposition (Step-by-Step)",
            "why": "Instructions with multiple sequential actions (e.g. 'read, then answer, then discuss') place a heavy load on working memory. Students with ADHD may lose track of later steps. Breaking these into a numbered Step 1 / Step 2 / Step 3 list lets students complete one action at a time (Sweller's Cognitive Load Theory).",
            "example": {
                "before": "Read the interview again, then answer the questions, and discuss your answers with a partner.",
                "after": "Step 1: Read the interview again.\nStep 2: Answer the questions below.\nStep 3: Discuss your answers with a partner."
            }
        },
        "multimedia": {
            "missing_count": len(mm_missing),
            "affected_chunks": mm_missing,
            "title": "Multimedia (Supporting Images)",
            "why": "Pairing text with a relevant image reduces the load on verbal working memory by letting students process information through both visual and verbal channels (Mayer's Multimedia Principle). Only reading, discussion, and vocabulary sections are suggested here — exercise items like fill-in-the-blank are excluded, since decorative images there add clutter rather than helping comprehension.",
            "example": {
                "before": "Text-only discussion prompt about locations in London (Kings Cross, Tufnell Park).",
                "after": "Same text, with an added placeholder: 'Suggested image: a simple map of London highlighting Kings Cross and Tufnell Park.'"
            }
        }
    }


# ================================================================
# Step 3: System Prompt
# ================================================================
SYSTEM_PROMPT = """\
You are an expert EFL teacher and instructional designer specialising in ADHD-friendly materials.

You will receive worksheet chunks and must:
1. Apply the selected ADHD strategies to each chunk
2. Output a SINGLE complete HTML document (not JSON) — a beautiful, print-ready ADHD-friendly worksheet

=== STRATEGY DEFINITIONS ===

PRE_TRAINING:
Add a green "KEY WORDS" box before tasks with no vocabulary preparation.
Extract 3-5 key vocabulary items from the chunk content.
Write definitions in simple English (A2-B1 level).
Skip if has_pre_training is already true.

SIGNALING:
Add "▶" before the main instruction.
Bold key action verbs.
Add a short "💡 TIP:" if the task is cognitively complex.
Skip if has_signaling is already true.

TASK_DECOMPOSITION:
Only apply if the instruction contains multiple sequential steps.
Break into numbered steps: Step 1, Step 2, Step 3.
Do NOT change exercise content.

MULTIMEDIA:
Add a shaded image placeholder box with a specific description.
Only for source_text, discussion, vocabulary, explanation chunks.
Skip if has_multimedia is already true.

=== HTML OUTPUT REQUIREMENTS ===

Output a complete, self-contained HTML file with embedded CSS. Requirements:

DESIGN:
- Clean, friendly design suitable for printing on A4
- Amber/warm colour scheme: #D98E2B for headers, #FBF3E6 for backgrounds
- Green (#1D9E75) for KEY WORDS boxes
- Blue (#3B6FA8) for task headers
- Font: system fonts (Arial, sans-serif) — no external fonts needed
- Page breaks between tasks using CSS

STRUCTURE per task:
- Task header with progress indicator (TASK 1 / 2, TASK 2 / 2 etc.)
- KEY WORDS box (green background) if pre_training applied
- Instruction with ▶ and bold verbs if signaling applied
- Numbered steps if task_decomposition applied
- The original content (text, table, numbered lines) — PRESERVE ALL ORIGINAL CONTENT EXACTLY
- Image placeholder box (dashed border) if multimedia applied

TABLES: If the original content contains a table structure, render it as an actual HTML <table> with proper borders and styling.

NUMBERED LINES: If content has numbered lines (1, 2, 3...), render them as a proper numbered list with answer lines, not as raw numbers.

CONTENT RULES:
- NEVER change any exercise content — questions, answer options, source texts must be word-for-word identical
- Only modify: instructions, add KEY WORDS box, add image placeholder
- Include a footer: "Adapted for students with attention differences using ADHD-friendly instructional design principles"

Output ONLY the HTML — no explanation, no markdown fences, no preamble.\
"""


# ================================================================
# Step 4: 调用Claude API，直接生成HTML
# ================================================================
def call_claude_api(chunks: list[dict], selected_strategies: list[str], worksheet_title: str) -> str:
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    # 精简chunk内容
    slim_chunks = []
    for c in chunks:
        slim_chunks.append({
            "chunk_id":            c.get("chunk_id"),
            "chunk_type":          c.get("chunk_type"),
            "task_title":          c.get("task_title", ""),
            "instruction":         c.get("instruction", ""),
            "content":             c.get("content", ""),
            "has_pre_training":    c.get("has_pre_training", False),
            "has_signaling":       c.get("has_signaling", False),
            "has_multimedia":      c.get("has_multimedia", False),
            "task_decomposition_needed": c.get("task_decomposition_needed", False),
        })

    strategies_str = ", ".join(selected_strategies) if selected_strategies else "none"
    total_tasks = len(set(c.get("task_title", "") for c in chunks if c.get("task_title")))

    user_prompt = f"""Worksheet title: {worksheet_title}
Total tasks: {total_tasks}
Teacher-selected strategies to apply: [{strategies_str}]

Here are the chunks:
{json.dumps(slim_chunks, ensure_ascii=False, indent=2)}

Generate the complete ADHD-friendly HTML worksheet now."""

    print(f"\n调用Claude API生成HTML...")
    print(f"  模型: {CLAUDE_MODEL}")
    print(f"  Chunks数量: {len(chunks)}")
    print(f"  选择的策略: {selected_strategies}")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    usage = response.usage
    print(f"\n  Token用量:")
    print(f"    Input:  {usage.input_tokens:,} tokens")
    print(f"    Output: {usage.output_tokens:,} tokens")
    estimated_cost = (usage.input_tokens / 1_000_000 * 3.0) + (usage.output_tokens / 1_000_000 * 15.0)
    print(f"    预估费用: ${estimated_cost:.4f}")

    html = response.content[0].text.strip()

    # 清理可能的markdown fence
    if html.startswith("```"):
        html = re.sub(r"^```(?:html)?\s*", "", html)
        html = re.sub(r"\s*```$", "", html)
        html = html.strip()

    print(f"\n✅ Claude返回HTML ({len(html)} 字符)")
    return html


# ================================================================
# 主函数
# ================================================================
def run(input_dir: str, selected_strategies: list[str], output_dir: str = None, worksheet_title: str = ""):
    invalid = [s for s in selected_strategies if s not in SELECTABLE_STRATEGIES]
    if invalid:
        raise ValueError(f"不支持的策略: {invalid}")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir.rstrip("/")), "adapted")
    os.makedirs(output_dir, exist_ok=True)

    chunks = load_chunks(input_dir)
    report = generate_strategy_report(chunks)

    if not worksheet_title:
        for c in chunks:
            if c.get("task_title"):
                worksheet_title = c["task_title"]
                break
        worksheet_title = worksheet_title or Path(input_dir).parent.name

    html = call_claude_api(chunks, selected_strategies, worksheet_title)

    # 保存HTML
    html_path = os.path.join(output_dir, "adapted.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML已保存: {html_path}")

    # 同时保存一个假的merged列表（main.py兼容性）
    merged = [{"chunk_id": c.get("chunk_id"), "html": True} for c in chunks]

    return merged, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True)
    parser.add_argument("--strategies", nargs="*", default=[])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--title",      default="")
    args = parser.parse_args()
    run(input_dir=args.input_dir, selected_strategies=args.strategies,
        output_dir=args.output_dir, worksheet_title=args.title)
