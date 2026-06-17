"""
level2_claude.py
================
第二层：读取第一层输出的chunk JSON → 调Claude API → 返回改编后的HTML
"""

import json
import os
import re
import sys
import argparse
import anthropic
from pathlib import Path

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
CLAUDE_MODEL = "claude-sonnet-4-6"

SELECTABLE_STRATEGIES = ["lesson_structure", "pre_training", "task_decomposition", "segmenting", "multimedia", "signaling"]


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


MULTIMEDIA_SUITABLE_TYPES = {"source_text", "explanation", "discussion", "vocabulary"}


def generate_strategy_report(chunks: list[dict]) -> dict:
    pre_missing = [c["chunk_id"] for c in chunks
                   if not c.get("has_pre_training") and c.get("instruction")]
    sig_missing  = [c["chunk_id"] for c in chunks
                    if not c.get("has_signaling") and c.get("instruction")]
    mm_missing   = [c["chunk_id"] for c in chunks
                    if not c.get("has_multimedia")
                    and c.get("chunk_type") in MULTIMEDIA_SUITABLE_TYPES]
    seg_missing  = [c["chunk_id"] for c in chunks
                    if c.get("chunk_type") == "source_text"]

    return {
        "lesson_structure": {
            "missing_count": 1,
            "affected_chunks": [],
            "title": "Lesson Structure (Overview Box)",
            "why": "Students with ADHD benefit from knowing the full picture before starting. An overview box at the top of the worksheet shows all tasks at a glance, reducing anxiety and helping students plan their effort (Sweller's Cognitive Load Theory).",
            "example": {
                "before": "Worksheet starts directly with Task 1.",
                "after": "A 📋 WHAT YOU WILL DO TODAY box appears first, listing each task with a one-line description."
            }
        },
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
        "task_decomposition": {
            "missing_count": None,
            "affected_chunks": "To be determined by AI analysis",
            "title": "Task Decomposition (Step-by-Step)",
            "why": "Instructions with multiple sequential actions place a heavy load on working memory. Breaking these into numbered steps lets students complete one action at a time (Sweller's Cognitive Load Theory).",
            "example": {
                "before": "Read the interview again, then answer the questions, and discuss your answers with a partner.",
                "after": "Step 1: Read the interview again.\nStep 2: Answer the questions below.\nStep 3: Discuss your answers with a partner."
            }
        },
        "segmenting": {
            "missing_count": len(seg_missing),
            "affected_chunks": seg_missing,
            "title": "Segmenting (Visual Breaks in Reading Text)",
            "why": "Long unbroken reading passages overwhelm working memory. Breaking the source text into smaller visual chunks with clear spacing lets students process one section at a time (Mayer's Segmenting Principle).",
            "example": {
                "before": "A long reading passage presented as one continuous block of text.",
                "after": "The same passage divided into clearly spaced sections of 2-3 sentences each, making it easier to track progress."
            }
        },
        "multimedia": {
            "missing_count": len(mm_missing),
            "affected_chunks": mm_missing,
            "title": "Multimedia (Supporting Images)",
            "why": "Pairing text with a relevant image reduces the load on verbal working memory (Mayer's Multimedia Principle).",
            "example": {
                "before": "Text-only discussion prompt.",
                "after": "Same text, with an added image placeholder."
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
        }
    }


SYSTEM_PROMPT = """\
You are an expert EFL teacher and instructional designer specialising in ADHD-friendly materials.

You will receive worksheet chunks and must output a SINGLE complete, self-contained HTML document.

=== CRITICAL CONTENT RULES ===
1. NEVER alter any exercise content — every word of questions, answer options, source texts, numbered lines must be IDENTICAL to the input.
2. Only add: KEY WORDS box, ▶ signaling marker, numbered steps, image placeholder.
3. If the chunk content contains a TABLE (rows of data with column headers), you MUST render it as a proper HTML <table> with <thead>, <tbody>, <tr>, <th>, <td> tags. Never flatten a table into plain text.
4. If the chunk content has NUMBERED LINES (lines starting with 1, 2, 3...), render each as a div with the number and a long underline for writing, like: <div class="answer-line"><span class="line-num">1</span><span class="line-blank">____________________________</span></div>
5. Preserve ALL footnotes exactly as they appear.
6. Use UTF-8 characters correctly — ▶ ✎ 💡 must render properly.

=== STRATEGIES TO APPLY ===

LESSON_STRUCTURE (if selected):
- Add an overview box at the very TOP of the HTML body, before all task sections.
- List every task with its number, title, and a one-line description of what students will do.
- Format: <div class="overview-box"><h2>📋 What you will do today</h2><ol>...</ol></div>
- Base the descriptions on task_title and chunk_type from the chunks.

PRE_TRAINING (if selected and has_pre_training is false):
- Add a green KEY WORDS box BEFORE the task instruction.
- Extract 3-5 key vocabulary items from the content.
- Write definitions in simple English (A2-B1 level).
- Format: <div class="key-words-box"><h3>✎ KEY WORDS — read these before you start</h3><dl>...</dl></div>

SIGNALING (if selected and has_signaling is false):
- Prepend ▶ to the main instruction.
- Wrap key action verbs in <strong>.
- Add 💡 TIP: if the task is complex.

TASK_DECOMPOSITION (if selected):
- Only if instruction has multiple sequential steps (read AND answer AND discuss etc.).
- Break into: Step 1: ... Step 2: ... Step 3: ...
- Use <ol class="steps"><li>...</li></ol>

SEGMENTING (if selected, ONLY for chunk_type === "source_text"):
- Split the source text into visual sections of 2-3 sentences each.
- Wrap each section in <div class="segment">...</div>
- Do NOT change any words. Only add visual spacing between sections.
- Do NOT apply to exercise items, tables, question lists, or any other chunk type.

MULTIMEDIA (if selected and has_multimedia is false, chunk_type in source_text/discussion/vocabulary/explanation):
- Add a dashed image placeholder box AFTER the content.
- Include a specific description of what image would help.

=== HTML STRUCTURE ===

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{worksheet_title}</title>
<style>
  /* All CSS embedded here */
  body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #fff; color: #333; }
  .task-section { margin-bottom: 40px; page-break-after: always; }
  .task-header { background: #3B6FA8; color: white; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; }
  .task-header h2 { margin: 0; font-size: 1.1em; text-transform: uppercase; letter-spacing: 1px; }
  .task-header .task-type { font-size: 1.3em; font-weight: bold; margin-top: 4px; }
  .key-words-box { background: #E8F5F0; border: 2px solid #1D9E75; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px; }
  .key-words-box h3 { color: #1D9E75; margin: 0 0 10px 0; font-size: 1em; }
  .key-words-box dl { margin: 0; }
  .key-words-box dt { font-weight: bold; color: #1D9E75; float: left; margin-right: 8px; }
  .key-words-box dd { margin-left: 0; margin-bottom: 6px; }
  .instruction { background: #FBF3E6; border-left: 4px solid #D98E2B; padding: 10px 15px; margin-bottom: 15px; border-radius: 0 6px 6px 0; }
  .steps { background: #FBF3E6; border-left: 4px solid #D98E2B; padding: 10px 15px 10px 30px; margin-bottom: 15px; border-radius: 0 6px 6px 0; }
  .steps li { margin-bottom: 6px; }
  .content-area { margin: 15px 0; line-height: 1.8; }
  .answer-line { display: flex; align-items: baseline; margin-bottom: 10px; gap: 8px; }
  .line-num { font-weight: bold; min-width: 25px; color: #555; }
  .line-blank { flex: 1; border-bottom: 1px solid #333; min-width: 200px; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; }
  th { background: #3B6FA8; color: white; padding: 10px 12px; text-align: left; border: 1px solid #2a5090; }
  td { padding: 10px 12px; border: 1px solid #ccc; vertical-align: top; }
  tr:nth-child(even) td { background: #f5f8ff; }
  .overview-box { background: #EEF4FB; border: 2px solid #3B6FA8; border-radius: 8px; padding: 16px 20px; margin-bottom: 30px; }
  .overview-box h2 { color: #3B6FA8; margin: 0 0 10px 0; font-size: 1em; }
  .overview-box ol { margin: 0; padding-left: 1.2rem; }
  .overview-box li { margin-bottom: 5px; font-size: 14px; color: #333; }
  .segment { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px dashed #ddd; }
  .segment:last-child { border-bottom: none; }
  .footer { margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd; font-size: 0.8em; color: #888; text-align: center; }
  @media print { .task-section { page-break-after: always; } }
</style>
</head>
<body>
  <h1 style="color:#D98E2B; border-bottom: 2px solid #D98E2B; padding-bottom:10px;">{worksheet_title}</h1>
  <p style="color:#888; font-size:0.9em;">ADHD-Friendly Adapted Version</p>
  
  <!-- One .task-section per chunk -->
  
  <div class="footer">Adapted for students with attention differences using ADHD-friendly instructional design principles</div>
</body>
</html>

Output ONLY the complete HTML — no explanation, no markdown fences, no preamble.\
"""


def call_claude_api(chunks: list[dict], selected_strategies: list[str], worksheet_title: str) -> str:
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    slim_chunks = []
    for c in chunks:
        slim_chunks.append({
            "chunk_id":                  c.get("chunk_id"),
            "chunk_type":                c.get("chunk_type"),
            "task_title":                c.get("task_title", ""),
            "instruction":               c.get("instruction", ""),
            "content":                   c.get("content", ""),
            "has_pre_training":          c.get("has_pre_training", False),
            "has_signaling":             c.get("has_signaling", False),
            "has_multimedia":            c.get("has_multimedia", False),
            "task_decomposition_needed": c.get("task_decomposition_needed", False),
        })

    strategies_str = ", ".join(selected_strategies) if selected_strategies else "none"
    total_tasks = len(set(c.get("task_title", "") for c in chunks if c.get("task_title")))

    user_prompt = f"""Worksheet title: {worksheet_title}
Total tasks: {total_tasks}
Selected strategies: [{strategies_str}]

Chunks:
{json.dumps(slim_chunks, ensure_ascii=False, indent=2)}

Generate the complete ADHD-friendly HTML worksheet now. Remember:
- Tables in content → proper HTML <table>
- Numbered lines → .answer-line divs with underlines
- Never change exercise content
- Output raw HTML only"""

    print(f"\n调用Claude API生成HTML...")
    print(f"  模型: {CLAUDE_MODEL}, Chunks: {len(chunks)}, 策略: {selected_strategies}")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    usage = response.usage
    estimated_cost = (usage.input_tokens / 1_000_000 * 3.0) + (usage.output_tokens / 1_000_000 * 15.0)
    print(f"  Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out | 费用: ${estimated_cost:.4f}")

    html = response.content[0].text.strip()
    if html.startswith("```"):
        html = re.sub(r"^```(?:html)?\s*", "", html)
        html = re.sub(r"\s*```$", "", html)
        html = html.strip()

    print(f"✅ Claude返回HTML ({len(html)} 字符)")
    return html


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

    html_path = os.path.join(output_dir, "adapted.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML已保存: {html_path}")

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
