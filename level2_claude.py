"""
level2_claude.py
================
第二层：读取第一层输出的chunk JSON → 调Claude API → 返回改编后的结构化JSON

流程：
  1. 读取某个worksheet目录下所有 chunk_*.json
  2. 生成策略缺失报告（供前端展示，让老师勾选）
  3. 接收老师勾选的策略列表
  4. 一次性调用Claude API，完成：
       a. Task Decomposition判断（所有chunk）
       b. 对勾选策略执行改编
  5. 输出改编后的JSON列表，供第三层ReportLab渲染PDF

用法（本地测试）：
  python level2_claude.py --input_dir ~/Desktop/output_A1/worksheet_name/worksheet_name \
                          --strategies pre_training signaling task_decomposition
"""

import json
import os
import re
import sys
import argparse
import anthropic
from pathlib import Path

# ===== 填你的Claude API Key =====
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
CLAUDE_MODEL   = "claude-sonnet-4-20250514"
# ================================

# ------------------------------------------------------------------ #
#  六个策略说明
#  - lesson_structure : 默认应用，不需要勾选（ReportLab层处理）
#  - segmenting       : 默认应用，BERT已完成分chunk
#  - pre_training     : 检测缺失 → 老师勾选 → Claude添加
#  - signaling        : 检测缺失 → 老师勾选 → Claude添加
#  - task_decomposition: Claude判断+改编
#  - multimedia       : 检测缺失 → 老师勾选 → Claude添加建议占位符
# ------------------------------------------------------------------ #

SELECTABLE_STRATEGIES = ["pre_training", "signaling", "task_decomposition", "multimedia"]


# ================================================================
# Step 1: 读取chunk JSON文件
# ================================================================
def load_chunks(input_dir: str) -> list[dict]:
    """读取目录下所有chunk_*.json，按chunk_id排序返回"""
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
# 适合配图的chunk_type——纯题型(填空/选择/匹配/表格/写作/翻译)配图通常是装饰性的，
# 不计入multimedia缺失统计，避免老师勾选后生成大量无意义图片
MULTIMEDIA_SUITABLE_TYPES = {"source_text", "explanation", "discussion", "vocabulary"}


def generate_strategy_report(chunks: list[dict]) -> dict:
    """
    分析所有chunk，统计每个可选策略影响到哪些chunk。
    每个策略附带 why（原理说明）和 example（before/after示例），
    供前端展示给老师，辅助其决定是否勾选。最终决定权在老师。

    返回格式：
    {
      "pre_training": {
          "missing_count": 3,
          "affected_chunks": [1,3,5],
          "title": "...",
          "why": "...",
          "example": {"before": "...", "after": "..."}
      },
      ...
    }
    """
    report = {}

    # ---------------- Pre-training ----------------
    pt_missing = [c["chunk_id"] for c in chunks if not c.get("has_pre_training", True)]
    report["pre_training"] = {
        "missing_count": len(pt_missing),
        "affected_chunks": pt_missing,
        "title": "Pre-training (Key Words Box)",
        "why": (
            "Students with ADHD often get stuck on unfamiliar vocabulary mid-task, "
            "which breaks their attention and makes it hard to return to the main task. "
            "Pre-teaching 3-5 key words before the task reduces this cognitive interruption "
            "(Mayer's Pre-training Principle)."
        ),
        "example": {
            "before": "Task starts directly with: \"Read the interview and answer the questions.\"",
            "after": (
                "A green 'KEY WORDS' box appears first:\n"
                "  buttered — grew up completely in one place\n"
                "  dairy — a small shop selling milk\n"
                "Then the task starts as normal."
            )
        }
    }

    # ---------------- Signaling ----------------
    sig_missing = [c["chunk_id"] for c in chunks if not c.get("has_signaling", True)]
    report["signaling"] = {
        "missing_count": len(sig_missing),
        "affected_chunks": sig_missing,
        "title": "Signaling (Visual Cues)",
        "why": (
            "Plain instruction text gives no visual anchor for where to focus. "
            "Adding bold action verbs and a clear marker (▶) helps students "
            "immediately locate 'what to do', reducing the time spent re-reading "
            "instructions (Mayer's Signaling Principle)."
        ),
        "example": {
            "before": "Read the statements. Circle YES or NO.",
            "after": "▶ Circle YES or NO for each statement."
        }
    }

    # ---------------- Task Decomposition ----------------
    report["task_decomposition"] = {
        "missing_count": None,
        "affected_chunks": "To be determined by AI analysis",
        "title": "Task Decomposition (Step-by-Step)",
        "why": (
            "Instructions with multiple sequential actions (e.g. 'read, then answer, "
            "then discuss') place a heavy load on working memory. Students with ADHD "
            "may lose track of later steps. Breaking these into a numbered Step 1 / "
            "Step 2 / Step 3 list lets students complete one action at a time "
            "(Sweller's Cognitive Load Theory)."
        ),
        "example": {
            "before": "Read the interview again, then answer the questions, and discuss your answers with a partner.",
            "after": "Step 1: Read the interview again.\nStep 2: Answer the questions below.\nStep 3: Discuss your answers with a partner."
        }
    }

    # ---------------- Multimedia ----------------
    mm_missing = [
        c["chunk_id"] for c in chunks
        if not c.get("has_multimedia", True)
        and c.get("chunk_type") in MULTIMEDIA_SUITABLE_TYPES
    ]
    report["multimedia"] = {
        "missing_count": len(mm_missing),
        "affected_chunks": mm_missing,
        "title": "Multimedia (Supporting Images)",
        "why": (
            "Pairing text with a relevant image reduces the load on verbal working "
            "memory by letting students process information through both visual and "
            "verbal channels (Mayer's Multimedia Principle). Only reading, discussion, "
            "and vocabulary sections are suggested here — exercise items like "
            "fill-in-the-blank are excluded, since decorative images there add clutter "
            "rather than helping comprehension."
        ),
        "example": {
            "before": "Text-only discussion prompt about locations in London (Kings Cross, Tufnell Park).",
            "after": "Same text, with an added placeholder: 'Suggested image: a simple map of London highlighting Kings Cross and Tufnell Park.'"
        }
    }

    return report


# ================================================================
# Step 3: 构建Claude API Prompt
# ================================================================

SYSTEM_PROMPT = """You are an expert EFL material adaptation specialist for students with ADHD.

You will receive a list of worksheet chunks (in JSON format) and a list of strategies to apply.
Your job is to:
  1. For EVERY chunk: judge whether task_decomposition is needed (true/false)
  2. For chunks where a strategy is needed AND the teacher selected it: apply the adaptation

=== TASK DECOMPOSITION JUDGMENT RULES ===
Set task_decomposition_needed = true ONLY if the instruction contains 2+ SEQUENTIAL action verbs
that require the student to perform distinct cognitive operations IN ORDER.

Examples that NEED decomposition:
- "Read the text, answer the questions, then discuss with a partner." (3 verbs, sequential)
- "Watch the video, note down 3 key points, and write a summary." (3 verbs, sequential)

Examples that do NOT need decomposition:
- "Circle the correct answer." (1 verb)
- "Fill in the blanks." (1 verb)
- "Match the words to the definitions." (1 verb, even if multiple items)
- "Answer questions 1-5." (1 verb, repeated on independent items)

=== STRATEGY DEFINITIONS ===

PRE_TRAINING:
Add a "key_words_box" before tasks that have no vocabulary preparation.
Extract 3-5 key vocabulary items directly from the chunk content.
Write definitions in simple English (A2-B1 level). Do NOT invent words not in the content.
Format: [{"word": "...", "definition": "simple explanation"}]
Skip if has_pre_training is already true.

SIGNALING:
Enhance the instruction text only (do NOT change exercise content).
Add "▶" before the main instruction.
Bold key action verbs using **verb** markdown.
Add a short "💡 TIP:" sentence at the end if the task is cognitively complex.
Skip if has_signaling is already true.

TASK_DECOMPOSITION:
Only apply if task_decomposition_needed = true.
Break the instruction into numbered steps: "Step 1: ... Step 2: ... Step 3: ..."
Each step should be one clear action. Do NOT change the exercise content itself.

MULTIMEDIA:
Add an image_suggestion string describing a relevant supporting image — but ONLY if the
chunk content is concrete enough to describe a meaningful image (e.g. a place, object,
diagram, or scene mentioned in the text).
The suggestion should be specific: e.g. "A simple diagram showing the water cycle with labels"
NOT vague: "An image related to the topic"
If the content is too abstract (e.g. a generic opinion question like "What do you think?"
with no concrete subject), set image_suggestion = null even if the strategy was selected.
Skip if has_multimedia is already true.

=== ABSOLUTE CONTENT RULES ===
- NEVER change any exercise content (questions, answer options, source texts, vocabulary items)
- ONLY modify: instruction phrasing, add key_words_box, add image_suggestion
- All original text in "content" field must remain word-for-word identical
- Output valid JSON array ONLY — no preamble, no markdown fences, no explanation

=== OUTPUT FORMAT ===
Return a JSON array. Each element:
{
  "chunk_id": <int>,
  "chunk_type": "<original>",
  "task_title": "<original>",
  "instruction_original": "<original instruction>",
  "task_decomposition_needed": <true|false>,
  "strategies_applied": ["list of strategy names actually applied to this chunk"],
  "key_words_box": null | [{"word": "...", "definition": "..."}],
  "instruction_adapted": "<adapted instruction, or same as original if no signaling/decomposition>",
  "task_steps": null | ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "image_suggestion": null | "<description string>",
  "content": "<original content, UNCHANGED>"
}"""


def build_user_prompt(chunks: list[dict], selected_strategies: list[str], worksheet_title: str) -> str:
    """构建发给Claude的user prompt"""

    # 精简chunk内容，只保留Claude需要的字段，减少token消耗
    slim_chunks = []
    for c in chunks:
        slim_chunks.append({
            "chunk_id":        c.get("chunk_id"),
            "chunk_type":      c.get("chunk_type"),
            "task_title":      c.get("task_title", ""),
            "instruction":     c.get("instruction", ""),
            "content":         c.get("content", ""),
            "has_pre_training":c.get("has_pre_training", False),
            "has_signaling":   c.get("has_signaling", False),
            "has_multimedia":  c.get("has_multimedia", False),
        })

    strategies_str = ", ".join(selected_strategies) if selected_strategies else "none"

    prompt = f"""Worksheet title: {worksheet_title}
Total chunks: {len(slim_chunks)}
Teacher-selected strategies to apply: [{strategies_str}]

Instructions:
- Assess task_decomposition_needed for ALL chunks regardless of selected strategies
- Apply selected strategies only where the strategy is missing (has_* = false)
- If no strategies are selected, still return all chunks with task_decomposition_needed assessed

Here are the chunks:
{json.dumps(slim_chunks, ensure_ascii=False, indent=2)}

Return the adapted JSON array now."""

    return prompt


# ================================================================
# Step 4: 调用Claude API
# ================================================================
def call_claude_api(chunks: list[dict], selected_strategies: list[str], worksheet_title: str) -> list[dict]:
    """调用Claude API，返回改编后的chunk列表"""

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    user_prompt = build_user_prompt(chunks, selected_strategies, worksheet_title)

    print(f"\n调用Claude API...")
    print(f"  模型: {CLAUDE_MODEL}")
    print(f"  Chunks数量: {len(chunks)}")
    print(f"  选择的策略: {selected_strategies}")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )

    # 打印token用量
    usage = response.usage
    print(f"\n  Token用量:")
    print(f"    Input:  {usage.input_tokens:,} tokens")
    print(f"    Output: {usage.output_tokens:,} tokens")
    print(f"    Total:  {usage.input_tokens + usage.output_tokens:,} tokens")
    estimated_cost = (usage.input_tokens / 1_000_000 * 3.0) + (usage.output_tokens / 1_000_000 * 15.0)
    print(f"    预估费用: ${estimated_cost:.4f}")

    raw = response.content[0].text.strip()

    # 清理可能的markdown fence
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

    try:
        adapted_chunks = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"\n❌ Claude输出解析失败: {e}")
        print("原始输出前500字符:")
        print(raw[:500])
        raise

    print(f"\n✅ Claude返回 {len(adapted_chunks)} 个改编chunk")
    return adapted_chunks


# ================================================================
# Step 5: 合并原始字段 + 保存输出
# ================================================================
def merge_and_save(original_chunks: list[dict], adapted_chunks: list[dict], output_dir: str) -> list[dict]:
    """
    将Claude的改编结果合并回原始chunk的完整字段，
    保存为 adapted_chunk_*.json，同时返回合并后列表。
    """
    # 建立原始chunk的id→dict索引
    original_map = {c["chunk_id"]: c for c in original_chunks}

    os.makedirs(output_dir, exist_ok=True)
    merged_list = []

    for adapted in adapted_chunks:
        cid = adapted.get("chunk_id")
        original = original_map.get(cid, {})

        # 合并：原始字段为基础，Claude的改编字段覆盖
        merged = {**original}
        merged["task_decomposition_needed"] = adapted.get("task_decomposition_needed", False)
        merged["strategies_applied"]        = adapted.get("strategies_applied", [])
        merged["key_words_box"]             = adapted.get("key_words_box", None)
        merged["instruction_adapted"]       = adapted.get("instruction_adapted", original.get("instruction", ""))
        merged["task_steps"]                = adapted.get("task_steps", None)
        merged["image_suggestion"]          = adapted.get("image_suggestion", None)
        # content字段永远保留原始值
        merged["content"]                   = original.get("content", adapted.get("content", ""))

        merged_list.append(merged)

        # 保存
        out_path = os.path.join(output_dir, f"adapted_chunk_{cid:03d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n改编结果已保存到: {output_dir}")
    return merged_list


# ================================================================
# Step 6: 打印摘要报告
# ================================================================
def print_summary(adapted_chunks: list[dict]):
    print(f"\n{'='*65}")
    print(f"  改编摘要报告")
    print(f"{'='*65}")
    print(f"  {'ID':<6} {'类型':<22} {'TD需要':<8} {'应用策略'}")
    print(f"  {'─'*60}")
    for c in adapted_chunks:
        td   = "✓" if c.get("task_decomposition_needed") else "✗"
        strats = ", ".join(c.get("strategies_applied", [])) or "—"
        ctype = c.get("chunk_type", "unknown")[:20]
        print(f"  {c['chunk_id']:<6} {ctype:<22} {td:<8} {strats}")
    print(f"{'='*65}\n")


# ================================================================
# 主函数
# ================================================================
def run(input_dir: str, selected_strategies: list[str], output_dir: str = None, worksheet_title: str = ""):
    """
    主入口，供外部（FastAPI）调用或本地测试。

    参数：
      input_dir          : 存放chunk_*.json的目录
      selected_strategies: 老师勾选的策略列表，例如 ["pre_training", "signaling"]
      output_dir         : 改编结果输出目录（默认在input_dir同级的adapted/目录）
      worksheet_title    : worksheet标题（用于prompt）
    """
    # 验证策略名
    invalid = [s for s in selected_strategies if s not in SELECTABLE_STRATEGIES]
    if invalid:
        raise ValueError(f"不支持的策略: {invalid}。可选策略: {SELECTABLE_STRATEGIES}")

    # 输出目录默认值
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_dir.rstrip("/")), "adapted")

    # 1. 读取chunks
    chunks = load_chunks(input_dir)

    # 2. 生成报告（供前端展示用，这里仅打印）
    report = generate_strategy_report(chunks)
    print("\n策略缺失报告:")
    for strategy, info in report.items():
        count = info["missing_count"]
        count_str = str(count) if count is not None else "TBD"
        print(f"  {strategy:<22}: 缺失 {count_str} 个chunk — {info['title']}")

    # 3. 获取worksheet标题（从第一个有task_title的chunk，或用传入参数）
    if not worksheet_title:
        for c in chunks:
            if c.get("task_title"):
                worksheet_title = c["task_title"]
                break
        worksheet_title = worksheet_title or Path(input_dir).parent.name

    # 4. 调用Claude API
    adapted_chunks = call_claude_api(chunks, selected_strategies, worksheet_title)

    # 5. 合并保存
    merged = merge_and_save(chunks, adapted_chunks, output_dir)

    # 6. 打印摘要
    print_summary(merged)

    return merged, report


# ================================================================
# 命令行入口
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="第二层：Claude API worksheet改编")
    parser.add_argument("--input_dir",   required=True,  help="chunk JSON所在目录")
    parser.add_argument("--strategies",  nargs="*", default=[], help="老师勾选的策略")
    parser.add_argument("--output_dir",  default=None,   help="输出目录（默认自动设置）")
    parser.add_argument("--title",       default="",     help="Worksheet标题")
    args = parser.parse_args()

    run(
        input_dir=args.input_dir,
        selected_strategies=args.strategies,
        output_dir=args.output_dir,
        worksheet_title=args.title,
    )
