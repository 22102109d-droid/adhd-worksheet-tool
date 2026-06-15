"""
main.py
================
FastAPI后端，整合三层pipeline，供Webflow前端调用。

Endpoints:
  POST /api/upload    - 上传PDF，运行level1，返回strategy_report
  POST /api/adapt     - 提交老师勾选的策略，运行level2+level3，返回PDF
  GET  /api/download/{worksheet_id} - 下载生成的PDF

注意:
  - 邮箱使用次数限制暂未实现(后续加)
  - level1部分目前用mock占位(MOCK_LEVEL1=True)，待level1接口化后切换
  - Railway文件系统是临时的，storage/下的文件需要定期清理
"""

import os
import uuid
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download, snapshot_download

# 启动时自动下载模型
BERT_MODEL_DIR = os.environ.get("BERT_MODEL_DIR", "model")
if not os.path.exists(f"{BERT_MODEL_DIR}/best_model.pt"):
    print("Downloading model from Hugging Face...")
    os.makedirs(f"{BERT_MODEL_DIR}/tokenizer", exist_ok=True)
    hf_hub_download(
        repo_id="mellyii/adhd-bert-model",
        filename="best_model.pt",
        local_dir=BERT_MODEL_DIR,
        token=os.environ.get("HF_TOKEN"),
    )
    snapshot_download(
        repo_id="mellyii/adhd-bert-model",
        allow_patterns="tokenizer/*",
        local_dir=BERT_MODEL_DIR,
        token=os.environ.get("HF_TOKEN"),
    )
    print("Model downloaded!")
    
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import fitz  # PyMuPDF，用于检查页数

import level1_pipeline
import level2_claude
import level3_pdf

# ================================================================
# 配置
# ================================================================
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

MAX_PAGES = 5
FILE_RETENTION_HOURS = 24  # 临时文件保留时长

# Level1所需配置 (BERT模型目录 + 豆包API)
DOUBAO_API_KEY = os.environ.get("DOUBAO_API_KEY", "")
DOUBAO_MODEL_ENDPOINT = os.environ.get("DOUBAO_MODEL_ENDPOINT", "")

# 是否使用mock level1 (本地无BERT模型/无豆包key时可设为True测试整体流程)
MOCK_LEVEL1 = os.environ.get("MOCK_LEVEL1", "false").lower() == "true"


# ================================================================
# FastAPI app
# ================================================================
app = FastAPI(title="ADHD Worksheet Adapter API")

# CORS: 允许Webflow域名调用
# 部署后把 "*" 替换为实际的Webflow域名，例如 "https://adhd-lesson-tool.webflow.io"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: 上线前改成具体域名
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# 工具函数
# ================================================================
def get_worksheet_dir(worksheet_id: str) -> Path:
    d = STORAGE_DIR / worksheet_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_old_worksheets():
    """删除超过FILE_RETENTION_HOURS的临时worksheet目录"""
    cutoff = datetime.now() - timedelta(hours=FILE_RETENTION_HOURS)
    for d in STORAGE_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                created = datetime.fromisoformat(meta["created_at"])
                if created < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                continue


def check_pdf_pages(pdf_path: Path) -> int:
    """返回PDF页数"""
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n


# ================================================================
# Level 1 接口
# ================================================================
def run_level1_mock(pdf_path: Path, has_two_columns: bool, output_dir: Path) -> list[dict]:
    """
    Mock版本：返回一份固定的chunk数据，不调用BERT/豆包/网络。
    用于本地无模型文件/无API key时测试整体流程 (MOCK_LEVEL1=true)。
    """
    mock_chunks = [
        {
            "chunk_id": 1, "order": 1, "chunk_type": "discussion",
            "type_description": "", "task_title": "Warmer",
            "instruction": "Read the text and discuss with a partner.",
            "content": "This is mock content for testing the pipeline without running BERT/Doubao.",
            "has_image": False, "image_files": [], "image_type": "none", "image_relevance": "none",
            "has_pre_training": False, "task_decomposition_needed": None, "linked_to": None,
            "has_signaling": False,
            "signaling_details": {"signals_found": []},
            "has_multimedia": False,
            "multimedia_details": {"meaningful_images": 0},
        },
    ]
    level1_pipeline.save_chunks(mock_chunks, str(output_dir))
    return mock_chunks


def run_level1(pdf_path: Path, has_two_columns: bool, output_dir: Path, image_dir: Path) -> list[dict]:
    """
    Level1主入口。

    - has_two_columns=False -> fitz提取 + BERT分chunk + 豆包打标签
    - has_two_columns=True  -> fitz提取全文 + 豆包一次性分chunk+打标签
    返回chunk dict列表，同时把每个chunk写入 output_dir/chunk_*.json
    """
    if MOCK_LEVEL1:
        return run_level1_mock(pdf_path, has_two_columns, output_dir)

    if not DOUBAO_API_KEY or not DOUBAO_MODEL_ENDPOINT:
        raise RuntimeError("DOUBAO_API_KEY / DOUBAO_MODEL_ENDPOINT 未配置")

    if has_two_columns:
        chunks = level1_pipeline.process_pdf_two_column(
            pdf_path=str(pdf_path),
            image_dir=str(image_dir),
            api_key=DOUBAO_API_KEY,
            model_endpoint=DOUBAO_MODEL_ENDPOINT,
        )
    else:
        chunks = level1_pipeline.process_pdf_single_column(
            pdf_path=str(pdf_path),
            image_dir=str(image_dir),
            model_dir=BERT_MODEL_DIR,
            api_key=DOUBAO_API_KEY,
            model_endpoint=DOUBAO_MODEL_ENDPOINT,
        )

    level1_pipeline.save_chunks(chunks, str(output_dir))
    return chunks


# ================================================================
# Endpoint: 上传PDF
# ================================================================
@app.post("/api/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    email: str = Form(...),
    has_two_columns: bool = Form(False),
):
    # --- 文件类型校验 ---
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # --- 生成worksheet_id, 创建工作目录 ---
    worksheet_id = str(uuid.uuid4())
    work_dir = get_worksheet_dir(worksheet_id)
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    image_dir = work_dir / "images"
    image_dir.mkdir(exist_ok=True)

    # --- 保存上传的PDF ---
    pdf_path = work_dir / "original.pdf"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # --- 页数校验 ---
    try:
        page_count = check_pdf_pages(pdf_path)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Could not read PDF file. Is it a valid PDF?")

    if page_count > MAX_PAGES:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"PDF has {page_count} pages. Maximum allowed is {MAX_PAGES} pages."
        )

    # --- 运行Level1 ---
    try:
        chunks = run_level1(pdf_path, has_two_columns, chunks_dir, image_dir)
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    # --- 生成策略报告 (Level2A, 不调用Claude) ---
    strategy_report = level2_claude.generate_strategy_report(chunks)

    # --- 提取worksheet标题 ---
    worksheet_title = ""
    for c in chunks:
        if c.get("task_title"):
            worksheet_title = c["task_title"]
            break
    worksheet_title = worksheet_title or "Adapted Worksheet"

    # --- 保存元数据 ---
    meta = {
        "worksheet_id": worksheet_id,
        "email": email,
        "has_two_columns": has_two_columns,
        "page_count": page_count,
        "worksheet_title": worksheet_title,
        "created_at": datetime.now().isoformat(),
        "status": "uploaded",
    }
    with open(work_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # --- 后台清理旧文件 ---
    background_tasks.add_task(cleanup_old_worksheets)

    return {
        "worksheet_id": worksheet_id,
        "page_count": page_count,
        "worksheet_title": worksheet_title,
        "strategy_report": strategy_report,
        "notices": {
            "content_scope": (
                "Please only upload material that can be given directly to students "
                "as practice (no syllabi, teacher notes, or personal/private information)."
            ),
            "listening_tasks": (
                "Note: this tool does not currently adapt listening exercises. "
                "Listening tasks will be processed but the adaptation may not be optimal."
            ),
        },
    }


# ================================================================
# Endpoint: 改编 (Level2 + Level3)
# ================================================================
@app.post("/api/adapt")
async def adapt_worksheet(
    worksheet_id: str = Form(...),
    strategies: str = Form(...),  # JSON字符串数组, e.g. '["pre_training","signaling"]'
):
    work_dir = STORAGE_DIR / worksheet_id
    if not work_dir.exists():
        raise HTTPException(status_code=404, detail="Worksheet not found. It may have expired.")

    meta_path = work_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Worksheet metadata not found.")
    meta = json.loads(meta_path.read_text())

    # --- 解析策略列表 ---
    try:
        selected_strategies = json.loads(strategies)
        if not isinstance(selected_strategies, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="`strategies` must be a JSON array of strings.")

    invalid = [s for s in selected_strategies if s not in level2_claude.SELECTABLE_STRATEGIES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategies: {invalid}. Allowed: {level2_claude.SELECTABLE_STRATEGIES}"
        )

    chunks_dir = work_dir / "chunks"
    adapted_dir = work_dir / "adapted"

    # --- Level 2: 调用Claude API ---
    try:
        merged, _report = level2_claude.run(
            input_dir=str(chunks_dir),
            selected_strategies=selected_strategies,
            output_dir=str(adapted_dir),
            worksheet_title=meta.get("worksheet_title", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude adaptation failed: {e}")

    # --- Level 3: 生成PDF ---
    pdf_path = work_dir / "adapted.pdf"
    try:
        level3_pdf.run(
            input_dir=str(adapted_dir),
            output_path=str(pdf_path),
            worksheet_title=meta.get("worksheet_title", "Adapted Worksheet"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    # --- 更新元数据 ---
    meta["status"] = "adapted"
    meta["selected_strategies"] = selected_strategies
    meta["adapted_at"] = datetime.now().isoformat()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "worksheet_id": worksheet_id,
        "download_url": f"/api/download/{worksheet_id}",
    }


# ================================================================
# Endpoint: 下载PDF
# ================================================================
@app.get("/api/download/{worksheet_id}")
async def download_pdf(worksheet_id: str):
    work_dir = STORAGE_DIR / worksheet_id
    pdf_path = work_dir / "adapted.pdf"

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Adapted PDF not found. Run /api/adapt first.")

    meta_path = work_dir / "meta.json"
    filename = "adapted_worksheet.pdf"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        title = meta.get("worksheet_title", "adapted_worksheet")
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        filename = f"{safe_title}.pdf"

    return FileResponse(path=pdf_path, filename=filename, media_type="application/pdf")


# ================================================================
# Health check
# ================================================================
@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
