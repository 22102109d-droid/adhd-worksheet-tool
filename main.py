"""
main.py
================
FastAPI后端，整合三层pipeline，供Webflow前端调用。
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import uuid
import shutil
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download, snapshot_download

BERT_MODEL_DIR = os.environ.get("BERT_MODEL_DIR", "model")
if not os.path.exists(f"{BERT_MODEL_DIR}/best_model.pt"):
    print("Downloading model from Hugging Face...")
    os.makedirs(f"{BERT_MODEL_DIR}/tokenizer", exist_ok=True)
    downloaded = hf_hub_download(
        repo_id="mellyii/adhd-bert-model",
        filename="best_model.pt",
        token=os.environ.get("HF_TOKEN"),
    )
    shutil.copy(downloaded, f"{BERT_MODEL_DIR}/best_model.pt")
    print(f"Copied model to {BERT_MODEL_DIR}/best_model.pt")
    snap = snapshot_download(
        repo_id="mellyii/adhd-bert-model",
        allow_patterns="tokenizer/*",
        token=os.environ.get("HF_TOKEN"),
    )
    for f in Path(snap).glob("tokenizer/*"):
        shutil.copy(str(f), f"{BERT_MODEL_DIR}/tokenizer/{f.name}")
    print("Model downloaded!")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import fitz

import level1_pipeline
import level2_claude
import level3_pdf

STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)

MAX_PAGES = 3
FILE_RETENTION_HOURS = 24
DOUBAO_API_KEY = os.environ.get("DOUBAO_API_KEY", "")
DOUBAO_MODEL_ENDPOINT = os.environ.get("DOUBAO_MODEL_ENDPOINT", "")
MOCK_LEVEL1 = os.environ.get("MOCK_LEVEL1", "false").lower() == "true"

# 后台线程池，用于运行BERT推理
executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="ADHD Worksheet Adapter API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_worksheet_dir(worksheet_id: str) -> Path:
    d = STORAGE_DIR / worksheet_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_old_worksheets():
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
    doc = fitz.open(pdf_path)
    n = doc.page_count
    doc.close()
    return n


def run_level1_mock(pdf_path, has_two_columns, output_dir):
    mock_chunks = [{
        "chunk_id": 1, "order": 1, "chunk_type": "discussion",
        "type_description": "", "task_title": "Warmer",
        "instruction": "Read the text and discuss with a partner.",
        "content": "This is mock content for testing.",
        "has_image": False, "image_files": [], "image_type": "none", "image_relevance": "none",
        "has_pre_training": False, "task_decomposition_needed": None, "linked_to": None,
        "has_signaling": False, "signaling_details": {"signals_found": []},
        "has_multimedia": False, "multimedia_details": {"meaningful_images": 0},
    }]
    level1_pipeline.save_chunks(mock_chunks, str(output_dir))
    return mock_chunks


def run_level1(pdf_path, has_two_columns, output_dir, image_dir):
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


def process_pdf_background(worksheet_id: str, pdf_path: Path, has_two_columns: bool,
                            chunks_dir: Path, image_dir: Path, work_dir: Path,
                            email: str, page_count: int):
    """在后台线程里跑BERT推理，完成后更新meta.json状态"""
    meta_path = work_dir / "meta.json"
    try:
        chunks = run_level1(pdf_path, has_two_columns, chunks_dir, image_dir)
        strategy_report = level2_claude.generate_strategy_report(chunks)

        worksheet_title = ""
        for c in chunks:
            if c.get("task_title"):
                worksheet_title = c["task_title"]
                break
        worksheet_title = worksheet_title or "Adapted Worksheet"

        meta = json.loads(meta_path.read_text())
        meta["status"] = "ready"
        meta["worksheet_title"] = worksheet_title
        meta["strategy_report"] = strategy_report
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[{worksheet_id}] 处理完成")

    except Exception as e:
        print(f"[{worksheet_id}] 处理失败: {e}")
        try:
            meta = json.loads(meta_path.read_text())
            meta["status"] = "error"
            meta["error"] = str(e)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


@app.post("/api/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    email: str = Form(...),
    has_two_columns: bool = Form(False),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    worksheet_id = str(uuid.uuid4())
    work_dir = get_worksheet_dir(worksheet_id)
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    image_dir = work_dir / "images"
    image_dir.mkdir(exist_ok=True)

    pdf_path = work_dir / "original.pdf"
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

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

    # 立刻写入meta，状态为processing
    meta = {
        "worksheet_id": worksheet_id,
        "email": email,
        "has_two_columns": has_two_columns,
        "page_count": page_count,
        "worksheet_title": "",
        "created_at": datetime.now().isoformat(),
        "status": "processing",
    }
    with open(work_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 后台线程跑BERT推理，不阻塞响应
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        process_pdf_background,
        worksheet_id, pdf_path, has_two_columns,
        chunks_dir, image_dir, work_dir, email, page_count
    )

    background_tasks.add_task(cleanup_old_worksheets)

    # 立刻返回worksheet_id，前端轮询/api/status
    return {
        "worksheet_id": worksheet_id,
        "page_count": page_count,
        "status": "processing",
    }


@app.get("/api/status/{worksheet_id}")
async def get_status(worksheet_id: str):
    """前端轮询这个接口，status变成ready后返回strategy_report"""
    work_dir = STORAGE_DIR / worksheet_id
    if not work_dir.exists():
        raise HTTPException(status_code=404, detail="Worksheet not found.")
    meta_path = work_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Worksheet not found.")
    meta = json.loads(meta_path.read_text())

    if meta["status"] == "processing":
        return {"status": "processing"}
    elif meta["status"] == "error":
        return {"status": "error", "error": meta.get("error", "Processing failed.")}
    else:
        return {
            "status": "ready",
            "worksheet_id": worksheet_id,
            "page_count": meta["page_count"],
            "worksheet_title": meta["worksheet_title"],
            "strategy_report": meta["strategy_report"],
        }


@app.post("/api/adapt")
async def adapt_worksheet(
    worksheet_id: str = Form(...),
    strategies: str = Form(...),
):
    work_dir = STORAGE_DIR / worksheet_id
    if not work_dir.exists():
        raise HTTPException(status_code=404, detail="Worksheet not found. It may have expired.")

    meta_path = work_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Worksheet metadata not found.")
    meta = json.loads(meta_path.read_text())

    if meta["status"] != "ready":
        raise HTTPException(status_code=400, detail="Worksheet is still processing. Please wait.")

    try:
        selected_strategies = json.loads(strategies)
        if not isinstance(selected_strategies, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="`strategies` must be a JSON array of strings.")

    invalid = [s for s in selected_strategies if s not in level2_claude.SELECTABLE_STRATEGIES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid strategies: {invalid}.")

    chunks_dir = work_dir / "chunks"
    adapted_dir = work_dir / "adapted"

    try:
        merged, _report = level2_claude.run(
            input_dir=str(chunks_dir),
            selected_strategies=selected_strategies,
            output_dir=str(adapted_dir),
            worksheet_title=meta.get("worksheet_title", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude adaptation failed: {e}")

    try:
        level3_pdf.run(
            input_dir=str(adapted_dir),
            output_path=str(work_dir / "adapted.pdf"),
            worksheet_title=meta.get("worksheet_title", "Adapted Worksheet"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File generation failed: {e}")

    meta["status"] = "adapted"
    meta["selected_strategies"] = selected_strategies
    meta["adapted_at"] = datetime.now().isoformat()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "worksheet_id": worksheet_id,
        "download_url": f"/api/download/{worksheet_id}",
    }


@app.get("/api/download/{worksheet_id}")
async def download_html(worksheet_id: str):
    work_dir = STORAGE_DIR / worksheet_id
    html_path = work_dir / "adapted.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Adapted file not found.")
    meta_path = work_dir / "meta.json"
    filename = "adapted_worksheet.html"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        title = meta.get("worksheet_title", "adapted_worksheet")
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        filename = f"{safe_title}.html"
    return FileResponse(path=html_path, filename=filename, media_type="text/html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
