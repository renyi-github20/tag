#!/usr/bin/env python3
"""
公告抽取服务 - 统一 HTTP API，按 type 参数调用不同类别的抽取逻辑。

支持类型: esg_report, periodic_report, meeting_materials, ir_qa, inquiry_letters, proposal_reference, governance
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v and v.isdigit() else default


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, "").strip() in ("1", "true", "yes")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 类型与中文名映射
TYPE_NAMES: dict[str, str] = {
    "esg_report": "ESG报告",
    "periodic_report": "定期报告",
    "meeting_materials": "会议资料",
    "ir_qa": "投关问答",
    "inquiry_letters": "闻讯函件",
    "proposal_reference": "议案参考",
    "governance": "治理制度",
}

# 抽取函数类型: (pdf_path, config, dpi, max_pages, **kwargs) -> dict | list[dict]
ExtractorFn = Callable[..., Union[dict[str, Any], list[dict[str, Any]]]]


def _get_extractor(type_key: str) -> tuple[ExtractorFn, bool]:
    """
    根据 type 返回对应的抽取函数。
    返回 (extract_fn, returns_list): 若 returns_list 为 True，则单文件可能返回多条记录。
    """
    if type_key == "esg_report":
        from scripts.extract_esg_reports import call_vl_extract, load_config, load_fields_and_comments

        def _run(pdf_path: Path, config: dict, dpi: int, max_pages: int, **kw) -> dict:
            fields, comments = load_fields_and_comments(kw["requirement_path"])
            parallel_workers = kw.get("parallel_workers")
            if parallel_workers is None and config.get("vl"):
                parallel_workers = config["vl"].get("parallel_workers", 0)
            jpeg_quality = kw.get("jpeg_quality")
            if jpeg_quality is None and config.get("vl"):
                jpeg_quality = config["vl"].get("jpeg_quality", 80)
            return call_vl_extract(
                pdf_path=pdf_path, fields=fields, comments=comments,
                config=config, dpi=dpi, max_pages=max_pages,
                parallel_workers=parallel_workers or 0,
                jpeg_quality=jpeg_quality or 80,
            )
        return _run, False

    if type_key == "periodic_report":
        from scripts.extract_periodic_reports import call_vl_extract, load_config, load_fields_and_comments

        def _run(pdf_path: Path, config: dict, dpi: int, max_pages: int, **kw) -> dict:
            fields, comments = load_fields_and_comments(kw["requirement_path"])
            return call_vl_extract(
                pdf_path=pdf_path, fields=fields, comments=comments,
                config=config, dpi=dpi, max_pages=max_pages,
            )
        return _run, False

    if type_key == "meeting_materials":
        from scripts.extract_meeting_materials import call_vl_extract, load_config, load_fields_and_comments

        def _run(pdf_path: Path, config: dict, dpi: int, max_pages: int, **kw) -> dict:
            fields, comments = load_fields_and_comments(kw["requirement_path"])
            return call_vl_extract(
                pdf_path=pdf_path, fields=fields, comments=comments,
                config=config, dpi=dpi, max_pages=max_pages,
            )
        return _run, False

    if type_key == "ir_qa":
        from scripts.extract_ir_qa import (
            get_llm_config,
            get_vl_config,
            load_config,
            load_fields_and_comments,
            process_one_file,
        )

        def _run(pdf_path: Path, config: dict, dpi: int, max_pages: int, **kw) -> list[dict]:
            fields, comments = load_fields_and_comments(kw["requirement_path"])
            llm_config = get_llm_config(config, "default")
            vl_config = get_vl_config(config) if config.get("vl") else None
            max_qa = kw.get("max_qa", 0)
            return process_one_file(
                pdf_path, fields, comments, llm_config,
                vl_config=vl_config, dpi=dpi, max_pages=max_pages, max_qa=max_qa,
            )
        return _run, True

    if type_key == "inquiry_letters":
        from scripts.extract_inquiry_letters import call_vl_extract, load_config, load_fields_and_comments

        def _run(pdf_path: Path, config: dict, dpi: int, max_pages: int, **kw) -> dict:
            fields, comments = load_fields_and_comments(kw["requirement_path"])
            return call_vl_extract(
                pdf_path=pdf_path, fields=fields, comments=comments,
                config=config, dpi=dpi, max_pages=max_pages,
            )
        return _run, False

    if type_key == "proposal_reference":
        from scripts.extract_proposal_reference import call_vl_extract, load_config, load_fields_and_comments

        def _run(pdf_path: Path, config: dict, dpi: int, max_pages: int, **kw) -> dict:
            _, comments = load_fields_and_comments(kw["requirement_path"])
            return call_vl_extract(
                pdf_path=pdf_path, comments=comments,
                config=config, dpi=dpi, max_pages=max_pages,
            )
        return _run, False

    if type_key == "governance":
        from scripts.extract_governance import call_vl_extract, load_config, load_fields_and_comments

        def _run(pdf_path: Path, config: dict, dpi: int, max_pages: int, **kw) -> dict:
            fields, comments = load_fields_and_comments(kw["requirement_path"])
            return call_vl_extract(
                pdf_path=pdf_path, fields=fields, comments=comments,
                config=config, dpi=dpi, max_pages=max_pages,
            )
        return _run, False

    raise ValueError(f"未知类型: {type_key}")


def _load_config() -> dict[str, Any]:
    from scripts.extract_esg_reports import load_config
    config_path = _PROJECT_ROOT / "config.yaml"
    requirement_path = _PROJECT_ROOT / "data" / "requirement" / "requirement_1.json"
    if not config_path.exists():
        raise RuntimeError(f"配置文件不存在: {config_path}")
    if not requirement_path.exists():
        raise RuntimeError(f"需求文件不存在: {requirement_path}")
    config = load_config(config_path)
    config["_requirement_path"] = requirement_path
    return config


app = FastAPI(
    title="公告抽取服务",
    description="按 type 参数选择公告类别，上传 PDF 抽取结构化字段。支持：ESG报告、定期报告、会议资料、投关问答、闻讯函件、议案参考、治理制度",
    version="1.0.0",
)

_config: dict | None = None


def _get_config() -> dict:
    global _config
    if _config is None:
        _config = _load_config()
    return _config


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok", "service": "extract", "types": list(TYPE_NAMES.keys())}


@app.get("/types")
def list_types():
    """列出支持的公告类型"""
    return {"types": TYPE_NAMES}


@app.post("/extract")
async def extract(
    file: UploadFile = File(..., description="PDF 文件"),
    type: str = Query(..., description="公告类型", alias="type"),
    dpi: int = Query(150, description="PDF 转图 DPI"),
    max_pages: int = Query(50, description="单批最大页数"),
    max_qa: int = Query(0, description="投关问答单文件最多 Q&A 条数（0=全部，仅 type=ir_qa 有效）"),
    parallel_workers: int = Query(0, description="多批时并行 VL 数（0=串行，2~4 可提速，仅部分 type 有效）"),
    jpeg_quality: int = Query(80, description="PDF 转图 JPEG 质量 1-100，越低越快（仅部分 type 有效）"),
):
    """
    上传 PDF，按 type 抽取对应类别的结构化字段。

    - **type**: 公告类型，必填
    - **file**: PDF 文件
    - **dpi**: 转图 DPI，默认 150
    - **max_pages**: 单批最大页数，默认 50
    - **max_qa**: 投关问答单文件最多 Q&A 条数（0=全部），仅 type=ir_qa 有效
    """
    type_key = type.strip().lower()
    if type_key not in TYPE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"未知类型: {type}。支持: {', '.join(TYPE_NAMES.keys())}",
        )

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="请上传 PDF 文件")

    try:
        config = _get_config()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "vl" not in config or not config.get("vl", {}).get("enable", True):
        raise HTTPException(
            status_code=503,
            detail="VL 抽取未启用或配置缺失，请检查 config.yaml 中的 vl 配置",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")

    extract_fn, returns_list = _get_extractor(type_key)
    req_path = config.get("_requirement_path", _PROJECT_ROOT / "data" / "requirement" / "requirement_1.json")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = extract_fn(
            tmp_path, config, dpi, max_pages,
            requirement_path=req_path,
            max_qa=max_qa,
            parallel_workers=parallel_workers,
            jpeg_quality=jpeg_quality,
        )
        if returns_list:
            for rec in result:
                if isinstance(rec, dict) and "filename" not in rec:
                    rec["filename"] = file.filename
                elif isinstance(rec, dict):
                    rec["filename"] = file.filename
            return JSONResponse(content={"records": result, "count": len(result)})
        else:
            if isinstance(result, dict):
                result["filename"] = file.filename
            return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"抽取失败: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)


class ExtractPathRequest(BaseModel):
    path: str
    type: str
    dpi: int = 150
    max_pages: int = 50
    max_qa: int = 0
    parallel_workers: int = 0
    jpeg_quality: int = 80


@app.post("/extract/path")
async def extract_by_path(body: ExtractPathRequest):
    """
    按本地路径抽取（用于服务内部或可信环境）。

    - **path**: PDF 文件相对或绝对路径
    - **type**: 公告类型，同 /extract
    """
    type_key = body.type.strip().lower()
    if type_key not in TYPE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"未知类型: {body.type}。支持: {', '.join(TYPE_NAMES.keys())}",
        )

    try:
        config = _get_config()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "vl" not in config or not config.get("vl", {}).get("enable", True):
        raise HTTPException(
            status_code=503,
            detail="VL 抽取未启用或配置缺失，请检查 config.yaml 中的 vl 配置",
        )

    pdf_path = Path(body.path)
    if not pdf_path.is_absolute():
        pdf_path = _PROJECT_ROOT / body.path
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {body.path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    extract_fn, returns_list = _get_extractor(type_key)
    req_path = config.get("_requirement_path", _PROJECT_ROOT / "data" / "requirement" / "requirement_1.json")

    try:
        result = extract_fn(
            pdf_path, config, body.dpi, body.max_pages,
            requirement_path=req_path,
            max_qa=body.max_qa,
            parallel_workers=body.parallel_workers,
            jpeg_quality=body.jpeg_quality,
        )
        if returns_list:
            return JSONResponse(content={"records": result, "count": len(result)})
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"抽取失败: {str(e)}")


def _iter_input_files(input_path: Path, type_key: str) -> Iterable[Path]:
    """按类型遍历输入路径下的文件（PDF 或 txt+pdf）。"""
    if type_key == "ir_qa":
        from scripts.extract_ir_qa import iter_input_files
        return iter_input_files(input_path)
    from scripts.extract_esg_reports import iter_pdfs
    return iter_pdfs(input_path)


def _load_done_filenames(output_path: Path) -> set[str]:
    """从已有 JSONL 读取已完成的 filename。"""
    from scripts.extract_esg_reports import load_done_filenames
    return load_done_filenames(output_path)


class ExtractBatchRequest(BaseModel):
    """批量抽取请求体"""
    path: str
    type: str
    limit: Optional[int] = None
    skip: Optional[int] = None
    output: str = ""
    append: bool = False
    resume: bool = False
    dpi: Optional[int] = None
    max_pages: Optional[int] = None
    max_qa: int = 0
    parallel_workers: int = 0
    jpeg_quality: int = 80
    no_verify_ssl: bool = False


@app.post("/extract/batch")
async def extract_batch(body: ExtractBatchRequest):
    """
    批量抽取：按目录路径处理多个文件。

    - **path**: 输入目录或单个文件路径（相对项目根或绝对路径）
    - **type**: 公告类型
    - **limit**: 最多处理数量（0=不限制）
    - **skip**: 跳过前 N 个文件
    - **output**: 输出 JSONL 路径（空则只返回结果不写入）
    - **append**: 追加写入 output
    - **resume**: 断点续跑，跳过 output 中已有的 filename
    - **dpi**, **max_pages**, **max_qa**: 同单文件抽取
    - **no_verify_ssl**: 关闭 VL 请求 SSL 校验
    """
    type_key = body.type.strip().lower()
    if type_key not in TYPE_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"未知类型: {body.type}。支持: {', '.join(TYPE_NAMES.keys())}",
        )

    try:
        config = _get_config()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    limit = body.limit if body.limit is not None else _env_int("EXTRACT_DEFAULT_LIMIT", 0)
    skip = body.skip if body.skip is not None else _env_int("EXTRACT_DEFAULT_SKIP", 0)
    dpi = body.dpi if body.dpi is not None else _env_int("EXTRACT_DEFAULT_DPI", 150)
    max_pages = body.max_pages if body.max_pages is not None else _env_int("EXTRACT_DEFAULT_MAX_PAGES", 50)
    no_verify_ssl = body.no_verify_ssl or _env_bool("EXTRACT_NO_VERIFY_SSL", False)

    if no_verify_ssl:
        config.setdefault("vl", {})["verify_ssl"] = False

    if "vl" not in config or not config.get("vl", {}).get("enable", True):
        raise HTTPException(
            status_code=503,
            detail="VL 抽取未启用或配置缺失，请检查 config.yaml 中的 vl 配置",
        )

    input_path = Path(body.path)
    if not input_path.is_absolute():
        input_path = _PROJECT_ROOT / body.path
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在: {body.path}")

    files = list(_iter_input_files(input_path, type_key))
    if skip > 0:
        files = files[skip:]
    if limit > 0:
        files = files[: limit]

    output_path = None
    if body.output:
        output_path = Path(body.output) if Path(body.output).is_absolute() else _PROJECT_ROOT / body.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if body.resume and output_path.exists():
            done = _load_done_filenames(output_path)
            files = [p for p in files if p.name not in done]

    extract_fn, returns_list = _get_extractor(type_key)
    req_path = config.get("_requirement_path", _PROJECT_ROOT / "data" / "requirement" / "requirement_1.json")

    records: list[dict[str, Any]] = []
    mode = "a" if (body.append or body.resume) and body.output else "w"
    out_file = output_path.open(mode, encoding="utf-8") if output_path else None

    try:
        for i, file_path in enumerate(files):
            try:
                result = extract_fn(
                    file_path, config, dpi, max_pages,
                    requirement_path=req_path,
                    max_qa=body.max_qa,
                    parallel_workers=body.parallel_workers,
                    jpeg_quality=body.jpeg_quality,
                )
                if returns_list:
                    for rec in result:
                        if isinstance(rec, dict):
                            records.append(rec)
                            if out_file:
                                out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                out_file.flush()
                else:
                    if isinstance(result, dict):
                        records.append(result)
                        if out_file:
                            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_file.flush()
            except Exception as e:
                records.append({"filename": file_path.name, "_error": str(e)})
        if out_file:
            out_file.close()
        return JSONResponse(content={
            "total": len(files),
            "success": len([r for r in records if "_error" not in r]),
            "records": records,
            "output": body.output if body.output else None,
        })
    except Exception as e:
        if out_file:
            out_file.close()
        raise HTTPException(status_code=500, detail=f"批量抽取失败: {str(e)}")
