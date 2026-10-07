"""API routes for image processing tasks."""

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.celery_app import celery_app
from app.core.config import Settings, get_settings
from app.schemas.task import (
    TaskMode,
    TaskResponse,
    TaskStatus,
    TaskStatusResponse,
    UploadResponse,
)
from app.services.storage import GCSStorage, LocalStorage, StorageService

router = APIRouter(prefix="/api/v1", tags=["tasks"])


def get_storage_service(
    settings: Settings = Depends(get_settings),
) -> StorageService:
    """Dependency to get storage service based on configuration."""
    if settings.storage_type == "gcs":
        return GCSStorage(
            bucket_name=settings.gcs_bucket_name,
            project_id=settings.gcs_project_id,
            credentials_path=settings.gcs_credentials_path,
        )
    return LocalStorage(base_dir=Path(settings.local_storage_path))


# In-memory store for task metadata (created_at, original_url, etc.)
# Task STATUS is queried from Celery/Redis
task_metadata_store: dict[str, dict[str, Any]] = {}


@router.post("/image-generation", response_model=UploadResponse)
async def generate_product_image(
    reference_image: UploadFile = File(...),
    prompt: str = Form(...),
    mode: str = Form("hero"),
    image_type: str = Form("overall"),
    position: int = Form(0),
    purpose: str = Form(""),
    size: str = Form("2048x2048"),
    storage: StorageService = Depends(get_storage_service),
) -> UploadResponse:
    """Dify-compatible asynchronous product-faithful image generation endpoint."""
    if not reference_image.content_type or not reference_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="reference_image must be an image")
    if mode not in {"hero", "cut", "initialization", "retry"}:
        raise HTTPException(status_code=400, detail="Unsupported generation mode")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    try:
        width, height = (int(part) for part in size.split("x", 1))
        if width < 256 or height < 256 or width > 4096 or height > 4096:
            raise ValueError
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="size must be between 256 and 4096 pixels") from exc

    task_id = str(uuid.uuid4())
    suffix = Path(reference_image.filename or "reference.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
        tmp.write(await reference_image.read())
        tmp_path = tmp.name
    try:
        stored_path = await storage.upload(tmp_path, f"uploads/{task_id}/reference{suffix}")
    finally:
        os.unlink(tmp_path)

    now = datetime.now(timezone.utc)
    task_metadata_store[task_id] = {
        "task_id": task_id,
        "created_at": now,
        "updated_at": now,
        "original_url": stored_path,
        "mode": mode,
        "image_type": image_type,
        "position": position,
        "purpose": purpose,
        "prompt": prompt,
        "size": size,
    }
    celery_task = celery_app.send_task(
        "app.tasks.product_generation.generate_product_image_task",
        args=[task_id, stored_path, prompt, mode, size, image_type, position, purpose],
        task_id=task_id,
    )
    return UploadResponse(
        task_id=task_id,
        message=f"Product image generation queued: {celery_task.id}",
        status=TaskStatus.PENDING,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    mode: str = Form("edit"),
    instruction: Optional[str] = Form(None),
    storage: StorageService = Depends(get_storage_service),
) -> UploadResponse:
    """
    Upload an image for processing.

    This endpoint accepts an image file and processing parameters:
    - mode: "remove_bg" for background removal, "edit" for instruction-based editing
    - instruction: Text instruction for editing (required for mode="edit")
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Validate mode
    try:
        TaskMode(mode)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be 'remove_bg' or 'edit'",
        )

    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Save uploaded file
    file_extension = file.filename.split(".")[-1] if file.filename else "jpg"
    destination_path = f"uploads/{task_id}/original.{file_extension}"

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{file_extension}", mode="wb"
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        stored_path = await storage.upload(tmp_path, destination_path)
    finally:
        os.unlink(tmp_path)

    # Store task metadata (not status - that comes from Celery)
    task_metadata_store[task_id] = {
        "task_id": task_id,
        "created_at": now,
        "updated_at": now,
        "original_url": stored_path,
        "mode": mode,
        "instruction": instruction,
    }

    # Queue the processing task - returns Celery task ID
    celery_task = celery_app.send_task(
        "app.tasks.image_processing.process_image",
        args=[task_id, stored_path, mode, instruction],
        task_id=task_id,  # Use same ID for easy lookup
    )

    return UploadResponse(
        task_id=task_id,
        message=f"Image uploaded successfully. Celery task: {celery_task.id}",
        status=TaskStatus.PENDING,
    )


@router.get("/task-status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Get the current status of a processing task.

    Queries Celery/Redis for the actual task state.
    """
    metadata = task_metadata_store.get(task_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Task not found")

    # Query Celery for actual task status
    celery_result = AsyncResult(task_id, app=celery_app)
    celery_state = celery_result.state

    # Map Celery state to our TaskStatus
    status = _celery_state_to_task_status(celery_state)
    result_url = None
    error = None

    # Get result info if available
    if celery_result.ready():
        if celery_result.successful():
            result_info = celery_result.result
            if isinstance(result_info, dict):
                result_url = result_info.get("result_url")
                if result_info.get("status") == "FAILED" or result_info.get("error"):
                    error = result_info.get("error")
                    status = TaskStatus.FAILED
        else:
            error = str(celery_result.result)

    return TaskStatusResponse(
        task_id=task_id,
        status=status,
        progress=_get_progress_message(status),
        result_url=result_url,
        error=error,
    )


@router.post("/image-generation/{task_id}/retry", response_model=UploadResponse)
async def retry_product_image(task_id: str) -> UploadResponse:
    """Requeue a failed Dify product-generation task with the original contract."""
    metadata = task_metadata_store.get(task_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Task not found")
    result = AsyncResult(task_id, app=celery_app)
    if not result.failed() and result.state not in {"REVOKED"}:
        raise HTTPException(status_code=409, detail="Task is not failed")
    new_task = celery_app.send_task(
        "app.tasks.product_generation.generate_product_image_task",
        args=[task_id, metadata["original_url"], metadata["prompt"], metadata["mode"], metadata.get("size", "2048x2048"), metadata.get("image_type", "overall"), metadata.get("position", 0), metadata.get("purpose", "")],
        task_id=task_id,
    )
    return UploadResponse(task_id=task_id, message=f"Retry queued: {new_task.id}", status=TaskStatus.PENDING)


@router.get("/result/{task_id}", response_model=TaskResponse)
async def get_result(task_id: str) -> TaskResponse:
    """
    Get the full result of a completed task.
    """
    metadata = task_metadata_store.get(task_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Task not found")

    # Query Celery for actual task status
    celery_result = AsyncResult(task_id, app=celery_app)
    celery_state = celery_result.state
    status = _celery_state_to_task_status(celery_state)

    result_url = None
    error = None

    if celery_result.ready():
        if celery_result.successful():
            result_info = celery_result.result
            if isinstance(result_info, dict):
                result_url = result_info.get("result_url")
                if result_info.get("status") == "FAILED" or result_info.get("error"):
                    error = result_info.get("error")
                    status = TaskStatus.FAILED
        else:
            error = str(celery_result.result)

    return TaskResponse(
        task_id=task_id,
        status=status,
        created_at=metadata["created_at"],
        updated_at=datetime.now(timezone.utc),
        original_url=metadata.get("original_url"),
        result_url=result_url,
        error=error,
    )


def _celery_state_to_task_status(celery_state: str) -> TaskStatus:
    """Map Celery state to our TaskStatus enum."""
    state_map = {
        "PENDING": TaskStatus.PENDING,
        "STARTED": TaskStatus.EDITING,
        "REMOVING_BG": TaskStatus.REMOVING_BG,
        "EDITING": TaskStatus.EDITING,
        "SUCCESS": TaskStatus.COMPLETED,
        "FAILURE": TaskStatus.FAILED,
        "REVOKED": TaskStatus.FAILED,
    }
    return state_map.get(celery_state, TaskStatus.PENDING)


def _get_progress_message(status: TaskStatus) -> str:
    """Get human-readable progress message for a status."""
    messages = {
        TaskStatus.PENDING: "Waiting in queue...",
        TaskStatus.REMOVING_BG: "Removing background...",
        TaskStatus.EDITING: "Editing image with AI...",
        TaskStatus.COMPLETED: "Processing complete!",
        TaskStatus.FAILED: "Processing failed",
    }
    return messages.get(status, "Unknown status")
