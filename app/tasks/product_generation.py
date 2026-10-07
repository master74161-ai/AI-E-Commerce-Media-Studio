"""Celery task for Dify-compatible product image generation."""

import asyncio
import shutil
import tempfile
from pathlib import Path

from celery import Task

from app.core.celery_app import celery_app
from app.core.config import get_settings
from app.services.product_compositing import generate_product_image
from app.services.storage import GCSStorage, LocalStorage, StorageService
from app.services.result_callback import publish_result


class ProductGenerationTask(Task):
    """Task base that exposes progress states to the existing API."""


@celery_app.task(
    bind=True,
    base=ProductGenerationTask,
    name="app.tasks.product_generation.generate_product_image_task",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_kwargs={"max_retries": 2},
)
def generate_product_image_task(
    self: ProductGenerationTask,
    task_id: str,
    reference_path: str,
    prompt: str,
    mode: str,
    size: str = "2048x2048",
    image_type: str = "overall",
    position: int = 0,
    purpose: str = "",
) -> dict[str, str]:
    """Generate and store a product-faithful JPEG."""
    settings = get_settings()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"product_{task_id}_"))
    try:
        if reference_path.startswith("gs://") or settings.storage_type == "gcs":
            storage: StorageService = GCSStorage(bucket_name=settings.gcs_bucket_name)
            local_input = temp_dir / "reference.jpg"
            source_key = reference_path.replace(f"gs://{settings.gcs_bucket_name}/", "")
            asyncio.run(storage.download(source_key, str(local_input)))
        else:
            storage = LocalStorage(base_dir=Path(settings.local_storage_path))
            local_input = temp_dir / Path(reference_path).name
            shutil.copy2(reference_path, local_input)

        output = temp_dir / "generated.jpg"
        self.update_state(state="GENERATING", meta={"task_id": task_id, "mode": mode})
        allowed_types = {"overall", "scene", "cut", "usage", "display"}
        if image_type not in allowed_types:
            raise ValueError(f"Unsupported initialization image_type: {image_type}")
        asyncio.run(generate_product_image(str(local_input), prompt, str(output), size))
        result_url = asyncio.run(storage.upload(str(output), f"processed/{task_id}/generated.jpg"))
        result = {"task_id": task_id, "status": "COMPLETED", "result_url": result_url, "image_type": image_type, "position": str(position), "purpose": purpose}
        asyncio.run(publish_result(result))
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
