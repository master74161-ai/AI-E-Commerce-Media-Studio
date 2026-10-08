"""Product-faithful image generation for the Dify workflows.

Seedream creates the scene while the original product cutout is composited back
over it. This keeps the product pixels out of the generative step's final image.
"""

import base64
import binascii
import logging
import re
from pathlib import Path
from urllib.parse import urljoin

import httpx
from PIL import Image, ImageFilter

from app.core.config import get_settings
from app.services.ai_service import AIServiceError, BackgroundRemovalService

logger = logging.getLogger(__name__)


class ProductCompositingError(AIServiceError):
    """Raised when product-faithful generation cannot complete."""


class SeedreamArkClient:
    """Small client for the official Volcengine Ark image API."""

    async def generate(
        self, reference_path: str | None, prompt: str, size: str = "2048x2048"
    ) -> bytes:
        settings = get_settings()
        if not settings.ark_api_key:
            raise ProductCompositingError("ARK_API_KEY is not configured")
        url = urljoin(settings.ark_base_url.rstrip("/") + "/", "images/generations")
        headers = {"Authorization": f"Bearer {settings.ark_api_key}"}
        payload = {
            "model": settings.seedream_model,
            "prompt": prompt,
            "size": size,
            "response_format": "url",
            "watermark": False,
        }
        if reference_path:
            image_bytes = Path(reference_path).read_bytes()
            encoded = base64.b64encode(image_bytes).decode("ascii")
            data_url = "data:image/jpeg;base64," + encoded
            payload["image"] = data_url
        async with httpx.AsyncClient(timeout=settings.ai_api_timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code >= 400:
                raise ProductCompositingError(
                    f"Seedream Ark error {response.status_code}: {response.text[:500]}"
                )
            body = response.json()
            item = (body.get("data") or [{}])[0]
            result_url = item.get("url") or item.get("b64_json")
            if not result_url:
                raise ProductCompositingError("Seedream response contains no image")
            if result_url.startswith("data:"):
                try:
                    return base64.b64decode(result_url.split(",", 1)[1])
                except (ValueError, binascii.Error) as exc:
                    raise ProductCompositingError("Invalid Seedream base64 image") from exc
            if re.fullmatch(r"[A-Za-z0-9+/=\s]+", result_url) and "://" not in result_url:
                return base64.b64decode(result_url)
            image_response = await client.get(result_url)
            image_response.raise_for_status()
            return image_response.content


class TencentGoodsMattingClient:
    """Tencent COS Data万象 GoodsMatting adapter.

    The service processes the source object with ``ci-process=GoodsMatting``
    and returns a transparent PNG in the response body.
    """

    async def process(self, image_path: str) -> str:
        import asyncio
        import uuid

        settings = get_settings()
        required = {
            "TENCENT_SECRET_ID": settings.tencent_secret_id,
            "TENCENT_SECRET_KEY": settings.tencent_secret_key,
            "TENCENT_COS_BUCKET": settings.tencent_cos_bucket,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ProductCompositingError(
                "Tencent GoodsMatting is not configured: " + ", ".join(missing)
            )

        def blocking() -> bytes:
            from qcloud_cos import CosConfig, CosS3Client

            config = CosConfig(
                Region=settings.tencent_cos_region,
                SecretId=settings.tencent_secret_id,
                SecretKey=settings.tencent_secret_key,
                Scheme="https",
            )
            client = CosS3Client(config)
            key = f"{settings.tencent_cos_prefix.rstrip('/')}/{uuid.uuid4().hex}.png"
            try:
                # Read the payload first. Passing an open file object can make the
                # SDK/proxy choose chunked transfer. Let requests derive the length
                # from the bytes payload; explicitly adding ContentLength causes the
                # installed COS SDK to sign a header that the proxy rewrites.
                source = Path(image_path).read_bytes()
                client.put_object(
                    Bucket=settings.tencent_cos_bucket,
                    Body=source,
                    Key=key,
                    ContentType="image/png",
                )
                signed_url = client.get_presigned_download_url(
                    Bucket=settings.tencent_cos_bucket,
                    Key=key,
                    Params={"ci-process": "GoodsMatting", "center-layout": "0"},
                    UseCiEndPoint=False,
                )
                import requests
                response = requests.get(signed_url, timeout=180)
                response.raise_for_status()
                body = response.content
                if not body.startswith(b"\x89PNG"):
                    raise ProductCompositingError("Tencent GoodsMatting did not return PNG")
                return body
            finally:
                try:
                    client.delete_object(Bucket=settings.tencent_cos_bucket, Key=key)
                except Exception:
                    logger.warning("Failed to delete temporary Tencent COS object", exc_info=True)

        output_path = str(Path(image_path).parent / "bg_removed.png")
        content = await asyncio.to_thread(blocking)
        Path(output_path).write_bytes(content)
        return output_path


def _fit_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Contain an image on a transparent canvas without stretching it."""
    image = image.convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox:
        left, top, right, bottom = bbox
        padding = int(max(right - left, bottom - top) * 0.03)
        image = image.crop((
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        ))
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    left = (size[0] - image.width) // 2
    top = (size[1] - image.height) // 2
    canvas.alpha_composite(image, (left, top))
    return canvas


def composite_product(background: bytes, cutout_path: str, output_path: str, size: str) -> None:
    """Composite the RMBG cutout over the generated scene with a soft shadow."""
    try:
        width, height = (int(part) for part in size.lower().split("x", 1))
    except (ValueError, AttributeError) as exc:
        raise ProductCompositingError(f"Invalid image size: {size}") from exc
    scene = Image.open(__import__("io").BytesIO(background)).convert("RGB").resize(
        (width, height), Image.Resampling.LANCZOS
    )
    settings = get_settings()
    product = _fit_canvas(
        Image.open(cutout_path),
        (int(width * settings.product_scale), int(height * settings.product_scale)),
    )
    alpha = product.getchannel("A")
    shadow = Image.new("RGBA", product.size, (0, 0, 0, settings.shadow_opacity))
    shadow.putalpha(alpha.filter(ImageFilter.GaussianBlur(settings.shadow_blur)))
    shadow_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pos = (
        (width - product.width) // 2,
        int(height * settings.product_vertical_position),
    )
    shadow_canvas.alpha_composite(shadow, (pos[0] + width // 90, pos[1] + height // 45))
    shadow_canvas.alpha_composite(product, pos)
    result = Image.alpha_composite(scene.convert("RGBA"), shadow_canvas).convert("RGB")
    result.save(output_path, format="JPEG", quality=95, subsampling=0)


async def generate_product_image(
    reference_path: str,
    prompt: str,
    output_path: str,
    size: str = "2048x2048",
) -> str:
    """Run RMBG, generate a scene with Ark, and restore the product pixels."""
    settings = get_settings()
    if settings.rmbg_backend == "tencent_goods_matting":
        cutout = await TencentGoodsMattingClient().process(reference_path)
    else:
        remover = BackgroundRemovalService(
            api_url=settings.rmbg_api_url or None,
            # API mode must not silently load a local model on a small CPU host.
            use_local_model=settings.rmbg_backend != "api",
        )
        if settings.rmbg_backend == "api":
            if settings.replicate_api_token:
                cutout = await remover._process_replicate(reference_path)
            elif settings.rmbg_api_url:
                cutout = await remover._process_api(reference_path)
            else:
                raise ProductCompositingError("RMBG API is not configured")
        else:
            cutout = await remover.process(reference_path)
    # Pass the transparent cutout to Seedream so the original background is not
    # reintroduced as a second product by image-to-image generation.
    scene = await SeedreamArkClient().generate(None, prompt, size)
    composite_product(scene, cutout, output_path, size)
    return output_path
