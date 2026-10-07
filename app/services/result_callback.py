"""Optional result callbacks for existing TOS/Supabase workflow orchestration."""
import httpx
from app.core.config import get_settings

async def publish_result(payload: dict) -> None:
    settings = get_settings()
    if settings.result_webhook_url:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(settings.result_webhook_url, json=payload)
            response.raise_for_status()
    if settings.supabase_url and settings.supabase_service_key:
        url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{settings.supabase_results_table}"
        headers = {"apikey": settings.supabase_service_key, "Authorization": f"Bearer {settings.supabase_service_key}", "Prefer": "return=minimal"}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
