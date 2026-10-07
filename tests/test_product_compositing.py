import base64
from pathlib import Path

import httpx
import pytest

from app.services.product_compositing import SeedreamArkClient


@pytest.mark.asyncio
async def test_seedream_client_parses_base64(monkeypatch, tmp_path):
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    class Response:
        status_code = 200
        def json(self):
            return {"data": [{"b64_json": base64.b64encode(b"scene").decode()}]}
        text = ""
    async def post(self, *args, **kwargs):
        return Response()
    monkeypatch.setattr(httpx.AsyncClient, "post", post)
    monkeypatch.setattr("app.services.product_compositing.get_settings", lambda: type("S", (), {"ark_api_key":"x", "ark_base_url":"https://ark.test", "seedream_model":"test", "ai_api_timeout":3})())
    assert await SeedreamArkClient().generate(str(source), "prompt") == b"scene"


@pytest.mark.asyncio
async def test_tencent_missing_configuration(monkeypatch, tmp_path):
    from app.services.product_compositing import TencentGoodsMattingClient, ProductCompositingError
    monkeypatch.setattr("app.services.product_compositing.get_settings", lambda: type("S", (), {"tencent_secret_id":"", "tencent_secret_key":"", "tencent_cos_bucket":""})())
    with pytest.raises(ProductCompositingError, match="not configured"):
        await TencentGoodsMattingClient().process(str(tmp_path / "x.png"))
