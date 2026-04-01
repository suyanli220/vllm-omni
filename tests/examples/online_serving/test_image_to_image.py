"""
Online serving tests: image-to-image.
See examples/online_serving/image_to_image/README.md
"""

import base64
import json
import sys
import urllib.request
from pathlib import Path

import pytest

from tests.conftest import OmniServer, OmniServerParams, assert_image_valid
from tests.examples.conftest import EXAMPLES, OUTPUT_DIR, run_cmd
from tests.utils import hardware_marks

pytestmark = [pytest.mark.advanced_model, pytest.mark.example, *hardware_marks(res={"cuda": "H100"})]

I2I_ONLINE_CLIENT = EXAMPLES / "online_serving" / "image_to_image" / "openai_chat_client.py"
EXAMPLE_OUTPUT_SUBFOLDER = "example_online_i2i"

TEST_IMAGE_URL = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png"
TEST_IMAGE_NAME = "qwen-bear.png"

# ---------------------------------------------------------------------------
# Server parameter sets.
# Tests sharing the same param list reuse one module-scoped server instance,
# so keep tests that use the same server adjacent to each other.
# ---------------------------------------------------------------------------

qwen_image_edit_server_params = [OmniServerParams(model="Qwen/Qwen-Image-Edit")]
qwen_image_layered_server_params = [OmniServerParams(model="Qwen/Qwen-Image-Layered")]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def example_output_dir() -> Path:
    d = OUTPUT_DIR / EXAMPLE_OUTPUT_SUBFOLDER
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def input_image() -> Path:
    """Download the test bear image once per module run."""
    image_path = OUTPUT_DIR / TEST_IMAGE_NAME
    if not image_path.exists():
        urllib.request.urlretrieve(TEST_IMAGE_URL, image_path)
    return image_path


# ---------------------------------------------------------------------------
# Method 1: Using curl
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omni_server", qwen_image_edit_server_params, indirect=True)
def test_api_calls_001(omni_server: OmniServer, example_output_dir: Path, input_image: Path):
    """curl: single-image edit via /v1/chat/completions."""
    case_dir = example_output_dir / "api_calls-001"
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "api_calls_001.png"

    img_b64 = base64.b64encode(input_image.read_bytes()).decode()
    url = f"http://{omni_server.host}:{omni_server.port}/v1/chat/completions"

    payload = json.dumps(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Convert this image to watercolor style"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                }
            ],
            "extra_body": {
                "num_inference_steps": 50,
                "guidance_scale": 1.0,
                "seed": 42,
            },
        }
    )

    req_file = case_dir / "request.json"
    req_file.write_text(payload, encoding="utf-8")

    run_cmd(
        f"curl -s '{url}'"
        " -H 'Content-Type: application/json'"
        f" -d @'{req_file}'"
        " | jq -r '.choices[0].message.content[0].image_url.url'"
        f" | cut -d',' -f2- | base64 -d > '{out}'",
        shell=True,
    )
    assert_image_valid(out)


# ---------------------------------------------------------------------------
# Method 2: Using Python Client Script
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omni_server", qwen_image_edit_server_params, indirect=True)
def test_api_calls_002(omni_server: OmniServer, example_output_dir: Path, input_image: Path):
    """openai_chat_client.py: single-image edit."""
    case_dir = example_output_dir / "api_calls-002"
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "api_calls_002.png"

    run_cmd(
        [
            sys.executable,
            str(I2I_ONLINE_CLIENT),
            "--input",
            str(input_image),
            "--prompt",
            "Convert to oil painting style",
            "--output",
            str(out),
            "--server",
            f"http://{omni_server.host}:{omni_server.port}",
            "--steps",
            "50",
            "--seed",
            "42",
        ]
    )
    assert_image_valid(out)


# ---------------------------------------------------------------------------
# Method 3: Gradio Demo — skipped (interactive UI, not automatable)
# ---------------------------------------------------------------------------


@pytest.mark.skip("README section 'Method 4: Using Gradio Demo' is intentionally excluded for example tests")
def test_api_calls_003(): ...


# ---------------------------------------------------------------------------
# Layered Image Generation (Qwen-Image-Layered)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omni_server", qwen_image_layered_server_params, indirect=True)
def test_layered_001(omni_server: OmniServer, example_output_dir: Path, input_image: Path):
    """openai_chat_client.py: layered image generation (Qwen-Image-Layered).

    The model returns multiple RGBA layer images; the client extracts the first
    layer, which is asserted to be a valid image.
    """
    case_dir = example_output_dir / "layered-001"
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "layered_001.png"

    run_cmd(
        [
            sys.executable,
            str(I2I_ONLINE_CLIENT),
            "--input",
            str(input_image),
            "--prompt",
            "a rabbit",
            "--output",
            str(out),
            "--server",
            f"http://{omni_server.host}:{omni_server.port}",
            "--steps",
            "50",
            "--seed",
            "0",
        ]
    )
    assert_image_valid(out)
