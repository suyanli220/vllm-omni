"""
Offline inference tests: image-to-image.
See examples/offline_inference/image_to_image/image_to_image.md
"""

import shutil
from pathlib import Path

import pytest

from tests.conftest import assert_image_valid
from tests.examples.conftest import EXAMPLES, OUTPUT_DIR, ExampleRunner, ReadmeSnippet
from tests.utils import hardware_marks

pytestmark = [pytest.mark.advanced_model, pytest.mark.example, *hardware_marks(res={"cuda": "H100"})]

I2I_SCRIPT = EXAMPLES / "offline_inference" / "image_to_image" / "image_edit.py"
README_PATH = I2I_SCRIPT.with_name("image_to_image.md")
EXAMPLE_OUTPUT_SUBFOLDER = "example_offline_i2i"


def _skip_readme_snippet(language: str, code: str, h2_title: str) -> tuple[bool, str]:
    if language == "bash" and code.strip().startswith("wget"):
        return True, "wget download commands are skipped; the test fixture provides the input image"
    # Qwen-Image-Layered produces N RGBA layer files named <prefix>_0.png, <prefix>_1.png, …
    # The test framework expects a single file at the --output path, so skip this variant.
    if "Qwen-Image-Layered" in code or "--color-format RGBA" in code:
        return True, "Qwen-Image-Layered produces multi-file layered output not supported by the snippet runner"
    return False, ""


README_SNIPPETS = ReadmeSnippet.extract_readme_snippets(README_PATH, skipif=_skip_readme_snippet)


class _I2IExampleRunner(ExampleRunner):
    """ExampleRunner that copies the required input image into each snippet's run directory."""

    def __init__(self, output_root: Path, input_image: Path) -> None:
        super().__init__(output_root)
        self._input_image = input_image

    def run(
        self,
        snippet: ReadmeSnippet,
        *,
        output_subfolder: Path = Path("."),
        env: dict[str, str] | None = None,
    ):
        run_dir = self.output_root / output_subfolder / snippet.test_id
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._input_image, run_dir / self._input_image.name)
        return super().run(snippet, output_subfolder=output_subfolder, env=env)


@pytest.fixture(scope="module")
def input_image() -> Path:
    """Return a locally generated synthetic PNG (514×556 RGB) as the I2I input image.

    Uses PIL instead of a runtime network fetch so CI stays hermetic even on
    runners without outbound internet access or during transient S3 issues.
    The file is named ``qwen-bear.png`` to match the hardcoded filename used in
    README code snippets (``Image.open("qwen-bear.png")`` / ``--image qwen-bear.png``).
    """
    from PIL import Image

    image_path = OUTPUT_DIR / "qwen-bear.png"
    if not image_path.exists():
        Image.new("RGB", (514, 556), color=(128, 160, 200)).save(image_path)
    return image_path


@pytest.fixture
def example_runner(input_image: Path) -> ExampleRunner:
    return _I2IExampleRunner(output_root=OUTPUT_DIR, input_image=input_image)


@pytest.mark.parametrize("snippet", README_SNIPPETS, ids=lambda snippet: snippet.test_id)
def test_image_to_image(snippet: ReadmeSnippet, example_runner: ExampleRunner):
    should_skip, reason = snippet.skip
    if should_skip:
        pytest.skip(reason)

    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_image_valid(asset)
