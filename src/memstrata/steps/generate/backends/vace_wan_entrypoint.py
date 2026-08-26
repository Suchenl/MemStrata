"""Launch VACE Wan inference without importing optional annotators eagerly."""

from __future__ import annotations

import argparse
import runpy
import sys
import types
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vace_script", required=True)
    parser.add_argument("--vace_module_root", required=True)
    known, rest = parser.parse_known_args()

    module_root = str(Path(known.vace_module_root).resolve())
    if module_root not in sys.path:
        sys.path.insert(0, module_root)
    _install_plain_prompt_annotator_shim()
    _install_sdpa_attention_fallback()
    sys.argv = [known.vace_script, *rest]
    runpy.run_path(known.vace_script, run_name="__main__")


def _install_plain_prompt_annotator_shim() -> None:
    annotators = types.ModuleType("annotators")
    utils = types.ModuleType("annotators.utils")

    def get_annotator(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError(
            "VACE prompt/preprocess annotators are not installed. Use `use_prompt_extend = \"plain\"` "
            "or install VACE annotator dependencies."
        )

    utils.get_annotator = get_annotator
    annotators.utils = utils
    sys.modules.setdefault("annotators", annotators)
    sys.modules.setdefault("annotators.utils", utils)


def _install_sdpa_attention_fallback() -> None:
    """Use Wan's PyTorch SDPA fallback when flash-attn is unavailable.

    Wan2.1 exposes a fallback in ``wan.modules.attention.attention()``, but the model blocks call
    ``flash_attention()`` directly. Rebinding the module globals avoids a hard dependency on
    flash-attn for smoke tests.
    """

    try:
        from wan.modules import attention as attention_module
        from wan.modules import model as wan_model
    except Exception:
        return
    if attention_module.FLASH_ATTN_2_AVAILABLE or attention_module.FLASH_ATTN_3_AVAILABLE:
        return
    attention_module.flash_attention = attention_module.attention
    wan_model.flash_attention = attention_module.attention


if __name__ == "__main__":
    main()
