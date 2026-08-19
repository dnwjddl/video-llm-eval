"""lmms-eval entry-point payload for the Gemma 4 wrapper."""

from lmms_eval.models.registry_v2 import ModelManifest

MANIFEST = ModelManifest(
    model_id="gemma4",
    simple_class_path="gemma4_lmms.gemma4.Gemma4",
)
