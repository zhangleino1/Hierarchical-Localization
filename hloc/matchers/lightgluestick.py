from pathlib import Path
import sys

from .. import logger
from ..utils.base_model import BaseModel


class LightGlueStick(BaseModel):
    """Point-line matcher wrapper.

    This wrapper keeps compatibility with HLoc's matcher interface:
    it always returns point matches (`matches0`) and optionally returns
    line matches (`line_matches0`) if provided by the backend model.
    """

    default_conf = {
        "features": "superpoint",
        "line_features": "lsd",
        "depth_confidence": 0.95,
        "width_confidence": 0.99,
        "compile": False,
    }
    required_inputs = [
        "image0",
        "keypoints0",
        "descriptors0",
        "image1",
        "keypoints1",
        "descriptors1",
    ]

    def _init(self, conf):
        self._use_fallback = False
        model_conf = conf.copy()

        # Prefer a local git submodule if present.
        submodule_root = Path(__file__).resolve().parents[2] / "third_party" / "LightGlueStick"
        if submodule_root.exists():
            sys.path.append(str(submodule_root))

        try:
            from lightglue_stick import LightGlueStick as LightGlueStick_  # type: ignore

            self.net = LightGlueStick_(model_conf)
            if model_conf.get("compile", False) and hasattr(self.net, "compile"):
                self.net.compile()
        except (ImportError, ModuleNotFoundError):
            # Fallback to standard LightGlue point matching to keep the pipeline runnable.
            from lightglue import LightGlue as LightGlue_  # type: ignore

            fallback_conf = model_conf.copy()
            features = fallback_conf.pop("features")
            fallback_conf.pop("line_features", None)
            self.net = LightGlue_(features, **fallback_conf)
            self._use_fallback = True
            logger.warning(
                "LightGlueStick is unavailable, falling back to point-only LightGlue."
            )

    def _forward(self, data):
        data = data.copy()
        data["descriptors0"] = data["descriptors0"].transpose(-1, -2)
        data["descriptors1"] = data["descriptors1"].transpose(-1, -2)

        if self._use_fallback:
            return self.net(
                {
                    "image0": {k[:-1]: v for k, v in data.items() if k.endswith("0")},
                    "image1": {k[:-1]: v for k, v in data.items() if k.endswith("1")},
                }
            )

        # Forward optional line features if available.
        model_input = {
            "image0": {k[:-1]: v for k, v in data.items() if k.endswith("0")},
            "image1": {k[:-1]: v for k, v in data.items() if k.endswith("1")},
        }
        return self.net(model_input)
