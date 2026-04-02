from pathlib import Path
import sys

import torch

from .. import logger
from ..utils.base_model import BaseModel


class LightGlueStick(BaseModel):
    """Point-line matcher wrapper for HLoc.

    The upstream LightGlueStick package expects point scores, line segments,
    line endpoint indices, and per-view image metadata. This wrapper uses the
    real LightGlueStick backend when those inputs are available; otherwise it
    falls back to point-only LightGlue.
    """

    default_conf = {
        "features": "superpoint",
        "line_features": "lsd",
        "depth_confidence": 0.95,
        "width_confidence": 0.99,
        "compile": False,
        "allow_fallback": True,
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
        self._fallback_reason = None
        model_conf = conf.copy()

        submodule_root = (
            Path(__file__).resolve().parents[2] / "third_party" / "LightGlueStick"
        )
        if submodule_root.exists() and str(submodule_root) not in sys.path:
            sys.path.insert(0, str(submodule_root))

        try:
            from lightgluestick import LightGlueStick as LightGlueStick_  # type: ignore

            self.net = LightGlueStick_(model_conf)
            if model_conf.get("compile", False) and hasattr(self.net, "compile"):
                self.net.compile()
        except Exception as exc:
            if not model_conf.get("allow_fallback", True):
                raise RuntimeError(
                    "Failed to initialize LightGlueStick and fallback is disabled. "
                    f"Original error: {exc}"
                ) from exc
            self._build_fallback(model_conf, f"backend import failed: {exc}")

    def _build_fallback(self, model_conf, reason):
        try:
            from lightglue import LightGlue as LightGlue_  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "LightGlueStick backend is unavailable and point-only fallback "
                f"also failed because lightglue could not be imported: {exc}"
            ) from exc

        fallback_conf = model_conf.copy()
        features = fallback_conf.pop("features")
        fallback_conf.pop("line_features", None)
        self.net = LightGlue_(features, **fallback_conf)
        self._use_fallback = True
        self._fallback_reason = reason
        logger.warning(
            "LightGlueStick fallback to point-only LightGlue: %s", reason
        )

    @staticmethod
    def _has_line_features(data):
        required = [
            "lines0",
            "lines1",
            "lines_junc_idx0",
            "lines_junc_idx1",
        ]
        if not all(key in data for key in required):
            return False
        return (
            data["lines0"].numel() > 0
            and data["lines1"].numel() > 0
            and data["lines_junc_idx0"].numel() > 0
            and data["lines_junc_idx1"].numel() > 0
        )

    def _prepare_descriptors(self, desc):
        input_dim = self.net.conf.input_dim if hasattr(self.net, "conf") else 256
        if desc.shape[-1] == input_dim:
            return desc
        if desc.shape[-2] == input_dim:
            return desc.transpose(-1, -2)
        raise RuntimeError(
            f"Unexpected descriptor shape {tuple(desc.shape)} for LightGlueStick; "
            f"expected one axis to equal input_dim={input_dim}."
        )

    def _forward(self, data):
        data = data.copy()
        data["descriptors0"] = self._prepare_descriptors(data["descriptors0"])
        data["descriptors1"] = self._prepare_descriptors(data["descriptors1"])

        if self._use_fallback:
            return self.net(
                {
                    "image0": {k[:-1]: v for k, v in data.items() if k.endswith("0")},
                    "image1": {k[:-1]: v for k, v in data.items() if k.endswith("1")},
                }
            )

        if not self._has_line_features(data):
            if not self.conf.get("allow_fallback", True):
                raise RuntimeError(
                    "LightGlueStick requires line features, but the current feature "
                    "file does not contain valid lines/lines_junc_idx entries."
                )
            if self._fallback_reason != "missing line features":
                self._build_fallback(self.conf.copy(), "missing line features")
            return self.net(
                {
                    "image0": {k[:-1]: v for k, v in data.items() if k.endswith("0")},
                    "image1": {k[:-1]: v for k, v in data.items() if k.endswith("1")},
                }
            )

        def build_view(suffix):
            image = data[f"image{suffix}"]
            image_size = (
                data[f"image_size{suffix}"]
                if f"image_size{suffix}" in data
                else torch.tensor(image.shape[-2:][::-1], device=image.device)[None]
            )
            return {"image": image, "image_size": image_size}

        keypoint_scores0 = data.get(
            "keypoint_scores0",
            torch.ones(
                data["keypoints0"].shape[:2],
                device=data["keypoints0"].device,
                dtype=data["keypoints0"].dtype,
            ),
        )
        keypoint_scores1 = data.get(
            "keypoint_scores1",
            torch.ones(
                data["keypoints1"].shape[:2],
                device=data["keypoints1"].device,
                dtype=data["keypoints1"].dtype,
            ),
        )

        model_input = {
            "view0": build_view("0"),
            "view1": build_view("1"),
            "keypoints0": data["keypoints0"],
            "keypoints1": data["keypoints1"],
            "descriptors0": data["descriptors0"],
            "descriptors1": data["descriptors1"],
            "keypoint_scores0": keypoint_scores0,
            "keypoint_scores1": keypoint_scores1,
            "lines0": data["lines0"].view(data["lines0"].shape[0], -1, 2, 2),
            "lines1": data["lines1"].view(data["lines1"].shape[0], -1, 2, 2),
            "lines_junc_idx0": data["lines_junc_idx0"].view(
                data["lines_junc_idx0"].shape[0], -1, 2
            ).long(),
            "lines_junc_idx1": data["lines_junc_idx1"].view(
                data["lines_junc_idx1"].shape[0], -1, 2
            ).long(),
            "line_scores0": data["line_scores0"],
            "line_scores1": data["line_scores1"],
        }
        return self.net(model_input)
