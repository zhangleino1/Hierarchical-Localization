from pathlib import Path
import sys

from ..utils.base_model import BaseModel


class Wireframe(BaseModel):
    default_conf = {
        "max_num_keypoints": 2048,
        "max_num_lines": 300,
        "force_num_keypoints": False,
        "force_num_lines": False,
        "nms_radius": 3,
        "detection_threshold": 0.005,
        "line_min_length": 15,
        "merge_points": True,
        "merge_line_endpoints": True,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        submodule_root = (
            Path(__file__).resolve().parents[2] / "third_party" / "LightGlueStick"
        )
        if submodule_root.exists() and str(submodule_root) not in sys.path:
            sys.path.insert(0, str(submodule_root))

        try:
            from lightgluestick.wireframe import WireframeExtractor  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Failed to import LightGlueStick wireframe extractor. "
                "Install LightGlueStick and its dependencies in the same Python "
                f"environment used for HLoc. Original error: {exc}"
            ) from exc

        extractor_conf = {
            "name": "wireframe",
            "trainable": False,
            "point_extractor": {
                "name": "superpoint",
                "trainable": False,
                "dense_outputs": True,
                "max_num_keypoints": conf["max_num_keypoints"],
                "force_num_keypoints": conf["force_num_keypoints"],
                "nms_radius": conf["nms_radius"],
                "detection_threshold": conf["detection_threshold"],
                "legacy_sampling": False,
            },
            "line_extractor": {
                "name": "lsd",
                "trainable": False,
                "max_num_lines": conf["max_num_lines"],
                "force_num_lines": conf["force_num_lines"],
                "min_length": conf["line_min_length"],
            },
            "wireframe_params": {
                "merge_points": conf["merge_points"],
                "merge_line_endpoints": conf["merge_line_endpoints"],
                "nms_radius": conf["nms_radius"],
            },
        }
        self.model = WireframeExtractor(extractor_conf)

    def _forward(self, data):
        pred = self.model(data)
        return {
            "keypoints": pred["keypoints"],
            "keypoint_scores": pred["keypoint_scores"],
            # HLoc stores local descriptors as [D, N] per image in HDF5.
            "descriptors": pred["descriptors"].transpose(-1, -2),
            "lines": pred["lines"],
            "line_scores": pred["line_scores"],
            "lines_junc_idx": pred["lines_junc_idx"],
        }
