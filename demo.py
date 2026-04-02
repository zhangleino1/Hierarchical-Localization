"""
简化版室内定位Demo - 基于层次化定位(Hierarchical Localization)

功能：
- 从图像构建3D地图（mapping）
- 定位查询图像（localization）
- 支持仅定位模式（使用已有3D地图）

使用示例：
    # 完整流程（建图+定位）
    python demo.py --images datasets/sacre_coeur --outputs outputs/demo

    python demo.py --images  datasets/chuan/session_1765241559057 --outputs outputs/chuan1
    python demo.py --images  datasets/chuan/session_1765242012556 --outputs outputs/chuan2
    python demo.py --images  datasets/chuan/20260413 --outputs outputs/qi3
    python demo.py --images  datasets/session_1768869109532 --outputs outputs/chuan10
    python demo.py --images  datasets/20260324 --outputs outputs/dalian

    # 仅定位模式（使用已有地图）
    python demo.py --images datasets/chuan/session_1766231113508 --outputs outputs/chuan4 --localization-only
    python demo.py --images datasets/chuan/session_1766231113508 --outputs outputs/chuan4 --localization-only
    python demo.py --images data/session_1768869253012 --outputs outputs/chuan1 --localization-only

    python demo.py --images  data/session_1772111822378 --outputs outputs/chuan12 --localization-only

    python demo.py --images datasets/20260324 --outputs outputs/dalian --seq-window 20 --retrieval-topk 10

    python demo.py --images datasets/chuan/session_1768319515070 --outputs outputs/chuan7 --seq-window 5 --retrieval-topk 40
"""

import argparse
import copy
import logging
import re
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    pairs_from_exhaustive,
    pairs_from_retrieval,
)
from hloc import visualization
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils.io import list_h5_names
from hloc.utils import viz, viz_3d

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


CAMERA_COLORS = [
    ("rgba(0,200,83,0.9)", "rgba(129,199,132,0.9)"),
    ("rgba(3,155,229,0.9)", "rgba(128,222,234,0.9)"),
    ("rgba(255,143,0,0.9)", "rgba(255,204,128,0.9)"),
    ("rgba(233,30,99,0.9)", "rgba(244,143,177,0.9)"),
]

LOCAL_FEATURE_CONF = extract_features.confs["wireframe-superpoint"]
MATCHER_CONF = copy.deepcopy(match_features.confs["superpoint+lightgluestick"])
MATCHER_CONF["model"]["allow_fallback"] = False
RETRIEVAL_CONF = extract_features.confs["megaloc"]

LOCAL_FEATURES_FILE = "features.h5"
GLOBAL_FEATURES_FILE = "global-feats-megaloc.h5"
SFM_MATCHES_FILE = "matches-sfm.h5"
LOC_MATCHES_FILE = "matches-loc.h5"

# ============================================================================
# 控制点配置 - 用于将SfM坐标转换为室内绝对坐标
# ============================================================================
# 格式: {"图像文件名": [x, y, z]}  单位:米
# 至少需要4个非共面的控制点,建议在房间四角+不同高度处拍摄
# 示例: 一个10m x 8m x 3m的房间
CONTROL_POINTS = {
    "mapping/img_001.jpg": [0.0, 0.0, 0.0],      # 房间西南角,地面
    "mapping/img_010.jpg": [10.0, 0.0, 0.0],     # 房间东南角,地面
    "mapping/img_020.jpg": [10.0, 8.0, 0.0],     # 房间东北角,地面
    "mapping/img_030.jpg": [0.0, 8.0, 2.5],      # 房间西北角,高处(如梯子上)
}

# 是否启用控制点标定(如果为False,则输出原始SfM相对坐标)
ENABLE_CONTROL_POINTS = True


# ============================================================================
# 辅助函数
# ============================================================================

def check_reconstruction_exists(sfm_dir):
    """检查3D重建是否存在（兼容新旧版本pycolmap）"""
    # 直接在sfm_dir根目录: sfm_dir/cameras.bin (hloc reconstruction.main 默认输出格式)
    if (sfm_dir / "cameras.bin").exists():
        return True
    # 旧版本格式: sfm_dir/0/cameras.bin
    if (sfm_dir / "0" / "cameras.bin").exists():
        return True
    # 新版本格式: sfm_dir/models/*/cameras.bin
    models_dir = sfm_dir / "models"
    if models_dir.exists():
        for subdir in models_dir.iterdir():
            if subdir.is_dir() and (subdir / "cameras.bin").exists():
                return True
    return False


def align_to_world_coordinates(model, control_points):
    """
    将SfM相对坐标系对齐到室内绝对坐标系
    
    原理: 使用Umeyama算法计算相似变换(7自由度: 3D旋转 + 3D平移 + 尺度)
    
    参数:
        model: pycolmap重建模型
        control_points: 控制点字典 {图像名: [x,y,z]绝对坐标}
    
    返回:
        transform_matrix: 4x4变换矩阵 (齐次坐标)
        scale: 尺度因子
    """
    # 提取对应点对
    sfm_points = []  # SfM坐标系中的相机位置
    world_points = []  # 实际室内坐标
    
    for img_name, world_pos in control_points.items():
        # 查找图像(支持相对路径)
        found = False
        for img in model.images.values():
            if img.name == img_name or img.name.endswith(img_name):
                # 获取相机中心在SfM坐标系中的位置
                sfm_pos = -img.cam_from_world.rotation.matrix().T @ img.cam_from_world.translation
                sfm_points.append(sfm_pos)
                world_points.append(world_pos)
                found = True
                logger.info(f"  控制点: {img_name} -> {world_pos}")
                break
        
        if not found:
            logger.warning(f"  警告: 控制点图像未找到: {img_name}")
    
    if len(sfm_points) < 4:
        raise ValueError(f"控制点不足! 需要至少4个,当前只有{len(sfm_points)}个")
    
    sfm_points = np.array(sfm_points)  # (N, 3)
    world_points = np.array(world_points)  # (N, 3)
    
    # Umeyama算法: 计算相似变换
    # 1. 计算质心
    sfm_center = sfm_points.mean(axis=0)
    world_center = world_points.mean(axis=0)
    
    # 2. 去中心化
    sfm_centered = sfm_points - sfm_center
    world_centered = world_points - world_center
    
    # 3. 计算尺度
    sfm_scale = np.sqrt((sfm_centered ** 2).sum() / len(sfm_points))
    world_scale = np.sqrt((world_centered ** 2).sum() / len(world_points))
    scale = world_scale / sfm_scale
    
    # 4. 计算旋转矩阵(使用SVD)
    H = sfm_centered.T @ world_centered  # 3x3
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 确保是右手坐标系
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # 5. 计算平移
    t = world_center - scale * R @ sfm_center
    
    # 6. 构建4x4齐次变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = scale * R
    transform[:3, 3] = t
    
    # 计算对齐误差
    transformed = (scale * R @ sfm_points.T).T + t
    errors = np.linalg.norm(transformed - world_points, axis=1)
    logger.info(f"  对齐误差: 平均={errors.mean():.3f}m, 最大={errors.max():.3f}m")
    logger.info(f"  尺度因子: {scale:.3f}")
    
    return transform, scale


def transform_pose_to_world(pose, transform_matrix):
    """
    将SfM位姿转换到世界坐标系
    
    参数:
        pose: pycolmap.Rigid3d 对象 (cam_from_world)
        transform_matrix: 4x4变换矩阵
    
    返回:
        world_position: 相机在世界坐标系中的位置 [x, y, z]
    """
    # 获取相机中心在SfM坐标系中的位置
    R_sfm = pose.rotation.matrix()
    t_sfm = pose.translation
    camera_center_sfm = -R_sfm.T @ t_sfm
    
    # 转换到世界坐标系(齐次坐标)
    camera_center_homo = np.append(camera_center_sfm, 1)
    world_position = transform_matrix @ camera_center_homo
    
    return world_position[:3]


def get_image_list(image_dir, subdir):
    """获取指定子目录下的所有图像文件列表"""
    img_path = image_dir / subdir
    if not img_path.exists():
        return []

    images = [
        p.relative_to(image_dir).as_posix()
        for p in img_path.iterdir()
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ]
    return sorted(images, key=natural_key)


def natural_key(path):
    parts = re.split(r'(\d+)', path)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def enable_detailed_file_logging(log_path):
    """Attach a file logger to both the demo logger and the hloc logger."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    logger_targets = [logging.getLogger(), logging.getLogger("hloc")]

    for target in logger_targets:
        has_same_file = any(
            isinstance(handler, logging.FileHandler)
            and Path(getattr(handler, "baseFilename", "")).resolve() == log_path.resolve()
            for handler in target.handlers
        )
        if has_same_file:
            continue

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        target.addHandler(file_handler)

    logger.info("详细日志文件: %s", log_path)


def log_image_list_summary(title, images, preview=5):
    logger.info("%s: 共 %d 张图像", title, len(images))
    if not images:
        return

    head = ", ".join(images[:preview])
    tail = ", ".join(images[-preview:]) if len(images) > preview else head
    logger.info("  前 %d 张: %s", min(preview, len(images)), head)
    if len(images) > preview:
        logger.info("  后 %d 张: %s", min(preview, len(images)), tail)


def read_pairs_file(path):
    pairs = []
    if not path.exists():
        return pairs

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def write_pairs_file(path, pairs):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in sorted(pairs):
            f.write(f"{a} {b}\n")


def group_pairs_by_query(path):
    grouped = {}
    for query_name, ref_name in read_pairs_file(path):
        grouped.setdefault(query_name, []).append(ref_name)
    return grouped


def log_feature_coverage(feature_path, expected_images, label):
    if not feature_path.exists():
        logger.warning("%s 不存在: %s", label, feature_path)
        return

    extracted = set(list_h5_names(feature_path))
    missing = [name for name in expected_images if name not in extracted]
    logger.info(
        "%s 覆盖情况: %d/%d 张图像已写入 %s",
        label,
        len(extracted),
        len(expected_images),
        feature_path,
    )
    if missing:
        logger.warning(
            "%s 缺少 %d 张图像特征，前几个是: %s",
            label,
            len(missing),
            ", ".join(missing[:10]),
        )


def ensure_feature_cache(feature_conf, images_dir, image_list, feature_path, label):
    image_list = sorted(set(image_list), key=natural_key)
    existing = set(list_h5_names(feature_path)) if feature_path.exists() else set()
    missing = [name for name in image_list if name not in existing]

    if missing:
        logger.info(
            "%s: extracting %d missing images into %s",
            label,
            len(missing),
            feature_path,
        )
        extract_features.main(
            feature_conf,
            images_dir,
            image_list=missing,
            feature_path=feature_path,
            overwrite=False,
        )

    log_feature_coverage(feature_path, image_list, label)


def log_pair_coverage(pairs_path, expected_images, label):
    pairs = read_pairs_file(pairs_path)
    if not pairs:
        logger.warning("%s 为空或不存在: %s", label, pairs_path)
        return

    degree = {name: 0 for name in expected_images}
    for name0, name1 in pairs:
        if name0 in degree:
            degree[name0] += 1
        if name1 in degree:
            degree[name1] += 1

    zero_degree = [name for name in expected_images if degree.get(name, 0) == 0]
    low_degree = sorted(
        ((name, count) for name, count in degree.items()),
        key=lambda item: (item[1], natural_key(item[0])),
    )[:10]
    avg_degree = sum(degree.values()) / max(len(expected_images), 1)

    logger.info(
        "%s: 共 %d 对，平均每张图像关联 %.2f 对，文件=%s",
        label,
        len(pairs),
        avg_degree,
        pairs_path,
    )
    logger.info(
        "  关联最少的图像: %s",
        ", ".join(f"{name}({count})" for name, count in low_degree),
    )
    if zero_degree:
        logger.warning(
            "%s 中有 %d 张图像没有任何候选配对，前几个是: %s",
            label,
            len(zero_degree),
            ", ".join(zero_degree[:10]),
        )


def log_sequence_break_hint(all_images, active_images, label):
    if not all_images:
        return

    active_set = set(active_images)
    missing = [name for name in all_images if name not in active_set]
    if not missing:
        logger.info("%s: 所有图像都已覆盖。", label)
        return

    first_missing_index = next(
        index for index, name in enumerate(all_images) if name not in active_set
    )
    previous_name = all_images[first_missing_index - 1] if first_missing_index > 0 else "无"
    current_name = all_images[first_missing_index]
    next_name = (
        all_images[first_missing_index + 1]
        if first_missing_index + 1 < len(all_images)
        else "无"
    )

    logger.warning("%s: 缺失 %d/%d 张图像。", label, len(missing), len(all_images))
    logger.warning(
        "  轨迹可能在第 %d 张附近断开: prev=%s, current=%s, next=%s",
        first_missing_index + 1,
        previous_name,
        current_name,
        next_name,
    )
    logger.warning("  首批未覆盖图像: %s", ", ".join(missing[:10]))


def log_match_file_summary(match_path, label):
    if not match_path.exists():
        logger.warning("%s 不存在: %s", label, match_path)
        return

    with h5py.File(str(match_path), "r", libver="latest") as fd:
        match_groups = list(fd.keys())

    logger.info("%s: 已写入 %d 个配对结果到 %s", label, len(match_groups), match_path)


def log_colmap_log_tail(sfm_dir, num_lines=30):
    log_candidates = sorted(sfm_dir.glob("colmap.LOG*"))
    if not log_candidates:
        logger.warning("未找到 COLMAP 日志文件: %s", sfm_dir)
        return

    latest_log = log_candidates[-1]
    logger.warning("COLMAP 日志文件: %s", latest_log)
    with open(latest_log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    tail_lines = [line.rstrip() for line in lines[-num_lines:] if line.strip()]
    if tail_lines:
        logger.warning("COLMAP 日志尾部 %d 行:", len(tail_lines))
        for line in tail_lines:
            logger.warning("  %s", line)


def find_missing_runs(all_images, active_images):
    active_set = set(active_images)
    missing_runs = []
    run = []
    for name in all_images:
        if name not in active_set:
            run.append(name)
        elif run:
            missing_runs.append(run)
            run = []
    if run:
        missing_runs.append(run)
    return missing_runs


def pick_registered_images_near_breaks(all_images, active_images, max_images=8):
    active_set = set(active_images)
    selected = []
    selected_set = set()
    missing_runs = find_missing_runs(all_images, active_images)

    def add_if_registered(index):
        if 0 <= index < len(all_images):
            name = all_images[index]
            if name in active_set and name not in selected_set:
                selected.append(name)
                selected_set.add(name)

    for run in missing_runs:
        start_idx = all_images.index(run[0])
        end_idx = all_images.index(run[-1])
        for offset in [2, 1, 0]:
            add_if_registered(start_idx - offset - 1)
        for offset in [0, 1, 2]:
            add_if_registered(end_idx + offset + 1)
        if len(selected) >= max_images:
            break

    if len(selected) < max_images:
        for name in active_images:
            if name not in selected_set:
                selected.append(name)
                selected_set.add(name)
            if len(selected) >= max_images:
                break

    return selected[:max_images]


def export_sfm_break_visualizations(
    model,
    images_dir,
    outputs_dir,
    all_images,
    n=8,
):
    registered_names = [img.name for img in model.images.values()]
    missing_runs = find_missing_runs(all_images, registered_names)
    if not missing_runs:
        logger.info("No missing runs detected; skip SfM 2D diagnostics.")
        return

    selected_names = pick_registered_images_near_breaks(
        all_images,
        registered_names,
        max_images=n,
    )
    if not selected_names:
        logger.warning("No registered images available for SfM 2D diagnostics.")
        return

    name_to_id = {img.name: image_id for image_id, img in model.images.items()}
    selected_ids = [name_to_id[name] for name in selected_names if name in name_to_id]
    diag_dir = outputs_dir / "diagnostics" / "sfm_2d"
    diag_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting SfM 2D diagnostics for %d registered images near missing runs: %s",
        len(selected_names),
        ", ".join(selected_names),
    )

    for color_by in ["visibility", "track_length", "depth"]:
        color_dir = diag_dir / color_by
        color_dir.mkdir(parents=True, exist_ok=True)
        for image_name, image_id in zip(selected_names, selected_ids):
            safe_name = image_name.replace("/", "__").replace("\\", "__")
            output_path = color_dir / f"{safe_name}.png"
            try:
                visualization.visualize_sfm_2d(
                    model,
                    images_dir,
                    color_by=color_by,
                    selected=[image_id],
                    n=1,
                )
                viz.save_plot(output_path)
                logger.info("  Saved %s diagnostic: %s", color_by, output_path)
            except Exception:
                logger.exception(
                    "Failed to export SfM %s diagnostic for %s",
                    color_by,
                    image_name,
                )
            finally:
                plt.close("all")


# ============================================================================
# 核心功能：构建3D地图
# ============================================================================

def build_3d_map(
    images_dir,
    outputs_dir,
    mapping_images,
    seq_window,
    retrieval_topk,
    sfm_diag_n,
):
    """
    从mapping图像构建3D地图

    参数:
        images_dir: 图像目录
        outputs_dir: 输出目录
        mapping_images: mapping图像列表（相对路径）

    返回:
        model: pycolmap重建模型
    """
    logger.info("=" * 80)
    logger.info("开始构建3D地图...")
    logger.info(f"  图像目录: {images_dir}")
    logger.info(f"  输出目录: {outputs_dir}")
    logger.info(f"  图像数量: {len(mapping_images)}")
    logger.info("=" * 80)

    # 创建输出目录
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 定义文件路径
    sfm_pairs = outputs_dir / "pairs-sfm.txt"
    sfm_dir = outputs_dir / "sfm"
    features_h5 = outputs_dir / LOCAL_FEATURES_FILE
    matches_h5 = outputs_dir / SFM_MATCHES_FILE
    retrieval_path = outputs_dir / GLOBAL_FEATURES_FILE

    # 步骤1: 提取特征
    mapping_images = sorted(mapping_images, key=natural_key)
    log_image_list_summary("Mapping images", mapping_images)
    logger.info("步骤 1/4: 提取特征点...")
    try:
        extract_features.main(
            LOCAL_FEATURE_CONF,
            images_dir,
            image_list=mapping_images,
            feature_path=features_h5,
            overwrite=True
        )
        log_feature_coverage(features_h5, mapping_images, "Mapping local features")
    except Exception:
        logger.exception("Step 1/4 failed while extracting mapping features.")
        raise

    # 步骤2: 生成图像配对（根据数据集大小选择策略）
    logger.info("步骤 2/4: 生成图像配对...")
    num_images = len(mapping_images)

    if num_images <= 50:
        # 小数据集: 使用穷举匹配确保重建质量
        logger.info(f"  使用穷举匹配策略 ({num_images} 张图像)")
        pairs_from_exhaustive.main(sfm_pairs, image_list=mapping_images)
    else:
        # 大数据集: 使用相邻帧+检索混合配对，提高连通性
        logger.info(f"  使用相邻帧+检索混合配对 ({num_images} 张图像)")

        def build_sequential_pairs(images, window):
            pairs = set()
            for i, name in enumerate(images):
                for j in range(i + 1, min(i + window + 1, len(images))):
                    a, b = name, images[j]
                    pairs.add((a, b) if a < b else (b, a))
            return pairs

        # 2.1 提取全局特征（MegaLoc）
        logger.info("    提取 MegaLoc 全局特征...")
        try:
            extract_features.main(
                RETRIEVAL_CONF,
                images_dir,
                image_list=mapping_images,
                feature_path=retrieval_path,
                overwrite=True
            )
            log_feature_coverage(retrieval_path, mapping_images, "Mapping global features")
        except Exception:
            logger.exception("Step 2/4 failed while extracting retrieval features.")
            raise

        # 2.2 相邻帧配对
        logger.info(f"    生成相邻帧配对 (window={seq_window})...")
        sequential_pairs = build_sequential_pairs(mapping_images, seq_window)

        # 2.3 检索配对
        retrieval_topk = min(max(retrieval_topk, 1), max(num_images - 1, 1))
        logger.info(f"    生成检索配对 (topk={retrieval_topk})...")
        retrieval_pairs_path = outputs_dir / "pairs-retrieval.txt"
        pairs_from_retrieval.main(
            retrieval_path,
            retrieval_pairs_path,
            num_matched=retrieval_topk,
        )
        retrieval_pairs = {
            (a, b) if a < b else (b, a)
            for a, b in read_pairs_file(retrieval_pairs_path)
        }

        # 2.4 合并并写入pairs-sfm.txt
        merged_pairs = sequential_pairs | retrieval_pairs
        write_pairs_file(sfm_pairs, merged_pairs)
        logger.info(f"    生成 {len(merged_pairs)} 个图像配对")

    # 步骤3: 匹配特征
    logger.info("步骤 3/4: 匹配特征...")
    log_pair_coverage(sfm_pairs, mapping_images, "SfM candidate pairs")
    try:
        match_features.main(
            MATCHER_CONF,
            sfm_pairs,
            features=features_h5,
            matches=matches_h5,
            overwrite=True
        )
        log_match_file_summary(matches_h5, "Feature matches")
    except Exception:
        logger.exception("Step 3/4 failed while matching local features.")
        raise

    # 步骤4: 三维重建（Structure-from-Motion）
    logger.info("步骤 4/4: 运行三维重建...")
    logger.info("  COLMAP logs directory: %s", sfm_dir)
    try:
        model = reconstruction.main(
            sfm_dir,
            images_dir,
            sfm_pairs,
            features_h5,
            matches_h5,
            image_list=mapping_images,
            camera_mode=pycolmap.CameraMode.SINGLE,
        )
    except Exception:
        logger.exception("Step 4/4 failed during 3D reconstruction.")
        log_colmap_log_tail(sfm_dir)
        raise

    # 打印重建统计信息
    if model is not None:
        registered_names = [img.name for img in model.images.values()]
        log_sequence_break_hint(mapping_images, registered_names, "Registered mapping images")
        export_sfm_break_visualizations(
            model,
            images_dir,
            outputs_dir,
            mapping_images,
            n=sfm_diag_n,
        )
        logger.info("✓ 重建成功!")
        logger.info(f"  注册图像: {model.num_reg_images()}/{len(mapping_images)}")
        logger.info(f"  3D点数量: {len(model.points3D)}")
        logger.info(f"  平均轨迹长度: {model.compute_mean_track_length():.2f}")
        logger.info(f"  平均重投影误差: {model.compute_mean_reprojection_error():.2f} px")
    else:
        logger.error("✗ 重建失败!")
        raise RuntimeError("三维重建失败")

    if model is None:
        log_colmap_log_tail(sfm_dir)
        raise RuntimeError("3D reconstruction failed")

    return model


# ============================================================================
# 核心功能：定位查询图像
# ============================================================================

def localize_queries(
    images_dir,
    outputs_dir,
    query_images,
    model,
    transform_matrix=None,
    retrieval_topk=20,
):
    """
    在3D地图中定位查询图像

    参数:
        images_dir: 图像目录
        outputs_dir: 输出目录
        query_images: 查询图像列表（相对路径）
        model: pycolmap重建模型

    返回:
        results: 定位结果字典 {图像名: {success, num_inliers, ...}}
    """
    logger.info("=" * 80)
    logger.info("开始定位查询图像...")
    logger.info(f"  查询图像数量: {len(query_images)}")
    logger.info("=" * 80)
    query_images = sorted(query_images, key=natural_key)

    # 定义文件路径
    loc_pairs = outputs_dir / "pairs-loc.txt"
    features_h5 = outputs_dir / LOCAL_FEATURES_FILE
    matches_h5 = outputs_dir / LOC_MATCHES_FILE
    retrieval_h5 = outputs_dir / GLOBAL_FEATURES_FILE

    # 获取已注册的参考图像
    reference_images = sorted((img.name for img in model.images.values()), key=natural_key)
    logger.info(f"  参考图像数量: {len(reference_images)}")

    # 步骤1: 提取查询图像特征
    logger.info("步骤 1/4: 提取查询图像特征...")
    ensure_feature_cache(
        LOCAL_FEATURE_CONF,
        images_dir,
        reference_images,
        features_h5,
        "Reference local features",
    )
    extract_features.main(
        LOCAL_FEATURE_CONF,
        images_dir,
        image_list=query_images,
        feature_path=features_h5,
        overwrite=True
    )
    log_feature_coverage(features_h5, query_images + reference_images, "Localization local features")

    ensure_feature_cache(
        RETRIEVAL_CONF,
        images_dir,
        reference_images,
        retrieval_h5,
        "Reference global features",
    )
    extract_features.main(
        RETRIEVAL_CONF,
        images_dir,
        image_list=query_images,
        feature_path=retrieval_h5,
        overwrite=True,
    )
    log_feature_coverage(
        retrieval_h5,
        query_images + reference_images,
        "Localization global features",
    )

    # 步骤2: 使用 MegaLoc 生成查询-参考配对
    logger.info("步骤 2/4: 使用 MegaLoc 生成查询-参考图像配对...")
    retrieval_topk = min(max(retrieval_topk, 1), len(reference_images))
    pairs_from_retrieval.main(
        retrieval_h5,
        loc_pairs,
        num_matched=retrieval_topk,
        query_list=query_images,
        db_list=reference_images,
        db_descriptors=retrieval_h5,
    )
    query_to_refs = group_pairs_by_query(loc_pairs)
    logger.info(
        "  MegaLoc candidate pairs ready: %d queries, topk=%d, file=%s",
        len(query_to_refs),
        retrieval_topk,
        loc_pairs,
    )

    # 步骤3: 匹配特征
    logger.info("步骤 3/4: 匹配特征...")
    match_features.main(
        MATCHER_CONF,
        loc_pairs,
        features=features_h5,
        matches=matches_h5,
        overwrite=True
    )
    log_match_file_summary(matches_h5, "Localization matches")

    # 步骤4: 估计相机位姿
    logger.info("步骤 4/4: 估计相机位姿...")

    # 定位配置（针对船舱室内场景优化）
    localization_config = {
        "estimation": {
            "ransac": {
                "max_error": 8,  # 收紧阈值（ARCore图像质量较好）
                "min_inlier_ratio": 0.15,  # 提高到15%（过滤低质量匹配）
                "min_num_trials": 500,  # 增加最小迭代次数
                "max_num_trials": 50000,  # 给复杂场景更多机会
                "confidence": 0.9999,
            }
        },
        "refinement": {
            "refine_focal_length": True,
            "refine_extra_params": True,
        },
    }

    # 创建定位器
    localizer = QueryLocalizer(model, localization_config)
    name_to_id = {img.name: img.image_id for img in model.images.values()}

    # 对每个查询图像进行定位
    results = {}
    for query in query_images:
        logger.info(f"\n定位: {query}")

        try:
            # 从EXIF推断相机参数
            camera = pycolmap.infer_camera_from_image(images_dir / query)
            retrieved_refs = query_to_refs.get(query, [])
            ref_ids = [name_to_id[name] for name in retrieved_refs if name in name_to_id]

            if not ref_ids:
                logger.warning("  ✗ 失败: MegaLoc 未返回有效参考图像")
                results[query] = {"success": False, "reason": "no_retrieval_candidates"}
                continue

            logger.info("  候选参考图像: %d", len(ref_ids))

            # 执行定位
            ret, log = pose_from_cluster(
                localizer, query, camera, ref_ids, features_h5, matches_h5
            )

            if ret is not None and ret["num_inliers"] > 0:
                # 定位成功
                num_inliers = ret["num_inliers"]
                num_matches = len(ret["inlier_mask"])
                inlier_ratio = num_inliers / num_matches

                logger.info(f"  ✓ 成功: {num_inliers}/{num_matches} 内点 ({inlier_ratio*100:.1f}%)")

                # 打印位姿信息
                pose = ret["cam_from_world"]
                
                # 如果有坐标转换矩阵,输出绝对坐标
                if transform_matrix is not None:
                    world_pos = transform_pose_to_world(pose, transform_matrix)
                    logger.info(f"    绝对位置 (m): [{world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}]")
                else:
                    # 否则输出相对坐标
                    center = -pose.rotation.matrix().T @ pose.translation
                    logger.info(f"    相对位置 (m): [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

                results[query] = {
                    "success": True,
                    "num_inliers": num_inliers,
                    "num_matches": num_matches,
                    "inlier_ratio": inlier_ratio,
                    "pose": pose,
                    "camera": ret["camera"],
                    "inlier_mask": ret.get("inlier_mask"),
                    "log": log,
                }
            else:
                # 定位失败
                logger.warning(f"  ✗ 失败: 未找到有效位姿")
                results[query] = {"success": False}

        except Exception as e:
            logger.error(f"  ✗ 错误: {e}")
            results[query] = {"success": False, "error": str(e)}

    # 打印统计
    successful = sum(1 for r in results.values() if r.get("success", False))
    logger.info(f"\n定位总结: {successful}/{len(query_images)} 成功")

    return results


def save_localization_visualization(outputs_dir, results, model):
    """将定位结果保存为可交互HTML点云页面。"""
    successful = [(name, res) for name, res in results.items() if res.get("success")]
    if not successful:
        logger.info("无成功定位结果，跳过3D可视化保存。")
        return

    try:
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, model, color="rgba(255,0,0,0.3)", name="mapping", points_rgb=True
        )

        for idx, (query_name, res) in enumerate(successful):
            cam_color, point_color = CAMERA_COLORS[idx % len(CAMERA_COLORS)]
            num_inliers = res.get("num_inliers", 0)
            num_matches = res.get("num_matches", 0)
            inlier_ratio = (num_inliers / num_matches) if num_matches else 0.0
            confidence = inlier_ratio * np.log1p(num_inliers)
            viz_3d.plot_camera_colmap(
                fig,
                res["pose"],
                res["camera"],
                color=cam_color,
                name=query_name,
                fill=True,
                text=(
                    f"inliers: {num_inliers} / {num_matches} "
                    f"confidence: {confidence:.3f}"
                )
            )

            log = res.get("log")
            inlier_mask = res.get("inlier_mask")
            if not log or inlier_mask is None:
                continue
            points_ids = np.array(log.get("points3D_ids", []))
            if len(points_ids) == 0:
                continue
            inlier_mask = np.array(inlier_mask, dtype=bool)
            if len(inlier_mask) != len(points_ids):
                continue
            pts = [
                model.points3D[pid].xyz
                for pid in points_ids[inlier_mask]
                if pid in model.points3D and pid != -1
            ]
            if pts:
                viz_3d.plot_points(
                    fig,
                    np.array(pts),
                    color=point_color,
                    ps=2,
                    name=f"{query_name}_inliers",
                )

        html_path = outputs_dir / "localization_viz.html"
        fig.write_html(str(html_path))
        logger.info("已保存定位结果可视化: %s", html_path)
        
        # Save GLB
        from hloc.utils.export import export_model_to_glb
        glb_path = outputs_dir / "localization.glb"
        export_model_to_glb(model, glb_path, results)
        
    except Exception as exc:
        logger.warning("保存定位可视化失败: %s", exc)


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="室内定位Demo - 基于层次化定位"
    )
    parser.add_argument(
        "--images", type=Path, default=Path("datasets/sacre_coeur"),
        help="图像目录路径"
    )
    parser.add_argument(
        "--outputs", type=Path, default=Path("outputs/demo"),
        help="输出目录路径"
    )
    parser.add_argument(
        "--mapping-dir", type=str, default="seq",
        help="mapping图像子目录名称"
    )
    parser.add_argument(
        "--query-dir", type=str, default="query",
        help="query图像子目录名称"
    )
    parser.add_argument(
        "--seq-window", type=int, default=50,
        help="相邻帧配对窗口"
    )
    parser.add_argument(
        "--retrieval-topk", type=int, default=30,
        help="建图阶段检索配对 topk"
    )
    parser.add_argument(
        "--loc-retrieval-topk", type=int, default=20,
        help="定位阶段 MegaLoc 候选参考图像 topk"
    )
    parser.add_argument(
        "--sfm-diag-n", type=int, default=8,
        help="SfM 2D 可视化导出张数，用于分析断点附近的已注册图像"
    )
    parser.add_argument(
        "--localization-only", action="store_true",
        help="仅定位模式（跳过建图，使用已有3D地图）"
    )

    args = parser.parse_args()

    images_dir = args.images
    outputs_dir = args.outputs
    sfm_dir = outputs_dir / "sfm"
    enable_detailed_file_logging(outputs_dir / "pipeline.log")

    # ========================================================================
    # 模式1: 仅定位模式
    # ========================================================================
    if args.localization_only:
        logger.info("运行模式: 仅定位")

        # 检查是否存在3D重建
        if not check_reconstruction_exists(sfm_dir):
            logger.error(f"未找到3D重建: {sfm_dir}")
            logger.error("请先运行完整流程构建3D地图，或指定正确的输出目录")
            return

        # 加载已有3D重建
        logger.info(f"加载3D重建: {sfm_dir}")
        model = pycolmap.Reconstruction(sfm_dir)
        logger.info(f"  图像数量: {model.num_reg_images()}")
        logger.info(f"  3D点数量: {len(model.points3D)}")

        # 获取查询图像
        query_images = get_image_list(images_dir, args.query_dir)
        if not query_images:
            logger.warning(f"未找到查询图像: {images_dir / args.query_dir}")
            return

        # 坐标对齐(如果启用)
        transform_matrix = None
        if ENABLE_CONTROL_POINTS and CONTROL_POINTS:
            try:
                logger.info("\n应用控制点标定...")
                transform_matrix, scale = align_to_world_coordinates(model, CONTROL_POINTS)
            except Exception as e:
                logger.warning(f"控制点标定失败: {e}")
                logger.warning("将使用原始SfM相对坐标")
        
        # 定位
        results = localize_queries(
            images_dir,
            outputs_dir,
            query_images,
            model,
            transform_matrix,
            retrieval_topk=args.loc_retrieval_topk,
        )
        save_localization_visualization(outputs_dir, results, model)

        logger.info("\n完成!")
        return

    # ========================================================================
    # 模式2: 完整流程（建图+定位）
    # ========================================================================
    logger.info("运行模式: 完整流程")

    # 获取mapping图像
    mapping_images = get_image_list(images_dir, args.mapping_dir)
    if not mapping_images:
        logger.error(f"未找到mapping图像: {images_dir / args.mapping_dir}")
        return

    # 构建3D地图
    model = build_3d_map(
        images_dir,
        outputs_dir,
        mapping_images,
        seq_window=args.seq_window,
        retrieval_topk=args.retrieval_topk,
        sfm_diag_n=args.sfm_diag_n,
    )

    # 获取查询图像
    query_images = get_image_list(images_dir, args.query_dir)
    if not query_images:
        logger.warning(f"未找到查询图像: {images_dir / args.query_dir}")
        logger.info("跳过定位步骤")
        return

    # 坐标对齐(如果启用)
    transform_matrix = None
    if ENABLE_CONTROL_POINTS and CONTROL_POINTS:
        try:
            logger.info("\n应用控制点标定...")
            transform_matrix, scale = align_to_world_coordinates(model, CONTROL_POINTS)
        except Exception as e:
            logger.warning(f"控制点标定失败: {e}")
            logger.warning("将使用原始SfM相对坐标")
    
    # 定位查询图像
    results = localize_queries(
        images_dir,
        outputs_dir,
        query_images,
        model,
        transform_matrix,
        retrieval_topk=args.loc_retrieval_topk,
    )
    save_localization_visualization(outputs_dir, results, model)

    logger.info("\n完成!")


if __name__ == "__main__":
    main()
