# MegaLoc + LightGlueStick 在 HLoc 中的改造方案（基于当前项目架构）

## 1. 目标与替换边界

本方案将新增两个可选后端，并保持与现有 HLoc 流水线的兼容：

1. **MegaLoc**：替换配对生成环节（`pairs_from_retrieval`），负责从图库中为每张查询图像检索候选参考图像。
2. **LightGlueStick**：替换特征匹配环节（`match_features`），在候选图像对之间执行点+线联合匹配。

> 说明：本仓库当前已经存在 `hloc/extractors/megaloc.py`，可作为 MegaLoc 的**全局描述子提取入口**。此次重点是将其从“可提特征”扩展到“可稳定用于配对生成与流水线配置”的完整集成。

---

## 2. 当前架构观察（与改造点对应）

### 2.1 配对生成（retrieval）

- 当前 `hloc/pairs_from_retrieval.py` 的核心逻辑：
  - 从 HDF5 读取 `global_descriptor`；
  - 计算查询与图库描述子的相似度矩阵；
  - 按 top-k 输出图像对列表。
- 因此 MegaLoc 接入的关键不是“改算法框架”，而是保证描述子来源、维度、归一化策略与缓存方式可控。

### 2.2 匹配环节（matcher）

- 当前 `hloc/match_features.py` 通过 `dynamic_load(matchers, conf["model"]["name"])` 动态加载匹配器。
- 数据输入由 `FeaturePairsDataset` 从局部特征 HDF5 中读取，默认是点特征字段（如 `keypoints`、`descriptors`、`scores`）+ `image_size`。
- 输出写入 `matches0` 和可选 `matching_scores0`。

因此，LightGlueStick 的改造要点是：

1. 增加**点+线**特征读取与字段约定；
2. 新增 matcher 封装，使其与现有 `match_features.py` 调用接口兼容；
3. 明确“联合匹配结果”如何投影回 HLoc 后续可消费的数据结构（例如先仅输出点匹配用于 PnP，线匹配作为附加产物）。

---

## 3. 子模块（submodule）落地建议

建议放在 `third_party/` 下，统一管理可复现依赖：

```text
third_party/
  MegaLoc/                # git submodule: gmberton/MegaLoc
  LightGlueStick/         # git submodule: aubingazhib/LightGlueStick
```

### 3.1 子模块引入命令

```bash
git submodule add https://github.com/gmberton/MegaLoc third_party/MegaLoc
git submodule add https://github.com/aubingazhib/LightGlueStick third_party/LightGlueStick
git submodule update --init --recursive
```

### 3.2 依赖策略

- 在 `requirements.txt` 中追加“可选依赖分组说明”（建议在文档中说明，而非强绑定到基础安装）。
- 对于 CUDA/编译型依赖（若 LightGlueStick 需要），提供 `doc/INSTALL_OPTIONAL.md` 风格说明，避免污染基础 CPU 安装路径。

---

## 4. MegaLoc 集成方案（替换 pairs_from_retrieval）

### 4.1 已有基础

仓库已有 `hloc/extractors/megaloc.py`，通过 `torch.hub.load("gmberton/MegaLoc", "get_trained_model")` 提供 `global_descriptor` 输出。

### 4.2 建议改造

1. **新增 retrieval 配置别名**（文档/流水线层）
   - 在 pipelines 中新增例如 `retrieval_conf = "megaloc"` 的标准模板。
   - 与已有 `netvlad/openibl/dir` 的用法保持一致，降低迁移成本。

2. **规范描述子后处理**
   - 在提取阶段增加可选 L2 归一化开关（若 MegaLoc 默认行为与当前检索假设不一致）。
   - 在 `pairs_from_retrieval.py` 中保留通用检索逻辑，不做模型特化分支。

3. **支持多库分片检索**
   - 当前 `pairs_from_retrieval.py` 已支持 `db_descriptors` 多文件输入，建议保留并在 MegaLoc 文档中标注“大规模图库实践方式”。

4. **工程化缓存**
   - 输出命名建议：`global-feats-megaloc.h5`。
   - 对固定数据库描述子进行持久化缓存，查询仅增量提取并执行配对。

### 4.3 兼容收益

- 不改变 `pairs_from_retrieval.py` 输入输出协议（输入 descriptor h5，输出 pairs txt）。
- 下游 `match_features`、`localize_sfm` 无需感知检索模型变化。

---

## 5. LightGlueStick 集成方案（替换 match_features）

### 5.1 新增模块建议

```text
hloc/
  matchers/
    lightgluestick.py      # 新增：封装第三方推理与张量协议转换
  extractors/
    lsd.py / sold2.py      # （可选）新增线段提取器包装
```

### 5.2 配置入口建议

在 `hloc/match_features.py` 的 `confs` 中新增：

- `superpoint+lightgluestick`（点：SuperPoint，线：LSD/SOLD2）
- `aliked+lightgluestick`（点：ALIKED，线：LSD/SOLD2）

示例（概念）：

```python
"superpoint+lightgluestick": {
  "output": "matches-superpoint-lightgluestick",
  "model": {
    "name": "lightgluestick",
    "point_features": "superpoint",
    "line_features": "lsd",
    "max_lines": 2048,
  },
}
```

### 5.3 数据协议扩展

当前 `FeaturePairsDataset` 只读取点特征；需扩展支持线段字段，例如：

- `lines`: `N x 4`（x1,y1,x2,y2）
- `line_descriptors`: `N x D`
- `line_scores`: `N`

并映射为双目输入键：

- `lines0`, `line_descriptors0`, `line_scores0`
- `lines1`, `line_descriptors1`, `line_scores1`

### 5.4 匹配输出与落盘策略

为兼容现有 HLoc：

1. **主输出保留**：`matches0`（点匹配索引）
2. **扩展输出新增**：
   - `line_matches0`
   - `line_matching_scores0`

即：不破坏现有消费者，仅让支持线信息的后处理可选读取新字段。

### 5.5 逐步上线策略

- **Stage A（低风险）**：LightGlueStick 仅输出点匹配（线仅参与内部注意力增强，不落盘）。
- **Stage B（增强）**：落盘线匹配，并在几何验证中尝试线约束（可选分支）。
- **Stage C（深入）**：探索点线联合重投影误差与位姿优化。

---

## 6. 具体改造清单（按实施优先级）

### P0（必须）

1. 新增第三方 submodule 路径与初始化文档。
2. 新增 `hloc/matchers/lightgluestick.py`，实现与 `BaseModel` 兼容的 `_init/_forward`。
3. 在 `hloc/match_features.py` 中新增对应 `confs` 项。
4. 扩展 `FeaturePairsDataset` 支持线段字段（向后兼容：字段缺失时可退化为纯点）。
5. 更新至少一个 pipeline 示例（如 Aachen 或 InLoc）展示 `megaloc + lightgluestick` 组合。

### P1（建议）

1. 新增线特征提取器包装（LSD/SOLD2），并定义 HDF5 字段标准。
2. 增加单测/集成检查：
   - 检索输出 pair 数量与覆盖率；
   - 匹配输出键完整性；
   - 缺失线特征时的退化行为。

### P2（优化）

1. 大规模图库分块检索与 Faiss 支持。
2. 点线联合几何验证（RANSAC with line constraints）。

---

## 7. 风险与规避

1. **依赖复杂度风险（LightGlueStick）**
   - 规避：隔离为可选组件，基础 HLoc 功能不受影响。

2. **I/O 体积增大（线特征）**
   - 规避：限制 `max_lines`，采用 `float16` 压缩 descriptor/score。

3. **与既有 H5 协议冲突**
   - 规避：新增字段，不改旧字段语义；读取端做存在性判断。

4. **推理吞吐下降**
   - 规避：分阶段上线，先验证召回收益再启用线输出。

---

## 8. 推荐最小可用路径（MVP）

1. 用 MegaLoc 生成 `global-feats-megaloc.h5`。
2. 用现有 `pairs_from_retrieval.py` 产出 pairs。
3. 接入 `lightgluestick.py`，先走“点匹配主输出 + 线内部增强但不落盘”。
4. 在 Aachen/InLoc 任一 pipeline 给出可复现脚本。

这样可以在**不改动 HLoc 主干接口**的前提下，先完成“可运行 + 可对比”的替换。

---

## 9. 与现有代码的对齐结论

- MegaLoc：项目已有 extractor 入口，适合直接接入到现有 `extract_features -> pairs_from_retrieval` 链路。
- LightGlueStick：建议以 matcher 插件方式接入 `match_features`，通过扩展数据字段支持点线联合。
- 两者都可通过 submodule 管理，并保持默认路径不受影响。

