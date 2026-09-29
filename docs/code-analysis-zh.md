# 我们的方案代码分析:CNN-DINOv2 Hybrid

> 代码:[`notebook.ipynb`](../notebook.ipynb)(代码与公开 kernel 逐语句一致;仓库副本仅将 6 处遗留法语注释译为英文并重写首格概述 markdown,无任何逻辑改动)。下文 `:NNN` 行号沿用原分析所依据的纯代码导出版 `notebook_full.py`(未随仓库发布)的行号,与本仓库 notebook 的大致对应:cell[2]≈:1–309,cell[4]≈:310–397;cell[5–8] 为可视化,不在该导出版中
> 成绩:银牌,Public LB 0.321 / Private LB 0.189
> 性质说明:本 notebook 与 350 票公开 kernel(pankajiitr,该 kernel 又演进自 ravaghi 的 467 票 kernel)代码一致,为原样 fork + 权重复用,未做任何逻辑改动(仓库副本仅翻译了 6 处法语注释、重写首格概述)。以下分析即对该谱系代码的完整解读。

---

## 1. 一句话概括

**冻结的 DINOv2-base 视觉 Transformer 提取整图特征 → 一个 3 层 CNN 小解码头逐像素预测"伪造概率图" → 自适应阈值 + 形态学得到 mask → 面积/置信度双门限决定输出 mask 还是 `authentic`。**

纯推理 notebook:不训练,直接加载公开权重 `CNNDINOv2-U52/model_seg_final.pt`。训练侧脚本见 [`train.py`](../train.py)(赛后按本 notebook 组件补写,未在本地执行,诚实声明见其文件头)。

---

## 2. 代码结构与数据流

```
test_images/*.png
      │
      ▼
pipeline_final(pil)                          ← 每张测试图的入口 (notebook_full.py:230)
      │
      ├─ segment_prob_map_with_tta(pil)      ← 3 路 TTA 概率图 (:186)
      │     ├─ resize 到 518×518,归一化 /255
      │     ├─ 原图 + 水平翻转 + 垂直翻转各跑一次 forward_seg,翻转结果翻回后取平均
      │     └─ forward_seg (:156)
      │           ├─ DINOv2 processor 预处理 → encoder 提特征(冻结,768 维)
      │           ├─ 取 patch token(去掉 CLS)reshape 成 768×37×37 特征图
      │           └─ DinoTinyDecoder 逐级上采样 → 518×518 logits → sigmoid 概率图
      │
      ├─ finalize_mask(prob, orig_size)      ← 概率图 → 二值 mask (:225)
      │     └─ enhanced_adaptive_mask (:212)
      │           ├─ Sobel 算梯度,按 α=0.45 与概率图融合(强化边界)
      │           ├─ 高斯模糊 → 自适应阈值 = mean + 0.3·std → 二值化
      │           └─ 形态学:5×5 闭运算(补洞) + 3×3 开运算(去毛刺)
      │
      └─ 双门限判定 (:238)
            ├─ mask 面积 < 200 像素        → "authentic"
            ├─ mask 内平均概率 < 0.22       → "authentic"
            └─ 否则                        → "forged",输出 mask
                  │
                  ▼
            rle_encode() → submission.csv   (:346,列存 run-length 编码)
```

---

## 3. 模型部分详解

### 3.1 骨干:冻结 DINOv2-base(`notebook_full.py:94-96`)

- 本地加载 Kaggle 数据集里的 DINOv2(`facebook/dinov2-base`),`local_files_only=True`
- 全部参数 `requires_grad=False`(:143)——DINOv2 只当特征提取器,不参与任何训练
- 输入 518×518(正好是 patch 14 的 37 倍),输出 37×37×768 的 patch 特征图

### 3.2 解码头 DinoTinyDecoder(`:98-137`)

```
768×37×37 ──Conv3×3→ 384 ─双线性上采样→ 74×74
          ──Conv3×3→ 192 ─双线性上采样→ 148×148
          ──Conv3×3→  96 ─双线性上采样→ 296×296
          ──Conv1×1→   1 ─双线性上采样→ 518×518
```

- 总共 3,484,417 参数(约 348 万),前两个 block 带 Dropout2d(0.1)
- 逐级上采样 + 卷积细化,是"特征图 → 分割图"的轻量做法
- 注意:这是一个**逐像素二分类**头,没有任何"比较两个区域"的机制——模型只能学"伪造区域长什么样",学不到"这两个区域互为复制"

### 3.3 训练过程(不在本 notebook 中)

权重来自作者公开的 Kaggle 数据集(`cnndinov2-pbd/CNNDINOv2-U52`),注释显示还有 U54 版本(LB 0.310,U52 为 0.321)。训练集即比赛训练集:authentic 图 mask 全零,forged 图配 `train_masks/*.npy` 标注,20% 随机划分做验证(`:163-164`)。

---

## 4. 推理与后处理详解

### 4.1 TTA(测试时增强,`:186-210`)

原图、水平翻转、垂直翻转各推理一次,翻转图的结果翻回来,三路概率图取平均。零成本提升稳定性。

### 4.2 自适应阈值(`:212-223`)

这是该谱系最有特色的设计:

1. **Sobel 梯度增强**:对概率图求 x/y 方向梯度,归一化后按 `0.55·prob + 0.45·grad` 融合——伪造区域边界处概率变化剧烈,梯度项强化边界响应;
2. **自适应二值化**:阈值 = `mean(enhanced) + 0.3·std(enhanced)`,每张图不同;
3. **形态学清理**:闭运算连断点补小洞,开运算去孤立噪点。

### 4.3 双门限判定(`:230-240`)

应对本赛指标的命门(authentic 图预测任何 mask 记 0 分):

- `AREA_THR = 200`:mask 总面积太小 → 认为是噪声,判 authentic
- `MEAN_THR = 0.22`:mask 内平均概率太低 → 认为是低置信响应,判 authentic

### 4.4 验证集网格搜索(`:245-309`)

在 20% 验证集上扫描 `MEAN_THR`(0.20→0.29,步长 0.01,AREA 固定 200),按官方 F1 规则(authentic 预测空记 1.0、预测非空记 0)选最优。**注意:评估和调参用的是同一个验证集**——这是 private 掉分的直接原因之一(详见第 6 节)。另:cell[2] 末尾打印的"验证 F1"仅用 `val_forg[:10]`(前 10 张伪造验证图)做快速 sanity check,并非完整验证集评分。

### 4.5 提交生成(`:346-397`)

- `rle_encode`:列优先展平,输出 `[start, length, start, length, ...]` 的 JSON 字符串;空 mask 输出 `"authentic"`
- 与 `sample_submission.csv` 按 `case_id` 对齐,缺失补 `authentic`
- 可视化(cell[5–8]):预测 mask/概率图叠图展示与 authentic 图对照,不影响提交管线

---

## 5. 这套代码做对的事

1. **选对了骨干**:DINOv2 自监督特征对"视觉上不一致"的区域天然敏感,冻结使用零训练成本;
2. **TTA + 集成思想**:三路翻转平均,稳定概率图;
3. **边界感知后处理**:Sobel 梯度融合是很巧的设计,概率图边界通常正是伪造痕迹所在;
4. **有 authentic 防线意识**:双门限 + 网格搜索,方向正确;
5. **工程完整**:可视化验证、F1 评估、提交格式对齐,闭环齐全。

---

## 6. 问题与失分点(复盘)

| 问题 | 位置 | 后果 |
|---|---|---|
| **建模范式错位**:copy-move 是"两处相同"的关系信号,逐像素分割只能依赖伪造视觉痕迹 | 整体架构 | 对处理干净的复制(无羽化/噪声差异)基本失效;resize 到 518 进一步抹掉细节 |
| **调参与评估同集**:grid search 和 F1 评估共用 20% 验证集 | `:245-309` | 阈值过拟合验证集 → public 0.321 / private 0.189 |
| **自适应阈值不可控**:mean+0.3·std 逐图漂移,测试集分布一变行为就偏 | `:219` | 阈值在 private 分布上失准 |
| **绝对面积门限**:200px 对不同尺寸的测试图意义不同 | `:59` | 大图漏检小伪造,小图误杀 |
| **无图内自匹配先验**:没有利用"复制区域互为镜像"这一本质特征 | 整体架构 | 15th 名同谱系仅加 SIFT 自匹配通道 + SAM 精修就拉开差距 |
| **整图缩放**:518×518 处理任意尺寸原图 | `:181` | 高分辨率图细节损失 |
| **训练环节缺失**:直接复用公开权重,未做任何域内微调 | `:56` | 特征未适配赛内数据分布 |

---

## 7. 与金牌方案的最小差距(同谱系改进路线)

按性价比排序的已知改进(全部有公开实现可参考):

1. 阈值改固定值 / 独立 tuning split(修 private 崩盘)
2. 面积门限百分比化(`% × 总像素`,top14 kernel)
3. 滑窗高分辨率推理 + 全局/局部 0.4/0.6 融合(gaurav kernel,0.332)
4. 叠加 SIFT 图内自匹配通道与概率图 0.7/0.3 融合 + SAM 精修(15th ik0zy)
5. 两阶段训练:冻结→解冻尾 12 block 小学习率微调(65th muhammad)

详细对比(`solution-comparison.md`)与通用经验(`kaggle-cv-playbook-general.md`)两份文档未随本仓库发布。
