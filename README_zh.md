# Recod.AI/LUC — 科学图像伪造检测：银牌方案

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/切换语言-English-blue?style=for-the-badge" alt="English"></a>
  <a href="README_zh.md"><img src="https://img.shields.io/badge/切换语言-简体中文-red?style=for-the-badge" alt="简体中文"></a>
</p>

> **Kaggle 比赛**：[Recod.AI/LUC — Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)（约 1,564 支队伍，2026 年 1 月结束）
> **成绩**：🥈 银牌 —— 名次 **72 / 1,564**（前 5%）· Public LB 0.321 / Private LB 0.189 · [最终榜单](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection/leaderboard)
> **方法**：冻结 DINOv2 + 轻量 CNN 解码头，做像素级伪造分割
> **快速上手**：[`notebook.ipynb`](notebook.ipynb) 仅推理（加载 checkpoint → 阈值网格搜索 → TTA 推理 → 生成提交）；训练代码在 [`train.py`](train.py)
> **作者**：Kaggle [@web3cainiao](https://www.kaggle.com/web3cainiao)（获奖见[个人主页](https://www.kaggle.com/web3cainiao)）· [GitHub @cainiao33](https://github.com/cainiao33) · [原始参赛 notebook](https://www.kaggle.com/code/web3cainiao/scientific-forensics-dinov2-cnn-ipynb)

> 🏅 **获奖记录**：[Kaggle 个人主页](https://www.kaggle.com/web3cainiao) · [全部比赛记录](https://www.kaggle.com/web3cainiao/competitions)

![Kaggle 比赛成就——Recod.AI/LUC 银牌（private 0.18958，第 72 / 1,564 名）](assets/kaggle-competitions.png)

---

## 1. 赛题

生物医学论文中存在复合插图（显微照片、western blot、图表）。一种常见学术不端是 **copy-move 伪造**：把图内某一区域复制后粘贴到*同一张图*的其他位置，伪造"重复实验"。任务：给定一张图，输出重复区域的分割 mask；图是干净的则输出 `authentic`。

赛题的两个性质决定了所有设计决策：

- **伪造有痕迹，但很微弱。** 粘贴时的缩放/旋转/羽化/压缩会轻微改变噪声统计与边缘过渡——人眼不可见，但希望对训练良好的特征提取器可见。
- **误报是灾难性的。** 评测指标（像素级 F1）规定：对 authentic 图**只要预测出任何 mask 就得 0.0 分**。保守的管线不是可选项——它本身就是策略。

## 2. 方案总览

```
插图 → resize 518×518
     → DINOv2-base（冻结）patch 特征 37×37×768
     → DinoTinyDecoder（768→384→192→96→1，渐进上采样）
     → 伪造概率图（三路翻转 TTA 取平均）
     → Sobel 梯度增强 + 自适应阈值（mean + 0.3·std）
     → 形态学（闭运算 5×5，开运算 3×3）
     → 双门限：面积 ≥ 200 px 且 平均概率 ≥ 0.22
     → RLE mask  或  "authentic"
```

**仅推理 notebook**：权重从预训练 checkpoint 加载，编码器从不微调。

### 2.1 为什么用冻结 DINOv2 做骨干

copy-move 检测需要对*局部统计不一致*敏感。DINOv2 在 1.42 亿张图上自监督预训练——它从没见过伪造图，但对"自然图像统计"有极强的先验。伪造区域恰恰因为偏离这个先验，才会在它的特征空间中凸显。

- **不微调**：有标注的训练图只有几百张；拿这个去微调 8600 万参数的模型必然过拟合。
- 输入 518×518 = 14×37 个 patch，得到干净的 37×37 token 网格。

### 2.2 刻意做小的解码头（约 348 万参数）

三个 3×3 卷积块 + 渐进双线性上采样（37→74→148→296→518），再接 1×1 卷积头。训练数据这么少时，**解码器的容量就是过拟合的容量**。这个头只需要学一件事：哪些 DINOv2 特征模式对应伪造区域。

### 2.3 测试时增强（TTA）

原图 + 水平翻转 + 垂直翻转，预测翻转回原方向后取平均。copy-move 没有优先朝向；各翻转间的分歧是模型噪声，平均可以消掉。

### 2.4 后处理——功夫所在

- **Sobel 梯度增强**（`0.55·prob + 0.45·grad`）：粘贴区域的*边界*携带最强证据（拼接缝），但概率图恰好在边界处最模糊。把梯度幅值融回概率图，可以锐化缝响应。
- **自适应阈值** `mean + 0.3·std` 逐图计算，用于应对图间亮度/对比度差异。
- **形态学**：闭运算填缝，开运算去噪点。
- **防误报双门限**：面积 < 200 px → authentic；mask 内平均概率 < 0.22 → authentic。面积门限杀散点噪声，置信度门限杀"大而含糊"的响应。

## 3. 做对了什么、做错了什么（诚实复盘）

### 对的

- 冻结的通用基础模型当异常检测器——银牌方案的骨干。
- 保守的、误报优先的设计，与指标对齐。
- 边界感知后处理（Sobel 融合）——真实、可测量的收益。

### 错的

1. **根本性误判**：copy-move 不是"这块区域看起来不对"，而是"**这两块区域相同**"。干净粘贴的区域是真实图像内容，可能*完全没有痕迹*。金牌方案（前 6 中 5 个）不找痕迹——它们**找重复**：面板检测 → 成对关键点匹配（SIFT/SuperPoint+LightGlue）→ 几何验证（RANSAC/MAGSAC）。观察 vs 比较是范式差距，调参补不上。
2. **在评估集上调阈值**：`MEAN_THR` 的网格搜索（0.20–0.29，步长 0.01，`AREA_THR` 固定 200）在与打分同一个 20% 验证集上做。经典自适应过拟合 → Public 0.321，Private 0.189。
3. **逐图自适应阈值**在分布漂移下行为不可预测（测试图是多面板复合图，训练图是单图）；固定阈值反而更稳。
4. **绝对像素门限**（200 px）在不同尺寸图上含义不同——应改为占图面积的比例。
5. **整图 resize 到 518²** 摧毁了伪造痕迹所在的局部细节；滑窗或面板原生分辨率推理严格更优。

## 4. 改进路线

按成本/收益排序；每条都有比赛中的公开参考：

| 优先级 | 改进 | 预期收益 | 参考 |
|---|---|---|---|
| P0 | 固定阈值 + 独立调参集（绝不在评估集上调参） | 修复 0.32→0.19 的泛化崩塌 | 第 6 名：零调参管线（public #398 → private #6） |
| P0 | 面积门限改为占图面积比例 | 尺寸无关的门限 | 前 14 kernel "percentage masks" |
| P1 | 滑窗高分辨率推理 + 0.6 局部 / 0.4 全局融合 | +0.01~0.02，更稳 | 公开 kernel（0.332 LB） |
| P1 | SIFT 自匹配通道与概率图融合（0.7/0.3）+ SAM mask 精修 | +0.02~0.05；注入真正的 copy-move 先验 | 第 15 名（银牌），同代码血统 |
| P2 | 两阶段训练：冻结 → 以 5e-7 解冻末层 | 域适应特征 | 第 65 名（银牌）仓库 |
| P2 | 面板检测前端；非复合图弃权 | 消除最大误报来源 | 第 1 / 4 名 |
| P3 | 范式转移：检测 + 成对匹配为主线，分割兜底 | 金牌范式 | 第 1（YOLO+嵌入检索+LightGlue+泳道级污渍匹配）、第 6（SuperPoint+LightGlue，免训练） |

## 5. 仓库内容

```
notebook.ipynb                  # 推理 notebook（模型、阈值网格搜索、TTA、后处理、提交）
train.py                        # 解码头训练脚本（赛后据 notebook 组件补写；未在本地执行，见文件头说明）
docs/code-analysis-zh.md        # 逐行代码走读（中文）
assets/kaggle-competitions.png  # Kaggle 成就截图（银牌记录，18 场比赛）
LICENSE                         # MIT
```

与本次发布同时产出的还有前 6 金牌方案完整复盘与一份泛化的 Kaggle CV 工程手册，但不包含在本仓库中。

## 6. 用法

两个入口都面向 Kaggle 运行时（挂比赛数据集 + DINOv2 数据集 + checkpoint 数据集）；均未在作者本地机器上运行（本地无 GPU——见 `train.py` 文件头的诚实说明）。代码在有 CUDA 时自动使用；注意本仓库归档的 notebook 运行记录是 **CPU**（Kaggle 元数据 `accelerator: none`，主 cell 约 80 分钟）——想要合理的推理速度请挂 GPU 会话。

**推理 —— `notebook.ipynb`**

1. 挂载数据集：比赛数据、`dinov2/pytorch/base/1`、checkpoint 数据集（`cnndinov2-pbd`）。
2. 全部运行——加载 checkpoint，做验证网格搜索（MEAN_THR 0.20–0.29，AREA_THR 固定 200；`GRID_SEARCH=True` 运行时覆盖打印的常量），对**前 10 张伪造验证图**打分（快速 sanity check，不是完整分割），然后对测试集做 TTA 推理并写出 `submission.csv`。

**训练 —— `train.py`**（赛后补写，从未执行）

```bash
python train.py \
  --data-dir /kaggle/input/recodai-luc-scientific-image-forgery-detection \
  --dino-path /kaggle/input/dinov2/pytorch/base/1 \
  --out-dir /kaggle/working/ckpt \
  --init-checkpoint /kaggle/input/cnndinov2-pbd/CNNDINOv2-U52/CNNDINOv2-U52/model_seg_final.pt
```

只训练解码头（AdamW、BCE+Dice、翻转增广；编码器在 `no_grad` 下保持冻结），每轮按官方 F1 约定验证，保存全模块 `state_dict` checkpoint，可被 notebook 加载。

## 7. 致谢

本 notebook 建立在 Mahdi Ravaghi 与 Pankaj Gupta 的公开 kernel 血统之上（`DinoTinyDecoder` 家族）。复盘参考了 vlad3996（第 1）、shiba-inu（第 2）、CoreyJamesLevinson（第 3）、Nivratti（第 4）、Guanshuo Xu（第 5）、Pavel Kazlou（第 6）、ik0zy（第 15）等人的公开方案——感谢所有开源者。

## AI 协作标注

参赛 notebook 的**代码逻辑**是公开 kernel 的原样 fork（血统声明见 [docs/code-analysis-zh.md](docs/code-analysis-zh.md)）；本仓库副本仅做了装饰性修改——6 处遗留法语注释译为英文、首格概述重写。本复盘与 `train.py` 由 **Claude Code** 协助完成，见提交历史。

## 许可

MIT（代码）。比赛数据仍受比赛条款约束。
