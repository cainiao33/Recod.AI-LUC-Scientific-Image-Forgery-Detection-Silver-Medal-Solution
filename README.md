# Recod.AI/LUC — Scientific Image Forgery Detection: Silver Medal Solution

> **Kaggle competition:** [Recod.AI/LUC — Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection) (~1,564 teams, ended Jan 2026)
> **Result:** 🥈 Silver Medal — Public LB 0.321 / Private LB 0.189 · [Final leaderboard](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection/leaderboard)
> **Approach:** Frozen DINOv2 + lightweight CNN decoder for pixel-level forgery segmentation
> **Quick start:** everything in one [`notebook.ipynb`](notebook.ipynb) — training, inference, and submission, end to end
> **Author:** [@web3cainiao](https://www.kaggle.com/web3cainiao) on Kaggle (award on [profile](https://www.kaggle.com/web3cainiao)) · [GitHub @cainiao33](https://github.com/cainiao33) · [Original competition notebook](https://www.kaggle.com/code/web3cainiao/scientific-forensics-dinov2-cnn-ipynb)

[中文摘要见文末](#中文摘要)

---

## 1. The Problem

Biomedical papers contain compound figures (microscopy photos, western blots, charts). A common form of research misconduct is **copy-move forgery**: copying a region within a figure and pasting it elsewhere in the *same* figure to fake replicated experiments. The task: given a figure, output segmentation masks of the duplicated regions, or `authentic` if the figure is clean.

Two properties of this task shaped every design decision:

- **Forgery leaves traces, but barely.** Resize/rotation/feathering/compression during pasting subtly alter noise statistics and edge transitions — invisible to humans, hopefully visible to a well-trained feature extractor.
- **False positives are catastrophic.** The metric (pixel-level F1) scores an authentic image **0.0 if you predict any mask at all**. A conservative pipeline is not optional — it is the strategy.

## 2. Approach Overview

```
figure → resize 518×518
       → DINOv2-base (frozen) patch features 37×37×768
       → DinoTinyDecoder (768→384→192→96→1, progressive upsampling)
       → forgery probability map (3-way flip TTA, averaged)
       → Sobel-gradient enhancement + adaptive threshold (mean + 0.3·std)
       → morphology (close 5×5, open 3×3)
       → dual gates: area ≥ 200 px AND mean prob ≥ 0.22
       → RLE mask  or  "authentic"
```

**Inference-only notebook**: weights are loaded from a pre-trained checkpoint; the encoder is never fine-tuned.

### 2.1 Why frozen DINOv2 as the backbone

Copy-move detection needs sensitivity to *local statistical inconsistency*. DINOv2 is self-supervised on 142M images — it has never seen a forged image, but it has an extremely strong prior of what "natural image statistics" look like. Forged regions stand out in its feature space precisely because they deviate from that prior.

- **No fine-tuning**: only ~hundreds of labeled training figures exist; fine-tuning a 86M-parameter model on that is a guaranteed overfit.
- Input 518×518 = 14×37 patches, giving a clean 37×37 token grid.

### 2.2 Deliberately tiny decoder (~1.7M params)

Three 3×3 conv blocks with progressive bilinear upsampling (37→74→148→296→518), then a 1×1 conv head. With this little training data, **decoder capacity is overfitting capacity**. The head only has to learn one thing: which DINOv2 feature patterns correspond to forged regions.

### 2.3 Test-time augmentation

Original + horizontal flip + vertical flip, predictions flipped back and averaged. Copy-move has no preferred orientation; disagreement across flips is model noise and averaging removes it.

### 2.4 Post-processing — where most of the craft is

- **Sobel gradient enhancement** (`0.55·prob + 0.45·grad`): the *boundary* of a pasted region carries the strongest evidence (the seam), but probability maps are blurry exactly at boundaries. Fusing the gradient magnitude back into the probability map sharpens seam response.
- **Adaptive threshold** `mean + 0.3·std` per image, meant to handle brightness/contrast variability across figures.
- **Morphology**: closing fills gaps, opening removes speckles.
- **Dual gates against false positives**: area < 200 px → authentic; mean probability inside mask < 0.22 → authentic. The area gate kills speckle noise; the confidence gate kills large but wishy-washy responses.

## 3. What Worked and What Didn't (Honest Retrospective)

### Right calls

- Frozen general-purpose foundation model as an anomaly detector — the silver-medal backbone of the solution.
- Conservative, false-positive-first design matching the metric.
- Boundary-aware post-processing (Sobel fusion) — real, measurable gains.

### Wrong calls

1. **The fundamental misjudgment**: copy-move is not "this region looks wrong" — it is "**these two regions are identical**". A cleanly pasted region is genuine image content; there may be *no trace at all*. Gold-medal solutions (5 of top 6) don't look for traces — they **look for repetition**: panel detection → pairwise keypoint matching (SIFT/SuperPoint+LightGlue) → geometric verification (RANSAC/MAGSAC). Observation vs. comparison is a paradigm gap that tuning cannot close.
2. **Threshold tuning on the evaluation set**: the gate thresholds were grid-searched on the same 20% validation split used for scoring. Classic adaptive overfitting → Public 0.321, Private 0.189.
3. **Per-image adaptive threshold** behaves unpredictably under distribution shift (test figures are multi-panel compounds, unlike training singles); a fixed threshold would have been more robust.
4. **Absolute pixel gates** (200 px) mean different things on different image sizes — should be a percentage of image area.
5. **Whole-image resize to 518²** destroys the local detail that forgery traces live in; sliding-window or panel-native resolution inference is strictly better.

## 4. Improvement Roadmap

Ranked by cost/benefit; every item has a working public reference from the competition:

| Priority | Improvement | Expected gain | Reference |
|---|---|---|---|
| P0 | Fixed threshold + separate tuning split (never tune on the eval set) | fixes the 0.32→0.19 generalization collapse | 6th place: zero-tuning pipeline (public #398 → private #6) |
| P0 | Area gate as % of image area | size-invariant gating | top-14 kernel "percentage masks" |
| P1 | Sliding-window high-res inference + 0.6 local / 0.4 global fusion | +0.01~0.02, more stable | public kernel (0.332 LB) |
| P1 | SIFT self-matching channel fused with the probability map (0.7/0.3) + SAM mask refinement | +0.02~0.05; injects the true copy-move prior | 15th place (silver), same code lineage |
| P2 | Two-stage training: frozen → unfreeze last blocks at 5e-7 | domain-adapted features | 65th place (silver) repo |
| P2 | Panel detection front-end; abstain on non-compound figures | eliminates the largest FP class | 1st / 4th place |
| P3 | Paradigm shift: detection + pairwise matching as the main line, segmentation as fallback | the gold-medal paradigm | 1st (YOLO+embedding retrieval+LightGlue+lane-level blot matching), 6th (SuperPoint+LightGlue, training-free) |

## 5. Repository Contents

```
notebook.ipynb                  # full solution notebook (data prep, model, TTA, post-processing, submission)
docs/code-analysis-zh.md        # line-by-line code walkthrough (Chinese)
```

Full post-mortem analysis of all top-6 gold solutions and a generalized Kaggle CV engineering playbook were produced alongside this release; links may be added later.

## 6. Usage

The notebook is designed for the Kaggle runtime (GPU, competition dataset + a DINOv2 dataset attached):

1. Attach datasets: competition data, `dinov2/pytorch/base/1`, and the checkpoint dataset (`cnndinov2-pbd`).
2. Run all cells — it performs validation grid search, scores the validation split, runs TTA inference on the test set, and writes `submission.csv`.

## 7. Acknowledgments

This notebook builds on the public kernel lineage by Mahdi Ravaghi and Pankaj Gupta (the `DinoTinyDecoder` family). The retrospective draws on the published solutions of vlad3996 (1st), shiba-inu (2nd), CoreyJamesLevinson (3rd), Nivratti (4th), Guanshuo Xu (5th), Pavel Kazlou (6th), ik0zy (15th) and others — thanks to all of them for open-sourcing.

## License

MIT (code). Competition data remains subject to the competition's terms.

---

## 中文摘要

**赛题**:检测生物医学论文插图中的 copy-move 伪造(同图内复制粘贴区域),输出分割 mask 或 `authentic`。约 1564 支队伍,获**银牌**(Public 0.321 / Private 0.189)。[获奖记录](https://www.kaggle.com/web3cainiao) · [最终榜单](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection/leaderboard)

**思路**:冻结的 DINOv2-base 当"通用异常探测器"(它太熟悉正常图像的统计规律,伪造区域会在特征空间凸显),接一个约 170 万参数的 3 层 CNN 小解码头(数据少,头的容量就是过拟合的容量)逐像素预测可疑度;推理用三路翻转 TTA;后处理是核心——Sobel 梯度增强(边界即拼接缝,证据最集中)、mean+0.3·std 自适应阈值、形态学清理,最后面积+置信度双门限防误报(本赛 authentic 图误报直接零分)。

**教训**:① copy-move 的本质是"两处相同"而非"有破绽",金牌方案都在"找重复"(匹配+几何验证),我们在"找破绽"——范式差距;② 阈值在评估用验证集上网格搜索导致 private 崩盘;③ 自适应阈值与绝对面积门限在分布漂移下不稳。

**改进路线**(全部有公开参考):固定阈值+独立调参集 → 面积门限百分比化 → 滑窗高分辨率推理 → 加 SIFT 自匹配通道+SAM 精修 → 长期转向"检测+匹配"范式。详见第 4 节。
