#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
train.py — DINOv2 冻结骨干 + 小型卷积解码头 的伪造分割训练脚本
================================================================

> ⚠️ **诚实声明:本脚本未在本地执行**(本地无 GPU,且未下载比赛数据集 / DINOv2 权重)。
> 组件逐一取自 `notebook.ipynb` cell[2](数据集 / 模型 / 划分方式),设计为在
> **Kaggle GPU 环境**(T4×2 / P100)或任何装有比赛数据与 DINOv2 权重的环境运行。
> 建议用法:把本仓库 Add Data 为 Notebook 输入,或上传 train.py 后按下方 CLI 示例运行。

补齐 notebook(纯推理)缺失的训练侧环节,共 8 项:

1. **恢复 encoder 前向的 `torch.no_grad()`** —— notebook cell[2] 中该行被注释掉
   (`# with torch.no_grad():`),推理期无碍,但训练期不恢复会导致 encoder 计算
   图被 autograd 追踪,显存占用暴涨(OOM 的直接原因)。本脚本在特征提取内
   恢复 no_grad,梯度只流向 decoder;
2. **损失 = BCEWithLogits + Soft-Dice** —— 伪造前景极稀疏,纯 BCE 会被背景
   主导,Dice 项直接优化前景重叠;
3. **AdamW 只优化 `requires_grad` 参数** —— 即仅 decoder(3,484,417 参数 ≈ 3.48M);
4. **epoch 循环 + 梯度裁剪(1.0) + 每 epoch 训练指标打印**;
5. **epoch 末按官方 F1 规则做验证集选优**(authentic 预测空 mask 记 1.0、预测
   非空记 0.0,forged 用 zero_division=0 —— 与 notebook grid_search 的评分
   完全一致),最优 checkpoint 存 `model_seg_best.pt`;
6. **保存整模块 state_dict**(键名 `encoder.*` + `seg_head.*`),与 notebook 的
   `model_seg.load_state_dict(...)` 直接兼容;
7. **水平 / 垂直翻转增广**(p=0.5 各自独立,图像与 mask 同步翻转)—— 镜像
   推理期 3 路 TTA 的对称性;
8. **数值路径与推理严格一致**:`(x*255).clamp(0,255).byte()` → DINOv2 processor
   归一化 → encoder,不做任何训练期捷径。

CLI(Kaggle 默认值):

    python train.py \
        --data-dir /kaggle/input/recodai-luc-scientific-image-forgery-detection \
        --dino-path /kaggle/input/dinov2/pytorch/base/1 \
        --out-dir /kaggle/working/ckpt \
        --epochs 3 --batch-size 2 --lr 2e-4

    # 从公开权重 CNNDINOv2-U52 热启动(即 notebook 加载的那份):
    python train.py --init-checkpoint \
        /kaggle/input/cnndinov2-pbd/CNNDINOv2-U52/CNNDINOv2-U52/model_seg_final.pt ...

Author: cainiao33(方案与 notebook 组件);脚本由 Claude Code 依据 notebook.ipynb
cell[2] 组件编写,未在本地执行(见顶部声明)。
"""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SIZE = 518

# 推理侧最终生效的门限(notebook 网格搜索产物:仅扫 MEAN_THR 0.20-0.29,AREA 固定 200)
AREA_THR = 200
MEAN_THR = 0.22


def seed_everything(seed=42):
    """与 notebook cell[2] 完全一致的固定种子函数 / Same seeding as notebook cell[2]."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 数据集 / Dataset
# ---------------------------------------------------------------------------

class ForgerySegTrainDataset(torch.utils.data.Dataset):
    """训练集:与 notebook 的 ForgerySegDataset 相同的读取/缩放逻辑,外加翻转增广。

    / Training set: same load+resize logic as notebook ForgerySegDataset,
    / plus horizontal / vertical flip augmentation applied to image AND mask jointly.
    """

    def __init__(self, auth_paths, forg_paths, mask_dir, img_size=IMG_SIZE,
                 flip_aug=True):
        self.samples = []
        for p in forg_paths:
            m = os.path.join(mask_dir, Path(p).stem + ".npy")
            if os.path.exists(m):
                self.samples.append((p, m))
        for p in auth_paths:
            self.samples.append((p, None))
        self.img_size = img_size
        self.flip_aug = flip_aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2  # 惰性导入,保持本模块可被无 cv2 环境导入 / lazy import
        from PIL import Image

        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if mask_path is None:
            mask = np.zeros((h, w), np.uint8)
        else:
            m = np.load(mask_path)
            if m.ndim == 3:
                m = np.max(m, axis=0)
            mask = (m > 0).astype(np.uint8)
        img_r = img.resize((self.img_size, self.img_size))
        mask_r = cv2.resize(mask, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_NEAREST)
        if self.flip_aug:
            if random.random() < 0.5:  # 水平翻转 / horizontal flip
                img_r = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                mask_r = mask_r[:, ::-1]
            if random.random() < 0.5:  # 垂直翻转 / vertical flip
                img_r = img_r.transpose(Image.FLIP_TOP_BOTTOM)
                mask_r = mask_r[::-1, :]
        img_t = torch.from_numpy(np.array(img_r, np.float32) / 255.).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_r)[None, ...].astype(np.float32))
        return img_t, mask_t


# ---------------------------------------------------------------------------
# 模型 / Model(结构与 notebook cell[2] 逐层一致)
# ---------------------------------------------------------------------------

class DinoTinyDecoder(nn.Module):
    """768×37×37 → 1×518×518 的轻量解码头(3,484,417 参数)。结构与 notebook 一致。"""

    def __init__(self, in_ch=768, out_ch=1):
        super().__init__()
        # Block 1: 768 -> 384
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        # Block 2: 384 -> 192
        self.block2 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        # Block 3: 192 -> 96
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Final Output: 96 -> 1
        self.conv_out = nn.Conv2d(96, out_ch, kernel_size=1)

    def forward(self, f, target_size):
        x = F.interpolate(self.block1(f), size=(74, 74), mode="bilinear", align_corners=False)
        x = F.interpolate(self.block2(x), size=(148, 148), mode="bilinear", align_corners=False)
        x = F.interpolate(self.block3(x), size=(296, 296), mode="bilinear", align_corners=False)
        x = self.conv_out(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class TrainableDinoSegmenter(nn.Module):
    """DinoSegmenter 的可训练版:属性名(encoder/processor/seg_head)与 notebook 一致,
    因此 state_dict 键名(`encoder.*` + `seg_head.*`)与 notebook 的加载完全兼容。

    与 notebook 的**唯一结构差异**:特征提取内恢复了 `torch.no_grad()`(notebook
    cell[2] 中被注释掉的那行)——encoder 冻结不训练,不需要 autograd 图,训练期
    不恢复会显存爆炸。
    """

    def __init__(self, encoder, processor):
        super().__init__()
        self.encoder, self.processor = encoder, processor
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.seg_head = DinoTinyDecoder(768, 1)

    def forward_features(self, x):
        # 数值路径与推理严格一致:(x*255).clamp().byte() -> processor 归一化
        imgs = (x * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        inputs = self.processor(images=list(imgs), return_tensors="pt").to(x.device)
        with torch.no_grad():  # ← 训练脚本恢复的关键行(notebook 中被注释)
            feats = self.encoder(**inputs).last_hidden_state
        B, N, C = feats.shape
        fmap = feats[:, 1:, :].permute(0, 2, 1)
        s = int(math.sqrt(N - 1))
        fmap = fmap.reshape(B, C, s, s)
        return fmap

    def forward_seg(self, x):
        fmap = self.forward_features(x)
        return self.seg_head(fmap, (IMG_SIZE, IMG_SIZE))


# ---------------------------------------------------------------------------
# 损失 / Loss(BCE + Soft-Dice,应对稀疏前景)
# ---------------------------------------------------------------------------

def combined_seg_loss(logits, targets, dice_weight=1.0, eps=1e-6):
    """BCEWithLogits + Soft-Dice。

    / BCE alone is dominated by the sparse background; the Dice term directly
    / optimizes foreground overlap (mirrors the competition metric).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    num = (probs * targets).sum(dim=(1, 2, 3))
    den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
    dice = 1.0 - (2.0 * num / den).mean()
    return bce + dice_weight * dice


# ---------------------------------------------------------------------------
# 验证:官方 F1 规则(authentic 空预测=1.0 / 非空=0.0),与 notebook grid_search 一致
# ---------------------------------------------------------------------------

@torch.no_grad()
def _prob_map(model, pil, device, tta=True):
    from PIL import Image as _PILImage  # noqa: F401 (文档性引用)
    x = torch.from_numpy(
        np.array(pil.resize((IMG_SIZE, IMG_SIZE)), np.float32) / 255.
    ).permute(2, 0, 1)[None].to(device)
    preds = [torch.sigmoid(model.forward_seg(x))]
    if tta:
        for dim in (3, 2):  # 水平翻转 / 垂直翻转,与 notebook 3 路 TTA 一致
            p = torch.sigmoid(model.forward_seg(torch.flip(x, dims=[dim])))
            preds.append(torch.flip(p, dims=[dim]))
    return torch.stack(preds).mean(0)[0, 0].cpu().numpy()


def _enhanced_adaptive_mask(prob, alpha_grad=0.45):
    """与 notebook 的 enhanced_adaptive_mask 完全一致(Sobel 融合 + 自适应阈值 + 形态学)。"""
    import cv2

    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_norm = grad_mag / (grad_mag.max() + 1e-6)
    enhanced = (1 - alpha_grad) * prob + alpha_grad * grad_norm
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    thr = np.mean(enhanced) + 0.3 * np.std(enhanced)
    mask = (enhanced > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask, thr


@torch.no_grad()
def evaluate_official_f1(model, forg_paths, auth_paths, mask_dir, device,
                         tta=True, max_images=0, verbose=True):
    """逐图官方 F1:forged 用 zero_division=0,authentic 空 mask 记 1.0。

    / Per-image official F1 with the same zero_division convention as the
    / notebook grid_search (authentic silence = 1.0, any prediction = 0.0).
    """
    import cv2
    from PIL import Image
    from sklearn.metrics import f1_score

    model.eval()
    items = [(p, "forged") for p in forg_paths] + [(p, "authentic") for p in auth_paths]
    if max_images and max_images < len(items):
        items = items[:max_images]
    f1s = []
    for p, label in items:
        pil = Image.open(p).convert("RGB")
        w, h = pil.size
        prob = _prob_map(model, pil, device, tta=tta)
        mask_raw, _ = _enhanced_adaptive_mask(prob)
        mask = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
        area = int(mask.sum())
        mask_small = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mean_in = float(prob[mask_small == 1].mean()) if area > 0 else 0.0
        is_forged = (area >= AREA_THR and mean_in >= MEAN_THR)
        m_pred = (mask > 0).astype(np.uint8) if is_forged else np.zeros((h, w), np.uint8)
        if label == "forged":
            m_gt = np.load(Path(mask_dir) / f"{Path(p).stem}.npy")
            if m_gt.ndim == 3:
                m_gt = np.max(m_gt, axis=0)
            m_gt = (m_gt > 0).astype(np.uint8)
            f1 = f1_score(m_gt.flatten(), m_pred.flatten(), zero_division=0)
        else:
            f1 = 1.0 if not is_forged else 0.0  # authentic:空=1.0,非空=0.0
        f1s.append(f1)
        if verbose:
            print(f"    [{label:9s}] {Path(p).stem} — F1={f1:.4f} area={area} mean={mean_in:.3f}")
    return float(np.mean(f1s)) if f1s else float("nan")


# ---------------------------------------------------------------------------
# 主流程 / Main
# ---------------------------------------------------------------------------

def build_model(dino_path, device):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(dino_path, local_files_only=True, use_fast=False)
    encoder = AutoModel.from_pretrained(dino_path, local_files_only=True).eval().to(device)
    return TrainableDinoSegmenter(encoder, processor).to(device)


def main():
    ap = argparse.ArgumentParser(description="DINOv2 冻结骨干 + 解码头 分割训练(未在本地执行,见文件头声明)")
    ap.add_argument("--data-dir", default="/kaggle/input/recodai-luc-scientific-image-forgery-detection")
    ap.add_argument("--dino-path", default="/kaggle/input/dinov2/pytorch/base/1")
    ap.add_argument("--init-checkpoint", default="",
                    help="可选热启动权重(整模块 state_dict,如 CNNDINOv2-U52/model_seg_final.pt)")
    ap.add_argument("--out-dir", default="/kaggle/working/ckpt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2, help="notebook 推理 BATCH_SIZE=2;518×518 ViT-B + 解码头上 T4/P100 建议 2-4")
    ap.add_argument("--lr", type=float, default=2e-4, help="仅训练 decoder,起点建议 1e-4 ~ 3e-4")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-flip-aug", action="store_true", help="关闭翻转增广")
    ap.add_argument("--no-val-tta", action="store_true", help="验证关闭 3 路 TTA(快 3 倍)")
    ap.add_argument("--val-max", type=int, default=0, help="验证集最多评估图数(0=全部)")
    ap.add_argument("--limit-train", type=int, default=0, help="仅用前 N 张训练图(冒烟测试用)")
    ap.add_argument("--amp", action="store_true", help="启用混合精度(decoder 侧)")
    args = ap.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠️ 未检测到 CUDA:ViT-B@518 在 CPU 上训练不可行,仅供极小冒烟(--limit-train 2 --epochs 1)")
    os.makedirs(args.out_dir, exist_ok=True)

    auth_dir = f"{args.data_dir}/train_images/authentic"
    forg_dir = f"{args.data_dir}/train_images/forged"
    mask_dir = f"{args.data_dir}/train_masks"
    auth_imgs = sorted([str(Path(auth_dir) / f) for f in os.listdir(auth_dir)])
    forg_imgs = sorted([str(Path(forg_dir) / f) for f in os.listdir(forg_dir)])
    if args.limit_train:
        auth_imgs, forg_imgs = auth_imgs[: args.limit_train], forg_imgs[: args.limit_train]

    # 与 notebook 完全一致的划分(random_state=42, 20% 验证)
    from sklearn.model_selection import train_test_split

    train_auth, val_auth = train_test_split(auth_imgs, test_size=0.2, random_state=42)
    train_forg, val_forg = train_test_split(forg_imgs, test_size=0.2, random_state=42)
    print(f"train: {len(train_auth)} authentic + {len(train_forg)} forged | "
          f"val: {len(val_auth)} authentic + {len(val_forg)} forged")

    train_loader = torch.utils.data.DataLoader(
        ForgerySegTrainDataset(train_auth, train_forg, mask_dir, flip_aug=not args.no_flip_aug),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = build_model(args.dino_path, device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"✅ 热启动权重: {args.init_checkpoint}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable(decoder)={n_train:,} / frozen(encoder)={n_total - n_train:,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.encoder.eval()  # 冻结骨干保持 eval(Dropout 等)
        running, seen = 0.0, 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                logits = model.forward_seg(imgs)
                loss = combined_seg_loss(logits, masks)
            if not torch.isfinite(loss):
                print(f"  ⚠️ epoch {epoch}: 非法 loss({loss.item()}),跳过该 batch")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * imgs.size(0)
            seen += imgs.size(0)
        print(f"epoch {epoch}/{args.epochs} — train loss {running / max(seen, 1):.4f}")

        val_f1 = evaluate_official_f1(
            model, val_forg, val_auth, mask_dir, device,
            tta=not args.no_val_tta, max_images=args.val_max)
        print(f"epoch {epoch} — val official-F1 {val_f1:.4f} (best {max(best_f1, val_f1):.4f})")

        torch.save(model.state_dict(), os.path.join(args.out_dir, "model_seg_last.pt"))
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model_seg_best.pt"))
            print(f"  ⭐ 新最优,已保存 model_seg_best.pt")

    print(f"完成。best val official-F1 = {best_f1:.4f};输出目录 {args.out_dir}")
    print("checkpoint 为整模块 state_dict(encoder.*+seg_head.*),可直接被 notebook.ipynb 加载")


if __name__ == "__main__":
    main()
