# TextAlign 样本对数量核对报告

日期：2026-01-15

## Step 1: 正样本统计 (train_coco_captions.json)
```text
== train_coco_captions.json ==
len(image_ids) = 9000
len(captions)  = 9000
unique image_ids = 9000
```

## Step 2: 硬负样本统计
**2.1 jsonl 行数**
```text
9000 data/nsd_text/train_coco_captions_hard_negs.jsonl
```

**2.2 hard_negs_clip.pt 的 shape**
```text
== hard_negs_clip.pt ==
type: <class 'dict'>
image_ids: (9000,)
neg_text_feats: (9000, 768)
valid_mask: (9000,)
```

## Step 3: subj01 训练用子集 (train_coco_text_clip.pt)
```text
== train_coco_text_clip.pt ==
type: <class 'dict'>
image_ids: (9000,)
text_feats: (9000, 768)
```
注意：此处 `train_coco_text_clip.pt` 显示为 **9000** 条样本，**并非** 预期的 1910 条。这说明 `train_coco_text_clip.pt` 包含了所有 COCO 覆盖的图像特征，而不仅仅是 subj01 训练集的子集（或者 subj01 所有的 9000 张图都在训练集中）。

根据之前 `audit_coco_caption_coverage.py` 的运行结果：
> Subj 01: Unique Train = 9000, With Caption = 9000, Coverage = 100.0%

这解释了为什么这里是 9000：**Subject 1 的训练集恰好就是这 9000 张有 COCO 标注的图**。预期中的 "1910" 可能是基于其他数据假设，但实测 S1 是 9000。

## 结论
全局 COCO-NSD caption 子集为 9000 条正样本 / 9000 条硬负样本；subj01 训练实际使用的子集也是 **9000 条**（Coverage 100%），**远高于** 预期的 1910 条。这也意味着 Subject 1 的数据覆盖非常完备。
