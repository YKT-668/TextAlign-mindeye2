# MindEye2 官方评测体系审计（以 final_evaluations.ipynb 实际代码为准）

来源：`source/TextAlign-mindeye2-231850a/src/final_evaluations.ipynb`（55 个 cell，其中 27 个 code cell）
所有结论均标注证据单元（cell 编号）与可信度（VERIFIED / INFERRED / BLOCKED）。

## 0. 三个必须先知道的"官方约定"

| # | 约定 | 证据 | 影响 |
|---|---|---|---|
| **A** | **enhanced 重建在评测前会与 blurry 重建融合**：`all_recons = all_recons*.75 + all_blurryrecons*.25`（仅当 `model_name` 含 `enhanced`） | cell 8（L196-198） | **所有下游图像指标（PixCorr/SSIM/AlexNet/Inception/CLIP/EffNet/SwAV/GNet）都是在融合后的图上算的**。我们本轮 A/B 评测用的是**未融合**的 `enhancedrecons`，因此数值**不可与论文表直接对比** |
| **B** | 所有图像先 Resize 到 **256**（cell 8），再按各模型自己的 preprocess 二次 resize | cell 8 | 需要保证评测输入尺寸一致 |
| **C** | "2-way Identification" 实际是 **N-way（N=1000）**：`np.corrcoef` 得到 1000×1000 相关矩阵，统计"对角是否严格大于该行其它所有值" | cell 17 `two_way_identification()` | 与"两两配对 2AFC"完全不同，**不能用普通余弦相似度冒充** |

---

## C1. Low-Level Image Metrics

### PixCorr（cell 19，VERIFIED）

```python
preprocess = transforms.Compose([transforms.Resize(425, interpolation=BILINEAR)])
all_images_flattened = preprocess(all_images).reshape(N, -1)
all_recons_flattened = preprocess(all_recons).view(N, -1)
corrsum += np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1]
pixcorr = corrsum / N
```
- 输入：`[0,1]` 的 float 张量（不做归一化）
- 尺寸：**Resize 425×425**（bilinear），**RGB 三通道**，直接 flatten
- 逐图计算 Pearson r → 取**算术平均**
- 不需要重复刺激平均（图像层面）

### SSIM（cell 21，VERIFIED）

```python
img_gray   = rgb2gray(preprocess(all_images).permute(0,2,3,1).cpu())
recon_gray = rgb2gray(preprocess(all_recons).permute(0,2,3,1).cpu())
ssim(rec, im, multichannel=True, gaussian_weights=True,
     sigma=1.5, use_sample_covariance=False, data_range=1.0)
```
- **转灰度**（`skimage.color.rgb2gray`，权重 0.2125/0.7154/0.0721）
- Resize **425**，`data_range=1.0`，Gaussian window σ=1.5
- 逐图 → 平均
- 依赖：**scikit-image**（当前 gpu_venv 未安装）

> 两者都只需 CPU，3000 张（A/B/GT）预计数分钟。

---

## C2. High-Level Visual Metrics（cell 17 的 `two_way_identification`）

统一算法（VERIFIED）：
```python
preds = model(preprocess(recon))   # 特征，flatten(1)
reals = model(preprocess(gt))
r = np.corrcoef(reals, preds)[:N, N:]      # N×N Pearson
success = r < np.diag(r)[:, None]          # 对角严格大于该行所有其它
perf = mean(success_cnt) / (N-1)
```
> 注意：用的是 **Pearson 相关**，不是余弦；且是 **N=1000 的全库排名**。

| 指标 | 模型 / 权重 | 特征层 | 预处理 | 依赖 |
|---|---|---|---|---|
| AlexNet(2) | `torchvision alexnet(AlexNet_Weights.IMAGENET1K_V1)` | `features.4` | `Resize(256, BILINEAR)` + ImageNet mean/std（**无 CenterCrop**） | torchvision 权重（**本地缺失**，233MB） |
| AlexNet(5) | 同上 | `features.11` | 同上 | 同上 |
| InceptionV3 | `inception_v3(Inception_V3_Weights.DEFAULT)` | `avgpool` | `Resize(342, BILINEAR)` + ImageNet mean/std | 权重 104MB（**缺失**） |
| CLIP ViT-L/14 | **OpenAI `clip` 包**：`clip.load("ViT-L/14")`，`clip_model.encode_image` | final layer | `Resize(224, BILINEAR)` + CLIP mean/std | `clip` 包 + `ViT-L-14.pt`（**缺失**）；可用已下载的 `openai/clip-vit-large-patch14`（**同源 OpenAI 权重**，HF 版）替代（INFERRED，需数值核对） |

## C3. Feature Distance Metrics（cell 29 / 31，VERIFIED）

| 指标 | 模型 | 特征层 | 预处理 | 公式 | 方向 |
|---|---|---|---|---|---|
| EffNet-B | `efficientnet_b1(EfficientNet_B1_Weights.DEFAULT)` | `avgpool` | `Resize(255)` + ImageNet norm | `scipy.spatial.distance.correlation(gt_i, fake_i)` 的均值 | **越低越好** |
| SwAV | `torch.hub.load('facebookresearch/swav:main','resnet50')` | `avgpool` | `Resize(224)` + ImageNet norm | 同上（相关距离） | **越低越好** |

- 没有逐维归一化，直接对原始 avgpool 特征做 **Pearson-based correlation distance**（1 − r）
- 依赖：`torch.hub` 在线下载 SwAV（**缺失**，约 100MB）

## C4. Retrieval Metrics（cell 14，VERIFIED）

```python
for loop in range(30):
    random_samps = np.random.choice(range(N), 300, replace=False)
    emb  = clip_img_embedder(all_images[random_samps])   # bigG tokens -> flatten -> L2norm
    emb_ = all_clipvoxels[random_samps]                  # CLIP-Brain -> flatten -> L2norm
    fwd = topk(batchwise_cosine_similarity(emb_, emb), labels, k=1)  # brain -> image
    bwd = topk(batchwise_cosine_similarity(emb, emb_), labels, k=1)  # image -> brain
percent_correct_fwd/bwd = mean over 30 loops（并给出 SE 与 95% CI）
```
| 要素 | 官方设定 |
|---|---|
| 库大小 | **300**（每轮随机抽 300，**做 30 轮**后取均值） |
| 随机基线 | 1/300 ≈ 0.0033 |
| 编码器 | 图像端 = **bigG（laion2b_s39b_b160k）tokens**；脑端 = **`clipvoxels`（CLIP-Brain）** |
| 方向 | **Forward = Brain→Image**；**Backward = Image→Brain** |

**关键判定（与我们的产物对应）**：

| 我们的 tensor | 是否官方 retrieval 的合法输入 | 说明 |
|---|---|---|
| `clipvoxels` (1000,256,1664) | ✅ **是**（脑端 = CLIP-Brain） | 官方 Fwd/Bwd Retrieval 的唯一正确脑端输入 |
| `prior_out` (1000,256,1664) | ❌ 不是 | 扩散 prior 的中间 token，**语义不同**，不能当 CLIP-Brain |
| `backbones` (1000,256,1664) | ❌ 不是 | backbone 输出，非 CLIP 空间 |
| `enhancedrecons` (1000,3,224,224) | 仅用于**生成图像侧**指标 | 不能与 clipvoxels 混用做"retrieval" |
| `ids` (1000,) | ✅ 索引对齐 | — |

> 因此：**官方 Fwd/Bwd Retrieval 属于"表征层"指标，A/B 精炼不会改变它**（因为 clipvoxels 未变）。若要做与图像相关的检索，只能用本轮定义的 1000-way 图像检索（CLIP 图像 vs GT 图像），二者不可混称。

---

## C5. Caption Evaluation（cell 40 → 53，VERIFIED）

官方同时用了**三套 caption**：
- `all_captions`：GT caption（我们已有）
- `all_predcaptions`：脑→GIT 预测 caption（我们已有）
- `all_git_generated_captions`：**用 GT 图像喂 GIT 生成的 caption**（`evals/all_git_generated_captions.pt`，**本地缺失**）← 作为"图像可生成的 caption"参照，用于**相对归一化**

三组两两组合：`brain↔ref`、`brain↔image`、`image↔ref`，并定义相对指标：
```
relative = M(brain, image_caption) / M(image_caption, ref)
```

| 指标 | 实现 | 输入 | 依赖 |
|---|---|---|---|
| METEOR | `evaluate.load('meteor')` | 三个 caption 列表 | `evaluate` + `nltk`（**均缺**） |
| ROUGE-1 / ROUGE-L | `evaluate.load('rouge')` | 同上 | 同上（需 `rouge_score`/`nltk`） |
| Sentence Similarity | `SentenceTransformer('all-MiniLM-L6-v2')` 余弦 | 同上 | `sentence-transformers` + 模型 90MB（**均缺**） |
| CLIP-B | `CLIPModel("openai/clip-vit-base-patch32").get_text_features` 余弦 | 同上 | 权重 600MB（**缺**） |
| CLIP-L | `CLIPModel("openai/clip-vit-large-patch14").get_text_features` 余弦 | 同上 | ✅ **本地已有** |

**重要判定**：
1. 官方这些 caption 指标**全部是 text-text 相似度**（没有图像输入）。因此 **A/B 两组在这些指标上完全相同**（predcaptions 未变），**不能**用于精炼效果。
2. `all_git_generated_captions.pt` 缺失 → 相对归一化的三项（`*_relative`）**BLOCKED**，只能算绝对项。补齐方式二选一：① 下载该小文件（pscotti/mindeyev2 `evals/`，KB 级）；② 用已下载的 GIT（CPU 可跑）对 1000 张 GT 图像重新生成 caption。
3. **不得**把 `all_predcaptions` 当 GT 使用（本轮评测已严格遵守）。

## C6. Brain Correlation（GNet，cell 33→37，部分 BLOCKED）

```python
test_voxels_averaged[i] = mean(betas[locs])          # 每张图 3 次 trial 平均
GNet = GNet8_Encoder(device, subject, model_path=f"{cache_dir}/gnet_multisubject.pt")
beta_primes = GNet.predict(recon_list)               # 由**生成图像**预测脑活动
PeC = PearsonCorrCoef(num_outputs=len(recon_list))
for region, mask in subject_masks:
    score = PeC(test_voxels_averaged[:, mask].T, beta_primes[:, mask].T)  # 逐体素 Pearson
    region_brain_correlations[region] = mean(score)
```
| 要素 | 官方设定 | 本地状态 |
|---|---|---|
| 脑区划分 | `brain_region_masks.hdf5`：`nsd_general, V1, V2, V3, V4, higher_vis` | **MISSING** |
| GNet 权重 | `gnet_multisubject.pt` | **MISSING** |
| 真实 beta | `betas_all_subj01_fp32_renorm.hdf5`（每图 3 次平均） | ✅ 已有 |
| 输入图像 | 融合后的 `all_recons`（`.75/.25`） | ✅ 可复算 |

**判定**：Brain Correlation **BLOCKED_MISSING_ASSETS**（缺 GNet 权重 + 脑区 mask）。该指标确实是"图像→脑"的合法 A/B 指标（它能反映精炼图像是否更接近真实诱发活动），但**必须补齐两个资产**才能算。

---

## C7. 官方指标清单与产物依赖（速查）

| 官方指标 | 依赖的我们的产物 | 依赖的模型/数据 | 现状 |
|---|---|---|---|
| PixCorr / SSIM | enhancedrecons(+blurry) & GT | 无（skimage） | ✅ 可算（需装 scikit-image） |
| AlexNet(2)/(5) | 同上 | alexnet 权重 | ⚠️ 需下载 233MB |
| InceptionV3 | 同上 | inception 权重 | ⚠️ 需下载 104MB |
| CLIP ViT-L/14 2-way | 同上 | `clip`+ViT-L-14.pt / 或 HF CLIP-L 替代 | ⚠️ 需核对等价性 |
| EffNet-B / SwAV | 同上 | effnet_b1 30MB / swav 100MB | ⚠️ 需下载 |
| CLIP-Brain Fwd/Bwd（300×30） | **clipvoxels** + GT 图像 | bigG（✅ 已下载） | ✅ 可算（GPU 更快） |
| Caption 绝对项 | predcaptions + GT captions | evaluate/nltk/sentence-transformers | ⚠️ 需装依赖 |
| Caption 相对项 | 追加 `all_git_generated_captions.pt` | 同上 | ❌ 文件缺失 |
| Brain Corr（6 区） | recons | gnet + masks | ❌ 两项缺失 |

## C8. 与论文口径的差异备忘（写论文时必须声明）

1. 本项目的 S1 推理脚本导出 `enhancedrecons` 为 **224**；官方评测会把图 Resize 到 **256** 再融合 blurry。
2. 我们此前的 A/B 评测用的是**未融合**的 enhancedrecons，且图像相似度用 bigG token 余弦（非官方的 Pearson N-way）。
3. 官方 retrieval 是 **brain(clipvoxels) ↔ GT 图像**；我们的是 **生成图像 ↔ GT 图像**，两者含义不同。
4. 官方 caption 指标是 text-text，**不含图像**。

