# 下游任务一：G4扰动对CRE的影响

## 1. 科学问题与预测目标

该任务研究整体或局部G4扰动后，候选顺式调控元件（CRE）的多组学状态如何变化。模型同时使用四类信息：

1. 每个候选G4及其局部多模态环境；
2. CRE在扰动前的多模态状态；
3. G4与CRE的直接重叠、线性距离和Hi-C互作；
4. G4扰动前后的变化量及是否被直接靶向。

监督目标是扰动后CRE内除G4之外的全部模态，包括Pol II S5P。任务不是预测“G4是否存在”，而是预测G4变化发生后CRE响应状态。

数据预处理实现于`preprocess/preprocess_downstream_G4RegCRE.py`，模型代码位于`downstream/G4RegCRE/`。

## 2. 原始输入与信息隔离

预处理必须同时提供：

```text
control-dir       扰动前多组学peak文件
perturbed-dir     扰动后多组学peak文件
hic-interactions  BEDPE格式的Hi-C互作
```

对照组负责定义候选G4、候选CRE和模型输入；扰动组负责构建G4变化量及CRE监督目标。候选区域只由对照组筛选，防止利用扰动后结果反向决定训练样本。

重复实验、scaling factor、peak选择、`self_ratio`、区域覆盖并集和有向重叠的定义与预训练完全一致。其核心通过共享函数完成：

```python
mapped = map_records_to_regions(
    regions, modalities[name].peaks, "value"
)
states[name] = [
    calculate_modality(records, region.start, region.end)
    for records, region in zip(
        mapped, regions.itertuples(index=False)
    )
]
```

这样下游输入进入预训练Encoder时仍保持相同的列含义。

## 3. 候选CRE与G4区域

### 3.1 CRE筛选

默认`ATACPeak`直接使用对照组完整ATAC peak作为候选CRE，只删除完全重复坐标，不延长、不缩短、不合并相邻peak：

```python
frame = modalities["ATAC"].peaks[
    ["chrom", "start", "end"]
].drop_duplicates()
```

可选`ATAC_NFR`使用ATAC与NFR的实际交集：

```python
start = max(a_start, b_start)
end = min(a_end, b_end)
if end > start:
    rows.append((chrom, start, end))
```

该方法输出真实相交区间，不把交集重新扩展为固定长度。两种筛选方法代表不同CRE定义，生成的数据集必须分目录保存，不能混合训练。

### 3.2 G4窗口

每个对照组G4 peak构成一个候选G4。当前任务一使用以原始peak中心为锚点的固定窗口，默认1000 bp：

```python
center = (peak.start + peak.end) // 2
start = center - size // 2
end = start + size
```

靠近染色体两端时整体平移窗口以维持长度。原始G4坐标`g4_start/g4_end`与环境窗口`start/end`分别保存：

- 原始G4坐标用于判断是否与CRE直接重叠；
- 固定窗口用于计算G4周围的多模态环境。

这与任务二中“避开相邻G4的自适应窗口”不同，不能互换解释。

## 4. 三套矩阵及observed mask

### 4.1 G4对照环境

G4窗口使用预训练的16个非Pol II模态，输出与预训练完全同序的288列：

```text
g4/control_features: G x 288
g4/observed_features: 288
```

同一窗口在扰动组中也计算一次，但完整矩阵不直接写为模型目标，只用于提取G4自身三项变化量。

### 4.2 CRE对照环境

CRE输入不包含G4和Pol II，只保留其余15个模态。矩阵仍对齐到288列预训练schema，无法计算的G4相关列保留为0，同时`observed_features=False`：

```python
matrix = np.zeros((len(regions), len(feature_names)), dtype=np.float32)
observed = np.zeros(len(feature_names), dtype=bool)

if numerator in allowed and denominator in allowed:
    observed[column] = True
```

当前可观测列数为：

```text
15 x 3 + 15 x 14 = 255
```

模型调用`encode(values, observed_features)`时会直接删除其余33个token。真实测量为0和该分支没有测量该特征因此具有不同含义。

### 4.3 扰动后CRE目标

目标排除G4，但包含Pol II S5P及其余15个模态，因此仍为288项：

```text
16 x 3 unary features + 16 x 15 directed overlaps = 288
```

列名按固定顺序写入`target_feature_names`。Decoder第j个输出必须由该名称表解释，不能假设它与预训练288列完全同义，因为两者的模态集合不同。

## 5. 扰动编码

每个G4的扰动向量为4维：

```text
[is_targeted,
 delta_G4_peak,
 delta_G4_self_ratio,
 delta_G4_bin_ratio]
```

变化量使用扰动后减扰动前：

```python
delta = perturbed_values[:, columns] - control_values[:, columns]
perturbation = np.column_stack([targeted, delta]).astype(np.float32)
```

全局扰动时所有`is_targeted=1`；局部扰动时使用`--targeted-g4-bed`与原始G4 peak求交，命中者为1。

Dataset读取时只对`delta_G4_peak`使用带符号`log1p`缩放，比例差值保持方向并裁剪到`[-1,1]`：

```python
perturb[:, 1] = np.sign(perturb[:, 1]) * np.log1p(
    np.abs(perturb[:, 1])
) / np.log1p(peak_scale)
perturb = np.clip(perturb, -1, 1)
```

负值表示扰动后降低，正值表示升高，不能在预处理时取绝对值。

## 6. G4-CRE空间关系

每个G4-CRE pair使用三维one-hot：

| 类别 | 编码 | 判定 |
|---|---|---|
| overlap | `[1,0,0]` | 原始G4 peak与CRE直接相交 |
| no-HiC | `[0,1,0]` | 不直接相交且没有输入Hi-C互作 |
| far/HiC | `[0,0,1]` | BEDPE支持两者存在互作 |

代码中第二类历史命名为`proximal_no_HiC`，但当前判定没有线性距离阈值，因此它实际表示“没有直接重叠且没有Hi-C证据”，不能直接解释为生物学近端。

关系判定优先级为直接重叠高于Hi-C。同一pair被多个loop支持时保留最大Hi-C score。

### 6.1 稀疏pair存储

每个CRE理论上与所有G4配对，完整笛卡尔积可能非常大。代码用一个整数唯一表示pair：

```python
pair_key = cre_index * g4_count + g4_index
```

HDF5只显式保存overlap和Hi-C pair key，剩余合法pair隐式属于no-HiC类。这不会丢失pair定义，但避免重复存储相同G4和CRE特征。

### 6.2 模型实际使用的G4集合

训练默认每个CRE最多读取256个G4。先保留直接重叠和Hi-C支持的G4，再从其余no-HiC集合中按`seed + cre_index`确定性采样：

```python
required = np.unique(np.concatenate([overlap, far]))
remaining = np.setdiff1d(
    np.arange(self.g_count), required, assume_unique=True
)
sampled = rng.choice(remaining, missing, replace=False)
```

`--max-g4-per-cre 0`表示使用全部G4。若有证据的G4本身超过上限，当前代码会截取排序后的前K个，因此高关系密度CRE建议使用0或增大K。

## 7. HDF5组织

预处理只生成一个因子化HDF5：

```text
g4_reg_cre_dataset.h5
├── g4/
│   ├── control_features        G x 288
│   ├── perturbation            G x 4
│   ├── observed_features       288
│   └── id/chrom/start/end/g4_start/g4_end
├── cre/
│   ├── control_features        C x 288
│   ├── perturbed_targets       C x 288
│   ├── observed_features       288
│   └── id/chrom/start/end
├── relations/
│   ├── overlap_pair_key
│   ├── far_pair_key
│   └── far_hic_score
├── pretrain_feature_names
├── target_feature_names
└── perturbation_feature_names
```

矩阵按第一维chunk并使用轻量gzip压缩。G4、CRE和pair关系不逐pair重复展开，因此文件大小主要随`G+C+有证据pair数`增长，而不是随`G*C`增长。

## 8. Dataset、划分和归一化

实现文件：`downstream/G4RegCRE/dataset.py`。

CRE按固定随机种子划分，默认80%训练、10%验证、10%测试。一个样本对应一个CRE及其选中的K个G4：

```text
g4             K x 288
cre             288
relation        K x 3
distance        K
cis             K
hic             K
perturbation    K x 4
target           288
```

输入peak严格复用预训练`feature_normalization.npz`。扰动后目标的peak列只用下游训练集重新计算99.5%分位数；目标比例保持原始值。对应参数保存到`target_normalization.npz`，避免使用验证集或测试集估计尺度。

线性距离按中心计算：同染色体时为`G4_center - CRE_center`，保留上下游方向；跨染色体pair的线性距离置0，并由`cis=0`单独标记。

## 9. 模型结构与张量流

实现文件：`downstream/G4RegCRE/model.py`。

### 9.1 双预训练Encoder

预训练checkpoint被复制为两个参数不共享的Encoder：

- `g4_encoder`学习扰动来源环境；
- `cre_encoder`学习响应区域的初始状态。

MAE Decoder被删除，下游只调用`encode()`得到`[BIN]`表示。以预训练维度D=512为例：

```text
G4:  B x K x 288 -> B x K x 512
CRE: B x 288     -> B x 512
```

两套Encoder初始权重相同，但微调时可以分别适应G4和CRE两种语义。`--freeze-encoders`可冻结二者，只训练新增模块。

### 9.2 空间与扰动编码

空间编码输入7项：三类关系one-hot、有符号log距离、绝对log距离、cis标记和log Hi-C score。

```python
features = torch.cat([
    relation,
    signed_log_distance.unsqueeze(-1),
    absolute_log_distance.unsqueeze(-1),
    cis.unsqueeze(-1),
    hic.unsqueeze(-1),
], dim=-1)
spatial = self.net(features)
```

4维扰动向量通过独立MLP投影到D维。每个G4-CRE pair token为：

```python
pairs = self.pair_norm(g4_repr + spatial + perturb)
```

这里的相加要求三部分具有相同维度，使一个pair token同时包含G4环境、空间证据和扰动方向。

### 9.3 关系感知Fusion Encoder

CRE表示作为query，分别对overlap、no-HiC和Hi-C三组G4执行三套参数独立的cross-attention：

```python
for relation_id, attention in enumerate(self.cross):
    absent = relation[..., relation_id] == 0
    value, _ = attention(
        query, pairs, pairs, key_padding_mask=safe_mask
    )
    contexts.append(value)
```

得到三个`B x 1 x D`关系上下文后，与原CRE token组成4个token：

```text
[CRE, overlap_context, no_HiC_context, HiC_context]
```

小型Transformer进一步建模三类证据之间的组合，再通过门控残差更新CRE表示：

```python
gate = sigmoid(W[cre; fused])
output = LayerNorm(cre + gate * fused)
```

门控使模型可以保留CRE自身状态，也可以在有证据时引入G4扰动上下文。

### 9.4 Decoder

最终MLP输出`B x 288`，Sigmoid将结果限制到`[0,1]`，与目标归一化范围一致：

```text
B x 512 -> B x 288
```

## 10. 损失与评价

训练使用非零加权Smooth-L1：

```python
error = smooth_l1_loss(prediction, target, reduction="none")
weights = torch.where(target > 0, nonzero_weight, 1.0)
loss = (error * weights).sum() / weights.sum()
```

Smooth-L1比MSE更不容易被少量大误差完全支配；非零权重降低模型仅预测零值获得低损失的倾向。默认`nonzero_weight=2.0`。

验证和测试报告：

- weighted Smooth-L1 loss；
- 全部目标MSE、RMSE、MAE；
- 仅真实非零目标上的MAE。

这些指标评价预测精度，不自动证明G4对CRE具有因果作用。因果解释仍依赖真实扰动设计、对照条件和重复实验。

## 11. 运行与输出

预处理示例：

```bash
cd /mnt/afan/G4RegFormer
/home/afan/anaconda3/envs/single_cell/bin/python \
  preprocess/preprocess_downstream_G4RegCRE.py \
  --control-dir <control_multiomics_dir> \
  --perturbed-dir <perturbed_multiomics_dir> \
  --hic-interactions <loops.bedpe.gz> \
  --g4-window-size 1000 \
  --cre-method ATACPeak \
  --perturbation-mode global
```

局部扰动需增加：

```bash
--perturbation-mode local --targeted-g4-bed <targeted_G4.bed>
```

训练：

```bash
bash downstream/G4RegCRE/train.sh
```

每次训练保存：

```text
downstream/G4RegCRE/output/<timestamp>/
├── args.json
├── run.log
├── split_indices.npz
├── target_normalization.npz
├── checkpoint-last.pth
├── checkpoint-best.pth
├── history.tsv
├── test_metrics.json
└── test_predictions.npz
```

最佳checkpoint按验证集loss选择，测试集只在训练结束后评估一次。

## 12. 当前状态与限制

- 服务器当前尚未登记正式扰动前后目录和Hi-C BEDPE路径，因此尚未生成正式任务一训练HDF5。
- no-HiC类别不等同于近端，缺失Hi-C信号也不等同于不存在空间互作。
- 当前CRE随机划分可能让相邻CRE进入不同集合；需要更严格泛化评价时应改为染色体或区块划分。
- `max_g4_per_cre`采样影响计算量和弱关系背景，实验比较时必须固定参数和随机种子。
- 模型预测扰动后状态，但只有真实干预、匹配对照和独立验证才能支持因果结论。
