# 下游任务二：CRE到基因的转录调控

## 1. 科学问题与任务边界

该任务把与G4相关的局部多模态状态聚合到基因层面，研究Promoter、Enhancer和TAD boundary相关G4如何共同关联基因转录活性。当前监督信号为Pol II S5P，模型执行连续值回归。

完整流程分为两个依次执行的数据阶段：

1. 为每个G4构建自适应局部窗口，计算288项多模态特征并标注位置角色；
2. 将不同角色的G4分配给基因，构建每个基因对应的变长G4集合和Pol II目标。

预处理实现于`preprocess/preprocess_downstream_CRERegGene.py`，训练实现于`downstream/CRERegGene/`。

## 2. G4自适应窗口

### 2.1 为什么不统一强制扩展

目标窗口长度默认1000 bp。若机械地以G4中心扩展，相邻G4可能被纳入同一窗口，使一个样本同时表示多个G4。当前实现先计算前后可用空间，再分配扩展长度：

```python
left_room = max(0, peak_start - previous_end)
right_room = max(0, next_start - peak_end)
extra = max(0, window_size - (peak_end - peak_start))

left_goal = extra // 2
right_goal = extra - left_goal
left_extension = min(left_goal, left_room)
right_extension = min(right_goal, right_room)
```

若一侧空间不足，缺少长度尽量转移到另一侧：

```python
left_deficit = left_goal - left_extension
right_extension += min(
    left_deficit, right_room - right_extension
)
right_deficit = max(0, right_goal - right_extension)
left_extension += min(
    right_deficit, left_room - left_extension
)
```

两侧都不足时，窗口保持实际可扩展长度，不跨入相邻G4。输出同时保存目标长度是否达到、左右扩展量和邻近peak冲突标记，用于质量控制。

### 2.2 G4自身特征

G4自身按原始peak与自适应窗口的实际交集计算，而不是固定写成1：

```python
anchor_start = max(g4_start, window_start)
anchor_end = min(g4_end, window_end)
anchor_bases = anchor_end - anchor_start

G4_self_ratio = anchor_bases / g4_length
G4_bin_ratio = anchor_bases / window_length
```

通常原始G4完整保留时`G4_self_ratio=1`；`G4_bin_ratio`表示G4覆盖整个自适应窗口的比例。若原始G4长于目标窗口，相关QC字段会单独标记。

## 3. G4区域多模态特征

每个自适应窗口沿用预训练的288维schema：16个非Pol II模态，每个模态3项自身特征和15项有向重叠比例。

对非G4模态统一调用：

```python
state = calculate_modality(
    modality_maps[name][index], region.start, region.end
)
```

其含义与预训练一致：

| 特征 | 含义 |
|---|---|
| `M_peak` | 与窗口相交的最高校正peak值 |
| `M_self_ratio` | 代表peak位于窗口内的比例 |
| `M_bin_ratio` | 模态区间并集覆盖窗口的比例 |
| `A_overlapping_B` | A与B共同覆盖碱基占B窗口内覆盖碱基的比例 |

所有特征按预训练列顺序写入矩阵并保留两位小数，保证可直接输入预训练Encoder。

## 4. G4位置标签

### 4.1 Promoter

基因注释使用Ensembl release 102、GRCm38。Promoter定义为链特异TSS上游2 kb、下游1 kb：

```python
if strand == "+":
    tss = start_1based - 1
    promoter = [tss - upstream, tss + downstream]
else:
    tss = end_1based - 1
    promoter = [tss - downstream, tss + upstream]
```

使用扩展后的G4窗口与Promoter求实际重叠。一个窗口可覆盖多个Promoter，所有命中基因都写入`promoter_genes`。

### 4.2 Enhancer

在当前G4窗口内分别裁剪ATAC、H3K27ac和H3K4me1区间，再计算三者共同交集：

```python
triple = intersect_intervals(
    intersect_intervals(
        states["ATAC"].intervals,
        states["H3K27ac"].intervals,
    ),
    states["H3K4me1"].intervals,
)
enhancer_evidence = interval_bases(triple) > 0
```

只有三个模态在窗口内存在共同覆盖碱基时才产生Enhancer证据，不是分别存在三个peak即可。

### 4.3 TAD

TAD位置证据读取正式CT-TADB边界概率文件，要求：

- 坐标为mm10/GRCm38；
- 至少包含`chrom/start/end/probability`；
- 概率位于`[0,1]`；
- 高于`--ct-tadb-threshold`，默认0.5。

```python
boundaries = boundaries[
    boundaries["value"].between(0, 1)
    & (boundaries["value"] >= threshold)
]
```

代码不会用CTCF峰替代CT-TADB，也不会将缺失的CT-TADB输入轨道填0。没有正式预测文件时，使用`--rebuild-g4-regions`会明确报错。

### 4.4 标签优先级

位置证据可能同时存在。单标签按固定优先级确定：

```text
Promoter > Enhancer > TAD > Others
```

```python
if promoter:
    label = "Promoter"
elif enhancer:
    label = "Enhancer"
elif tad:
    label = "TAD"
else:
    label = "Others"
```

所有原始命中同时写入`position_multilabel`，并用`position_ambiguous=1`标记多重证据。训练角色使用优先级后的单标签，但原始证据仍可用于敏感性分析。

## 5. G4位置数据输出

```text
g4_position_1kb_full.tsv.gz
    坐标、标签、288项特征和各模态原始区间

g4_position_1kb_regions.tsv
    G4窗口、扩展QC、位置标签和证据

g4_position_1kb_columns.tsv
    288项输入及label列定义

g4_position_1kb_dataset.h5
    matrix: N x 289，其中最后一列为label

g4_position_1kb_overview.png
    类别数量、窗口长度、染色体组成和模态覆盖率
```

位置HDF5中的Pol II S5P属性为`included=False`，它不参与G4局部输入。

## 6. 基因与CRE/G4归属

### 6.1 基因和表达量

基因坐标及链方向来自同一Ensembl GTF。RNA表中的重复列取平均形成`rna_tpm`，用于筛选E-G候选基因和构建代理证据。RNA值写入基因级HDF5，但当前回归模型不把RNA作为输入。

### 6.2 Promoter G4直接归属

Promoter角色G4根据窗口与链特异Promoter的实际重叠直接连接基因：

```python
assignments.append({
    "gene_id": gene_id,
    "g4_id": region.g4_id,
    "g4_role": "Promoter",
    "assignment_method": "expanded_G4_overlap_Ensembl_promoter",
    "assignment_score": region.promoter_overlap_ratio,
})
```

一个G4覆盖多个Promoter时保留多条assignment；不会强制选择最近基因。

### 6.3 Enhancer候选元素与ABC基础分数

候选调控元素由ATAC peak与H3K27ac重叠定义。元素活性为：

```text
activity = sqrt(ATAC_peak x H3K27ac_peak)
```

对每个表达量达到阈值的基因，在默认500 kb内计算距离接触近似：

```python
contact = (
    maximum(distance, min_distance) / min_distance
) ** (-gamma)
contribution = activity * contact
ABC = contribution / sum(contribution for candidate elements of gene)
```

默认`min_distance=5 kb`、`gamma=0.87`、ABC阈值0.02。这里使用的是距离幂律近似，不是当前样本5 kb KR-normalized Hi-C接触矩阵，因此应称为`ABC-compatible`实现，不等同于正式ABC结果。

## 7. 六类E-G证据的当前实现

当前代码没有调用六个原始模型的官方checkpoint或完整流程，而是在同一候选pair集合上构建六个透明代理分数。这个区别必须保留在结果表和论文表述中。

| 名称 | 当前代码使用的主要证据 | 当前状态 |
|---|---|---|
| ABC | ATAC、H3K27ac、距离幂律contact | ABC-compatible |
| CIA | enhancer活性、距离、CTCF、公共边界 | CIA-inspired proxy |
| ENCODE-rE2G | ABC、活性、RNA、距离、H3K4me3、邻域 | rE2G-inspired proxy |
| EpiMap | 单条件活性、RNA兼容性、距离 | EpiMap-inspired proxy |
| GraphReg | 活性消息、距离和边界约束 | GraphReg-inspired proxy |
| Enformer | G4局部peak/覆盖、活性和距离 | Enformer-inspired proxy |

例如CIA代理分数为：

```python
CIA_score = (
    activity
    * distance_contact_percentile
    * (0.5 + 0.5 * CTCF_peak_percentile)
    * same_domain_proxy
)
```

GraphReg代理先计算每个pair的消息，再在同一基因内归一化；Enformer代理没有使用DNA序列模型输出，而是使用G4 peak与覆盖比例构造局部sequence proxy。因此这些结果只能用于流程开发和候选筛选，不能写成“六个正式模型一致预测”。

### 7.1 共识计算

每个代理方法在同一G4候选基因内取前`top_n`，默认3个。ABC还必须达到绝对阈值：

```python
rank = evidence.groupby("g4_id")[score].rank(
    ascending=False, method="min"
)
support = rank <= top_n
ABC_support &= ABC_score >= abc_threshold
```

随后统计：

```text
strict_consensus   = 6/6支持
majority_consensus = 至少4/6支持
accepted           = strict_consensus
```

只有严格六路一致的Enhancer-Gene pair进入当前训练HDF5；多数支持结果只保留用于分析。

正式接入ABC、CIA-adapted、ENCODE-rE2G、EpiMap、GraphReg和Enformer后，应保留每个方法原始分数、阈值、版本和证据等级，再重新生成assignment。不可用模型应标记为NA，不能当作反对票。

## 8. TAD boundary G4归属

G4位置标签中的TAD证据来自CT-TADB；基因归属阶段当前使用公共E12.5心肌细胞Hi-C边界。对每个TAD角色G4：

1. 找到距离G4中心最近的公共边界；
2. 找到该边界左侧和右侧最近的TSS；
3. 只保留距离不超过`boundary_max_distance`的基因。

```python
split = np.searchsorted(tss, boundary, side="right")
for position, side in ((split - 1, "left"), (split, "right")):
    distance = abs(gene_tss - boundary)
```

该方法表达“边界两侧候选基因”，不是已验证的调控靶点，输出状态明确写为`public_reference_proxy`。

## 9. Pol II S5P监督目标

Pol II不进入288项G4输入，而是分别在完整基因体和链特异Promoter内计算：

```python
body = summarize("gene_start", "gene_end")
promoter = summarize("promoter_start", "promoter_end")
targets = np.column_stack([
    body[:, 0], body[:, 1],
    promoter[:, 0], promoter[:, 1],
])
```

四个回归目标为：

```text
polii_gene_peak
polii_gene_ratio
polii_promoter_peak
polii_promoter_ratio
```

因此一个基因样本的模型输出是`1 x 4`，其中4不是实验重复，而是基因体和Promoter两个区域上的peak强度及覆盖比例。如果研究问题最终只保留“Promoter Pol II peak”，需要显式修改target选择和回归头，不能把当前4维输出直接解释为单一peak。

## 10. 基因级ragged HDF5

每个基因关联的G4数量不同。代码把所有assignment按基因连续拼接，并用累计偏移定位：

```python
offsets[1:] = np.cumsum([
    counts.get(index, 0) for index in range(len(summary))
])
selected = features[
    ordered["g4_row_index"].to_numpy(dtype=np.int64)
]
```

第i个基因的区域位于：

```text
features[gene_offsets[i] : gene_offsets[i+1]]
```

文件结构：

```text
cre_reg_gene_dataset.h5
├── features          A x 288，所有assignment拼接
├── gene_offsets      (N_gene + 1)
├── gene_index        A
├── g4_row_index      A
├── role              A，Promoter=0/Enhancer=1/TAD_boundary=2
├── polii_targets     N_gene x 4
├── rna_tpm           N_gene
├── gene_id           N_gene
└── feature_names     288
```

当前已生成数据包含13,611个基因和18,985条G4 assignment，其中Promoter 15,794条、Enhancer 2,779条、TAD boundary 412条。

## 11. Dataset与batch padding

实现文件：`downstream/CRERegGene/dataset.py`。

基因按固定随机种子划分，默认80%训练、10%验证、10%测试。每个样本读取自己的变长G4集合。`collate_genes()`按当前batch最大长度补零并建立有效位mask：

```python
features = torch.zeros(B, K_max, 288)
roles = torch.zeros(B, K_max, dtype=torch.long)
valid = torch.zeros(B, K_max, dtype=torch.bool)

features[row, :length] = item["features"]
valid[row, :length] = True
```

输入peak复用预训练归一化。Pol II四个目标中，第1和第3列为peak，分别用训练集非零99.5%分位数缩放；两个ratio保持原值。

## 12. 模型结构与张量流

实现文件：`downstream/CRERegGene/model.py`。

### 12.1 区域编码

每个G4的288项特征先通过预训练Encoder，得到D维`[BIN]`表示：

```text
B x K x 288 -> B x K x D
```

代码先展平区域维度，再恢复batch：

```python
encoded = self.region_encoder.encode(
    features.reshape(-1, width)
)
encoded = encoded.reshape(batch, count, -1)
```

padding行也会经过区域Encoder，但随后在基因聚合Transformer中被mask，不参与注意力汇总。这保证语义正确，但会产生一部分额外计算。

### 12.2 角色编码

同样的局部多模态特征在Promoter、Enhancer和TAD boundary中具有不同含义，因此为每个G4表示加入角色embedding：

```python
encoded = encoded + self.role_embedding(role)
```

角色编码只表示assignment类型，不替代多模态特征本身。

### 12.3 基因聚合

每个基因序列前加入可学习`[GENE]` token：

```text
[GENE], G4_1, G4_2, ..., G4_K
```

padding mask为：

```python
padding = torch.cat([
    torch.zeros(batch, 1, dtype=torch.bool),
    ~valid,
], dim=1)
```

两层Transformer聚合同一基因关联的全部G4，取首位`[GENE]`输出作为基因表示。默认D=512：

```text
B x (K+1) x 512 -> B x 512
```

回归头输出四个归一化Pol II指标：

```text
B x 512 -> B x 4
```

Sigmoid保证输出位于`[0,1]`，与目标归一化范围一致。`--freeze-encoder`可以固定预训练区域Encoder，只训练角色编码、基因聚合器和回归头。

## 13. 损失与评价

训练使用非零加权Smooth-L1：

```python
error = smooth_l1_loss(prediction, target, reduction="none")
weights = torch.where(target > 0, nonzero_weight, 1.0)
loss = (error * weights).sum() / weights.sum()
```

默认非零目标权重为2。验证与测试报告：

- weighted Smooth-L1 loss；
- 四个目标合并计算的MSE、RMSE和MAE；
- 对每个有方差的目标分别计算Pearson，再取平均。

平均Pearson用于总体比较，但正式结果应同时报告四个目标各自的Pearson和误差，避免高表现目标掩盖低表现目标。

## 14. 运行与输出

预处理：

```bash
cd /mnt/afan/G4RegFormer
/home/afan/anaconda3/envs/single_cell/bin/python \
  preprocess/preprocess_downstream_CRERegGene.py
```

只有在正式CT-TADB结果准备完成并需要重建位置数据时使用：

```bash
--rebuild-g4-regions \
--ct-tadb-boundaries data/reference/ct_tadb_mm10_boundaries.bed.gz
```

训练：

```bash
bash downstream/CRERegGene/train.sh
```

训练输出：

```text
downstream/CRERegGene/output/<timestamp>/
├── args.json
├── run.log
├── split_indices.npz
├── target_normalization.npz
├── checkpoint-last.pth
├── checkpoint-best.pth
├── history.json
├── test_metrics.json
└── test_predictions.npz
```

最佳模型按验证集loss选择，测试集在训练结束后只评估一次。

## 15. 当前结果的解释边界

- 当前六类E-G分数是代理流程，不是六个官方模型的正式输出，不能据此宣称六模型验证。
- ABC当前使用距离幂律contact，尚未替换为匹配样本的5 kb KR/SCALE Hi-C。
- TAD位置标签与TAD基因归属使用不同证据来源，前者是CT-TADB，后者是公共心肌Hi-C边界代理。
- 基因随机划分可能让共享G4或邻近基因跨越训练、验证和测试集合，严格泛化评估应考虑染色体或区域分组。
- Pol II关联预测体现统计关联，G4到基因的因果结论仍需真实扰动数据和独立实验验证。
