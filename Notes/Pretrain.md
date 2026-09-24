# G4RegFormer 预训练

## 1. 任务定义

预训练以 mm10/GRCm38 的固定长度基因组 bin 为样本，学习 G4、染色质可及性、核小体状态、TF、组蛋白修饰和 R-loop 在同一局部区域内的联合表示。模型不预测类别，而是随机隐藏部分已知特征，再根据剩余特征重建被隐藏值。

当前预训练输入包含 16 个模态，每个模态有 3 个自身特征和 15 个有向跨模态重叠特征：

```text
16 x (3 + 15) = 288 features
```

Pol II S5P 不进入这 288 项输入。它只登记在跨任务共享的原始数据目录中，供下游转录活性任务读取，因此不会通过自身信号或 Pol II 相关重叠列泄漏到预训练表示。

预训练不构建 unit，不连接相邻 bin，也不使用染色体顺序位置编码。每一行只表示一个 bin 内部的多模态状态。

## 2. 数据预处理

实现文件：`preprocess/preprocess_pretrain_bin.py`。

脚本默认将结果写入`data/preprocessed/pretrain_data/bin_1000bp/`。如果通过`--bin_size`修改区间长度，目录名和文件名前缀会同步变化，例如512 bp对应`bin_512bp/mm10_512_*`，因此不同参数的结果不会相互覆盖。预处理完成后生成以下文件：

| 文件 | 保存内容 | 用途 |
|---|---|---|
| `mm10_1000_multimodal_chrall_all_bins.tsv.gz` | mm10所有完整bin的坐标、288项数值特征、16个模态的原始peak区间、零值数和非零值数 | 保留最完整的可读结果，用于按坐标检查任意bin的计算过程；包含全零bin，不直接用于预训练 |
| `mm10_1000_multimodal_chrall_nonallzero_bins.tsv.gz` | 从完整表中筛出的至少含一个非零训练特征的bin，列结构与完整表相同 | 快速核查实际进入训练集合的区域，同时保留原始区间列；不作为模型的高效读取文件 |
| `validation_summary.tsv` | 每条染色体及全基因组的总bin数、全零与非全零bin数、非全零比例、最大比例值和非法比例数量 | 检查筛选规模以及所有比例是否位于合法范围 |
| `column_manifest.tsv` | 每个模态的源文件、重复实验列、实际使用的scaling factor和清洗后的peak数量 | 追踪输入来源并确认每个重复实验是否正确匹配校正系数 |
| `chromosome_nonzero_heatmap.png` | 每条染色体每1000个bin中非全零bin所占比例 | 展示多模态信号在染色体上的区域密度，并比较不同长度染色体；空白部分不是全零区域，而是较短染色体没有对应位置 |
| `modality_cooccurrence_log2_enrichment.tsv` | 16个模态两两共现的`log2`富集值矩阵 | 提供可继续统计分析的数值结果；正值表示共同出现高于独立分布期望，负值表示低于期望 |
| `modality_cooccurrence_enrichment_heatmap.png` | 上述共现富集矩阵的热图 | 直观查看经常共同覆盖同一bin的模态组合；它描述统计关联，不代表因果关系 |
| `pretrain_input/mm10_1000_regions.tsv` | 每个训练样本的`chrom/start/end/name/index` | 保存HDF5每一行对应的基因组区间；`index`与矩阵行号严格一一对应 |
| `pretrain_input/mm10_1000_feature_columns.tsv` | 288项训练特征的列索引和列名 | 定义HDF5矩阵每一列的生物学含义，也是构建模态、数据类型和特征身份编码的依据 |
| `pretrain_input/mm10_1000_feature_matrix.h5` | 仅包含非全零bin的`float32`型`matrix`数据集，形状为`N x 288`，采用chunk和gzip压缩 | 模型训练实际读取的数值矩阵；不含坐标、原始区间字符串、零值统计和Pol II S5P |

其中，三个`pretrain_input/`文件共同构成正式预训练输入：HDF5提供数值，`feature_columns.tsv`解释列，`regions.tsv`解释行。三者不能单独重新排序；如果过滤或重建矩阵，必须同步更新区域文件中的`index`。

### 2.1 输入模态与实验重复

原始数据并不是通过模糊匹配或遍历目录自动推断，而是在`INPUT_SPECS`中明确登记模态名、固定文件路径和必要的 scaling factor 前缀。这样可以避免名称相似的文件被误读，也能保证每次运行的模态顺序一致。NFR 和 Nucleosome 与 ATAC 来自同一批重复实验，因此显式指定同一个 ATAC 样本前缀；其他模态直接由实验列名恢复样本名。

```python
@dataclass(frozen=True)
class InputSpec:
    name: str
    relative_path: str
    factor_prefix: str | None = None


INPUT_SPECS = (
    InputSpec("G4", "CPC_G4Seq.tsv"),
    InputSpec("ATAC", "ATAC/CPC_ATACpeak.tsv"),
    InputSpec(
        "NFR", "ATAC/CPC_Diff_NFR.tsv",
        "V6.5_CMDiff_D5_ATACseq",
    ),
    InputSpec(
        "Nucleosome", "ATAC/CPC_Diff_Nucleosome.tsv",
        "V6.5_CMDiff_D5_ATACseq",
    ),
    # 其余 TF、组蛋白修饰和 R-loop 以相同方式固定登记
)
```

`load_modalities()`严格按照该元组的顺序读取所有模态。返回值不是松散的 DataFrame 列表，而是同时保存模态名、来源文件、实验列、校正系数和清洗后 peak 表的`Modality`对象，便于后续验证每个结果来自哪个文件和哪组重复实验。

```python
def load_modalities(data_dir, scaling_lookup, chromosomes):
    return [
        load_modality(
            spec.name,
            data_dir / spec.relative_path,
            spec.factor_prefix,
            scaling_lookup,
            chromosomes,
        )
        for spec in INPUT_SPECS
    ]
```

scaling factor 文件采用`样本名>数值`格式，真正的系数位于`>`右侧。读取时先跳过空行和以`#`开头的说明行，再且仅在第一个`>`处分割，避免把整行或左侧样本名误当成数值：

```python
for raw_line in handle:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if ">" not in line:
        raise ValueError(f"Invalid scaling-factor line: {line}")
    sample, value = line.split(">", 1)
    factors[sample.strip()] = float(value.strip())
```

实验列名与 factor 样本名的对应在`scaling_factor_name()`中完成。普通实验列只删除末尾的`_reads`；NFR 和 Nucleosome 等使用共享重复实验的文件，则从列名提取`rep1`、`rep2`等编号，再与配置中的真实前缀组合。这样，`V6.5_CMDiff_D5_ATACseq_rep1_reads`和相应的 NFR/Nucleosome 列最终都会准确对应到`V6.5_CMDiff_D5_ATACseq_rep1`。

```python
def scaling_factor_name(column, factor_prefix):
    sample = column.removesuffix("_reads")
    if factor_prefix is None:
        return sample
    if "_rep" not in sample:
        raise ValueError(
            f"Cannot extract replicate number from column: {column}"
        )
    replicate = sample.rsplit("_rep", 1)[1]
    return f"{factor_prefix}_rep{replicate}"
```

每个文件的前三列固定解释为`chrom/start/end`，其余列全部视为重复实验，因而不会只读取第一个实验。factor 名称生成后先进行完整性检查；任何实验列无法匹配时立即报错，并在错误信息中列出原始列名，不允许静默使用 1.0：

```python
experiment_columns = [str(column) for column in frame.columns[3:]]
factor_names = [
    scaling_factor_name(column, factor_prefix)
    for column in experiment_columns
]
missing = [
    column
    for column, factor_name in zip(experiment_columns, factor_names)
    if factor_name not in scaling_lookup
]
if missing:
    raise ValueError(
        f"No scaling factor for {path.name}: {', '.join(missing)}"
    )
```

确认全部匹配后，每个 peak 的所有重复实验先分别乘自己的 scaling factor，再对校正值取算术平均：

```python
factors = [scaling_lookup[name] for name in factor_names]
experiments = frame[experiment_columns].apply(
    pd.to_numeric, errors="coerce"
).fillna(0.0).to_numpy(dtype=np.float64)

peak_values = (experiments * np.asarray(factors)).mean(axis=1)
```

相同`chrom/start/end`的重复区间保留最大校正信号：

```python
peaks = peaks.groupby(
    ["chrom", "start", "end"], as_index=False, sort=False
)["value"].max()
```

坐标在进入计算前还需要满足主染色体、非负起点和正长度三个条件。`pd.to_numeric(..., errors="coerce")`会先把无法解析的起止坐标变成缺失值并删除，随后再执行区间合法性筛选：

```python
peaks = peaks.dropna(subset=["start", "end"])
peaks[["start", "end"]] = peaks[["start", "end"]].astype(np.int64)
peaks = peaks.loc[
    peaks["chrom"].isin(allowed_chroms)
    & peaks["start"].ge(0)
    & peaks["end"].gt(peaks["start"])
]
```

因此负起点、`end <= start`或非主染色体记录不会进入bin映射。当前G4文件没有非数字坐标、负起点、`end == start`、`end < start`或超出主染色体长度的记录；被过滤的20条记录均来自非目标染色体，其中包括6条chrM、3条random contig和11条未定位contig。这些记录不是数值错误，而是不属于当前训练所定义的mm10主染色体集合。当前G4文件也没有重复坐标；如果其他模态出现重复坐标，上述`groupby(...).max()`会将其合并并保留最大校正信号。

### 2.2 bin 划分

默认 bin 大小为 1000 bp，可通过`--bin_size`修改。只处理`chr1-chr19、chrX、chrY`，排除`chrM`。每条染色体只生成完整 bin：

```python
bin_count = chrom_size // bin_size
starts = np.arange(bin_count, dtype=np.int64) * bin_size
```

整数除法会删除染色体末端不足一个完整 bin 的余数，避免在同一矩阵中混入不同长度样本。

peak和bin统一按0-based、左闭右开区间`[start, end)`解释（BED 的官方定义）。右端点`end`不属于区间，因此最后一个实际覆盖碱基是`end - 1`，最后命中的bin必须按下式确定：

```python
clipped_start = max(0, start)
clipped_end = min(end, region_end)
if clipped_end <= clipped_start:
    continue

first_bin = clipped_start // bin_size
last_bin = (clipped_end - 1) // bin_size
```

这里先将 peak 裁剪到当前染色体能够形成完整 bin 的范围；如果裁剪后没有正长度交集，直接跳过。随后用`clipped_end - 1`定位最后一个真实覆盖碱基。若不减 1，终点恰好为 bin 边界的 peak 会被错误分配到右侧 bin。

例如G4区间`chr1:153332665-153333000`只属于`chr1:153332000-153333000`，不会进入从153333000开始的右侧bin。当前数据中有16个G4终点恰好是1000的整数倍，逐条核对后均未错误进入右侧相邻bin；另有18个G4起点位于bin边界，因为左端点属于区间，所以它们从该边界对应的新bin开始计算。对于`end <= start`的异常记录，前面的合法性筛选会在映射前删除；对于超出完整 bin 区域的染色体末端 peak，`min(end, region_end)`只保留能够落入完整 bin 的部分，不会生成短 bin。

只要peak与某个bin存在至少1 bp交集，就会进入该bin。跨越边界的peak会依次映射到所有被覆盖bin，并在各bin中独立计算重叠长度：

```python
for bin_index in range(first_bin, last_bin + 1):
    overlap = overlap_length(
        peak_start, peak_end, bin_start, bin_end
    )
    self_ratio = overlap / (peak_end - peak_start)
```

当前有5,690个G4跨越至少两个1000 bp bin，其中大多数只是跨过bin边界，本身并不长于1000 bp；真正长度大于1000 bp的G4只有7个。程序不会把超长peak强行压缩到一个bin，而是按其与各bin的实际交集分别计算。

### 2.3 单模态特征

对每个模态和每个 bin 计算：

| 特征 | 定义 |
|---|---|
| `M_peak` | 与 bin 相交的 peak 中，校正信号最大的值 |
| `M_self_ratio` | 代表 peak 落入 bin 的碱基数 / 该 peak 完整长度 |
| `M_bin_ratio` | 模态 M 在 bin 内覆盖并集长度 / bin 长度 |
| `M_iv` | 所有相交 peak 的原始完整区间，仅写入核查 TSV |

代表 peak 的选择先比较信号，信号相同时选择`self_ratio`更高的 peak：

```python
if value > peak_values[bin_index] or (
    value == peak_values[bin_index]
    and self_ratio > self_ratios[bin_index]
):
    peak_values[bin_index] = value
    self_ratios[bin_index] = self_ratio
```

该条件也覆盖“一个 bin 内有两个或更多 peak”的情况：所有 peak 都会参与遍历，最大校正信号成为代表 peak；若信号完全相同，则选择落入该 bin 比例更高的 peak。这里不会把多个 peak 的信号相加，因为 peak 数量增多不应直接放大测量强度。

`peak`和覆盖比例表达不同信息。`peak`只取一个代表信号；`bin_ratio`先合并当前模态所有 peak 在 bin 内的区间，因此多个重叠 peak 不会重复贡献碱基数：

```python
merged_intervals = merge_intervals(raw_intervals, region_end)
covered_bases = covered_bases_per_bin(
    merged_intervals, bin_size, bin_count
)
bin_ratio = covered_bases.astype(np.float32) / float(bin_size)
```

区间合并函数使用`start <= previous_end`同时合并重叠区间和首尾相接的区间。首尾相接本身没有重复碱基，但合并后不改变总长度，并能简化后续双指针求交。负坐标、超过完整 bin 区域的终点和裁剪后为空的区间也在这里再次受到保护：

```python
clipped = sorted(
    (max(0, start), min(end, region_end))
    for start, end in intervals
    if min(end, region_end) > max(0, start)
)
for start, end in clipped[1:]:
    previous_start, previous_end = merged[-1]
    if start <= previous_end:
        merged[-1] = (previous_start, max(previous_end, end))
    else:
        merged.append((start, end))
```

当一个bin命中多个同模态peak时，`M_iv`记录全部原始区间，`M_bin_ratio`统计全部区间并集，但`M_peak`和`M_self_ratio`只对应选出的代表peak。以G4为例：

```text
G4_peak       = 校正信号最大的G4 peak值
G4_self_ratio = 该代表peak落入当前bin的比例
G4_bin_ratio  = 全部G4在bin内的覆盖并集 / bin长度
G4_iv         = 当前bin命中的全部G4原始区间
```

例如`chr1:13589000-13590000`同时命中`13589574-13589894`和`13589909-13590140`，输出的`G4_iv`保留两个区间，而`G4_peak=35.04`和`G4_self_ratio=1.00`属于最高信号peak，`G4_bin_ratio=0.41`则来自两个区间的总覆盖。

当前共有1,468个bin命中至少两个G4，其中1,415个命中2个、53个命中3个，单个bin最多命中3个。代表peak不一定是bin内覆盖最长的peak，例如`chr10:81430000-81431000`包含3个G4，最高信号peak只以较短部分进入当前bin，因此`G4_self_ratio=0.08`，而三个G4的联合覆盖由`G4_bin_ratio=0.64`表示。两个值不同不是冲突，而是分别描述代表信号和总体空间覆盖。

### 2.4 有向跨模态重叠

对于两个模态 A 和 B，先对各自区间取并集，再计算它们在 bin 内的共同覆盖碱基。两个方向共用分子，但分母不同：

```text
A_overlapping_B = shared(A, B) / covered(B)
B_overlapping_A = shared(A, B) / covered(A)
```

代码中的方向对应为：

```python
right_overlapping_left = np.divide(
    shared_bases, left.covered_bases,
    out=np.zeros(bin_count, dtype=np.float32),
    where=left.covered_bases > 0,
)
left_overlapping_right = np.divide(
    shared_bases, right.covered_bases,
    out=np.zeros(bin_count, dtype=np.float32),
    where=right.covered_bases > 0,
)
```

例如`CTCF_overlapping_G4`表示共同覆盖碱基占 G4 在 bin 内覆盖碱基的比例。分母模态不存在时结果为 0。

同一模态内部存在相互重叠的peak时，会先取区间并集，再参与`bin_ratio`和跨模态交集计算，因此同一碱基不会被重复计数。当前15,246个有效G4坐标中没有发现真实互相重叠的G4 pair；同一bin内出现多个G4时，它们是彼此分离的区间，或从相邻bin跨入当前bin。

```python
while left_index < len(left) and right_index < len(right):
    start = max(left[left_index][0], right[right_index][0])
    end = min(left[left_index][1], right[right_index][1])
    if end > start:
        shared.append((start, end))
    if left[left_index][1] <= right[right_index][1]:
        left_index += 1
    else:
        right_index += 1
```

最终的模态融合并不是把不同模态的原始 peak 行直接拼接，而是先为每条染色体分别计算16个`BinModality`，再为每一对模态生成两个有向比例，最后严格按模态分组的列顺序写入表格。每个模态块由3个自身特征和15个“其他模态相对本模态”的重叠特征构成：

```python
results = {
    modality.name: calculate_modality_for_bins(
        modality, chrom, bin_size, bin_count
    )
    for modality in modalities
}
pair_columns = calculate_pair_columns(
    modalities, results, bin_size, bin_count
)

def modality_feature_names(modality, all_modalities):
    return [
        f"{modality}_peak",
        f"{modality}_self_ratio",
        f"{modality}_bin_ratio",
        *[
            f"{other}_overlapping_{modality}"
            for other in all_modalities
            if other != modality
        ],
    ]
```

因此列总数固定为`16 x (3 + 15) = 288`。当某个模态在 bin 内不存在时，其`peak/self_ratio/bin_ratio`均由初始化数组保持为0；任何以该模态覆盖长度为分母的比例也通过`where=covered_bases > 0`保持为0，不会产生`NaN`或无穷值。原始区间列只写入核查 TSV，不进入288列训练矩阵。

### 2.5 舍入、非零筛选和输出

所有数值在统计零值和写入训练矩阵前统一保留两位小数：

```python
for column in numeric_features:
    data[column] = np.round(
        np.asarray(data[column], dtype=np.float32), 2
    )

zero_count = (numeric == 0).sum(axis=1).astype(np.int16)
nonzero_count = (numeric.shape[1] - zero_count).astype(np.int16)
```

因此绝对值小于 0.005 且舍入为 0 的值按零值处理。完整 TSV 保留所有 bin；训练 HDF5 只写入`nonzero_count > 0`的 bin。坐标和数值矩阵分开存储：

```text
data/preprocessed/pretrain_data/bin_1000bp/
├── mm10_1000_multimodal_chrall_all_bins.tsv.gz
├── mm10_1000_multimodal_chrall_nonallzero_bins.tsv.gz
├── validation_summary.tsv
├── column_manifest.tsv
├── chromosome_nonzero_heatmap.png
├── modality_cooccurrence_log2_enrichment.tsv
├── modality_cooccurrence_enrichment_heatmap.png
└── pretrain_input/
    ├── mm10_1000_regions.tsv
    ├── mm10_1000_feature_columns.tsv
    └── mm10_1000_feature_matrix.h5
```

HDF5 按染色体流式追加，默认 chunk 为`2048 x 288`。脚本不会把全基因组宽表一次性保留在内存中。

针对坐标边界和多peak情况的逐条核查结果集中保存在`bin_1000bp/audit/g4_coordinate_cases/`。其中`multi_g4_bins.tsv.gz`可对照多G4 bin的原始区间与实际`G4_iv`，`boundary_end_peaks.tsv.gz`可检查整数边界是否被重复分配，`invalid_coordinates.tsv`记录未进入训练集合的坐标，`all_peak_coordinate_flags.tsv.gz`则汇总每个G4的跨bin和长度标记。本次核查中，多G4映射不一致数和边界右侧泄漏数均为0。

## 3. 训练数据读取与归一化

实现文件：`pretrain/dataset.py`。

### 3.1 特征选择

当前预处理文件已经直接包含 288 个非 Pol II 特征。

```python
if any(term in name.lower() for term in terms):
    excluded_indices.append(source_index)
else:
    kept_indices.append(source_index)
```

### 3.2 训练集和验证集

全部样本按固定随机种子打乱，默认 3% 用作验证集，其余用于训练，不设置预训练测试集：

```python
shuffled = generator.permutation(row_count)
validation_count = max(1, int(round(row_count * validation_ratio)))
train = np.sort(shuffled[validation_count:])
validation = np.sort(shuffled[:validation_count])
```

实际索引保存到`split_indices.npz`，使断点恢复、模型比较和归一化重复使用同一划分。

### 3.3 peak归一化

`self_ratio`、`bin_ratio`和`overlapping`本身具有`[0,1]`含义，不再变换。每个`*_peak`列只用训练集非零值估计 99.5% 分位数`Q_f`。

默认方法：

```text
x'_f = clip(log(1 + max(x_f, 0)) / log(1 + Q_f), 0, 1)
```

对应代码：

```python
peaks = np.maximum(values[..., peak_mask], 0.0)
peaks = np.log1p(peaks) / np.log1p(scales)
values[..., peak_mask] = np.clip(peaks, 0.0, 1.0)
```

`log_quantile`适合长尾测序信号，降低极端 peak 对重建损失的支配。`linear_quantile`使用`x/Q_f`，更接近保留线性倍数关系。两者都会把高于分位数的值截断为 1。

归一化参数保存到`feature_normalization.npz`，所有下游模型必须复用同一`peak_mask`、`peak_scales`和`feature_names`。

### 3.4 HDF5读取优化

`H5BinDataset`在每个 DataLoader worker 内延迟打开 HDF5，避免跨进程共享文件句柄。批量读取时先对行号去重；若行号集中，则读取连续切片，减少随机 I/O。

`BlockShuffleSampler`以 HDF5 chunk 大小为单位打乱块顺序，同时保留块内局部连续性。这仍然改变每个 epoch 的样本顺序，但比完全随机逐行读取更适合压缩 HDF5。

## 4. 输入token编码

实现文件：`pretrain/model.py`。

对于 batch 大小 B，输入数值张量为：

```text
values: B x 288
```

每一列被转换为一个 feature token。以`mae_bin_large`为例，token维度 D=512：

```text
B x 288 -> B x 288 x 512
```

一个token由五部分相加：

```text
token_f = value_embedding(x_f)
        + feature_identity_embedding(f)
        + denominator_modality_embedding(f)
        + numerator_modality_embedding(f)
        + data_form_embedding(f)
```

代码入口为：

```python
ids = torch.arange(self.feature_count, device=values.device)
x = self.feature_encoder(ids).unsqueeze(0) + self.value_encoder(values)
if self.typed_features:
    x = x + self.feature_type_embeddings().unsqueeze(0)
return self.input_dropout(self.input_norm(x))
```

各编码的作用如下：

| 编码 | 作用 | 是否保留 |
|---|---|---|
| 数值编码 | 表示当前 bin 中的实际数值 | 保留 |
| 特征身份编码 | 区分具体的288列 | 保留 |
| 分母模态编码 | 表示比例所描述的主体模态 | 保留 |
| 分子模态编码 | 表示重叠特征中的覆盖模态 | 保留 |
| 数据形式编码 | 区分 peak、self ratio、bin ratio、overlap | 保留 |
| 列序号位置编码 | 仅反映表格顺序，没有DNA距离意义 | 不使用 |

对于`CTCF_overlapping_G4`：

```text
denominator/primary = G4
numerator/partner   = CTCF
data form           = overlap
```

特征身份编码不能被模态编码替代，因为同一模态内部仍有 peak、两个比例及多个方向不同的重叠特征。

## 5. 掩码策略

### 5.1 随机特征掩码

`random`对每个样本生成独立随机排序，保留前`int(F*(1-r))`个token：

```python
ids_shuffle = torch.rand(batch, length, device=x.device).argsort(dim=1)
len_keep = int(length * (1 - mask_ratio))
```

默认`r=0.4`时隐藏约40%的独立特征，适合学习任意局部缺失值恢复。

### 5.2 整模态掩码

`modality`先随机选择若干模态，再隐藏所有与这些模态相关的列，包括：

- 该模态自身的`peak/self_ratio/bin_ratio`；
- 该模态作为重叠比例分母的列；
- 该模态作为重叠比例分子的列。

```python
mask = torch.gather(selected, 1, primary)
mask |= (partner < self.modality_count) & torch.gather(
    selected, 1, partner.clamp(max=self.modality_count - 1)
)
```

由于模态相关列存在集合重叠，实际遮盖比例不一定正好等于40%。`modality_mask_plan()`枚举被选模态数，选择实际遮盖列数最接近目标比例的方案，并把真实比例写入日志。

该策略避免模型从尚未遮盖的同模态派生列直接恢复答案，更接近“由其他组学推断整个缺失模态”的任务。

## 6. 编码器、Decoder和张量流

以默认`mae_bin_large`为例：

```text
输入                       B x 288
feature tokens             B x 288 x 512
掩码后可见 tokens           B x K x 512
加入 [BIN]                 B x (K+1) x 512
12层 Encoder               B x (K+1) x 512
投影到 Decoder 维度         B x (K+1) x 256
补回 [MASK] 并恢复列顺序     B x 289 x 256
4层 Decoder                B x 289 x 256
逐token数值头               B x 288
```

`[BIN]`位于序列首位，用于汇总当前区域全部可见特征。它不是额外的基因组区间，也不表示多个bin组成的unit。

Decoder先用`ids_restore`把可见token和`[MASK]` token放回原始特征列位置，再重新加入特征身份、模态角色和数据形式编码：

```python
x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
x_ = torch.gather(
    x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
)
x = torch.cat([x[:, :1, :], x_], dim=1)
```

共享数值头对每个位置输出一个连续预测值，不设置模态专属Decoder。

## 7. 损失、优化和评价

损失只作用于人工遮盖位置：

```python
loss = (((pred.float() - target) ** 2) * mask).sum() / mask.sum()
```

未遮盖token不贡献重建损失。默认关闭`norm_pix_loss`，因为逐样本再次标准化会破坏已定义好的比例尺度和逐列peak归一化。

训练同时记录遮盖位置MSE和MAE。

优化器为AdamW，学习率按有效batch线性缩放：

```python
effective_batch = batch_size * accum_iter * world_size
lr = base_lr * effective_batch / 256
```

前`warmup_epochs`线性升温，之后使用余弦衰减到`min_lr`。支持梯度累积、梯度裁剪、bfloat16/float16、断点恢复、DDP和可选`torch.compile`。

## 8. 运行、日志和模型保存

```bash
cd /mnt/afan/G4RegFormer
bash pretrain/train_pretrain.sh
```

直接运行示例：

```bash
/home/afan/anaconda3/envs/single_cell/bin/python pretrain/main_pretrain.py \
  --data data/preprocessed/pretrain_data/bin_1000bp/pretrain_input/mm10_1000_feature_matrix.h5 \
  --model mae_bin_large \
  --mask-strategy modality \
  --mask-ratio 0.4 \
  --valid-ratio 0.03 \
  --peak-normalization log_quantile
```

每次运行创建`时间戳-模型名`目录，主要文件包括：

```text
pretrain/output/<timestamp>-mae_bin_large/
├── args.json
├── run.log
├── events.out.tfevents.*
├── features.tsv
├── feature_normalization.npz
├── split_indices.npz
├── split_summary.tsv
└── checkpoint-<epoch>.pth
```

`features.tsv`记录模型索引、原始列索引、模态角色和数据形式；checkpoint保存模型、优化器、scaler、epoch和参数，可用于断点恢复及下游初始化。

## 9. 重建可视化

实现文件：`pretrain/visualization/pretrain_visualize.py`。

可视化脚本读取指定checkpoint及其归一化文件，选择若干真实bin，重新施加掩码并保存：

- 完整逐特征真实值与预测值；
- 每个bin的MSE、RMSE和MAE；
- 原始输入、掩码重建和遮盖误差热图；
- 真实值与预测值散点图；
- 每个bin误差最大的若干特征对比图。

```bash
/home/afan/anaconda3/envs/single_cell/bin/python \
  pretrain/visualization/pretrain_visualize.py \
  --checkpoint pretrain/output/<run>/checkpoint-19.pth \
  --num-bins 6 \
  --mask-strategy modality
```

热图中的“重建输入”只用预测值替换人工遮盖位置，未遮盖位置保留真实值。因此它用于检查模型能否从其他模态恢复缺失信息，而不是生成新的全基因组测量数据。
