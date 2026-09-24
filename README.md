# G4RegFormer

G4RegFormer 使用多组学 peak、区间覆盖关系和 G4 扰动信息学习转录调控表示。项目按数据、预处理、预训练、下游任务和方法记录五部分组织。

```text
G4RegFormer/
├── data/
│   ├── raw_multiomics_data/
│   ├── annotation/
│   ├── reference/
│   ├── preprocessed/pretrain_data/bin_1000bp/pretrain_input/
│   └── downstream/
│       ├── G4RegCRE/G4_1000bp_Reg_CRE_ATACPeak/training_data/
│       └── CRERegGene/CRE_500kb_Reg_Gene_mm10/train_data/
├── preprocess/
│   ├── preprocess_pretrain_bin.py
│   ├── preprocess_downstream_G4RegCRE.py
│   └── preprocess_downstream_CRERegGene.py
├── pretrain/
│   ├── dataset.py, model.py, engine.py, main_pretrain.py
│   ├── output/
│   └── visualization/
│       ├── pretrain_visualize.py
│       └── visualization_output/
├── downstream/
│   ├── 4RegCRE/
│   └── CRERegGene/
└── Notes/
    ├── Pretrain.md
    ├── downstream_G4RegCRE.md
    └── downstream_CRERegGene.md
```

## Data preprocessing

```bash
/home/afan/anaconda3/bin/python preprocess/preprocess_pretrain_bin.py --bin-size 1000

/home/afan/anaconda3/bin/python preprocess/preprocess_downstream_G4RegCRE.py \
  --control-dir <control_multiomics> \
  --perturbed-dir <perturbed_multiomics> \
  --hic-interactions <loops.bedpe.gz>

/home/afan/anaconda3/bin/python preprocess/preprocess_downstream_CRERegGene.py
```

CRERegGene 会优先复用已存在的 G4 区域文件；增加 `--rebuild-g4-regions` 时需要先提供正式 CT-TADB 边界结果。

## Training

```bash
bash pretrain/train_pretrain.sh
bash downstream/4RegCRE/train.sh
bash downstream/CRERegGene/train.sh
```

详细的数据定义、模型结构、损失和结果解释见 `Notes/`。训练输出只写入各任务自己的 `output/`，预训练可视化结果只写入 `pretrain/visualization/visualization_output/`。
