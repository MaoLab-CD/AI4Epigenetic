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
