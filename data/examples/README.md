# Data Examples

This directory illustrates the expected input structure.

## Directory layout (Google Drive)

```
54816/
├── 24_chromosomes_object/
│   ├── JEPG/          # ~5,000 metaphase images (JPEG)
│   └── annotations/   # Pascal VOC XML annotations
├── single_chromosomes_object/
│   ├── JEPG/          # Single chromosome crops
│   └── annotations/   # One object per XML
├── normal.csv
├── number_abnormalities.csv
└── structural_abnormalities.csv
```

## Chromosome class labels (24 classes)

`A1 A2 A3 B4 B5 C6 C7 C8 C9 C10 C11 C12 D13 D14 D15 E16 E17 E18 F19 F20 G21 G22 X Y`
