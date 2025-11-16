
# SwinV2-SF: A Parallel Polyp Segmentation Architecture Leveraging Swin Transformer V2 and Spatial-Frequency Dual-Domain Attention

Official code repository for: SwinV2-SF: A Parallel Polyp Segmentation Architecture Leveraging Swin Transformer V2 and Spatial-Frequency Dual-Domain Attention
## 1. Overview

### 1.1 Abstract
We propose SwinV2-SF, a novel hybrid neural network for colorectal polyp segmentation. This architecture integrates the spatial-frequency dual-domain network SFUNet with Swin Transformer V2, achieving semantic alignment between frequency and spatial domain features to enable holistic consideration of polyp features.

### 1.2 Architecture

## 2. Usage

### 2.1 Preparation

### 2.2 Examples of Produced Segmentation Maps

## 3. License


## 4. Citation

## 5. Acknowledgements

This work makes use of data from the Kvasir-SEG dataset, available at https://datasets.simula.no/kvasir-seg/.


This work makes use of data from the ETIS-LaribDB dataset, available at https://polyp.grand-challenge.org/ETISLarib/. 

Results are obtained using ImageNet pre-trained weights for the SwinV2 Encoder system, available at [SwinV2 Encoder Weights](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth)

This repository includes code from the following sources:

[FCB-Former](https://github.com/ESandML/FCBFormer)

[SwinV2 Encoder](https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/swin_transformer_v2.py)

[FCB-SwinV2](https://github.com/KerrFitzgerald/Polyp_FCB-SwinV2Transformer)

## 6. Additional information

Contact: 13504296944d@163.com
