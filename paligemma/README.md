# PaliGemma Implementation

This repository contains a complete implementation of **PaliGemma**, a vision-language model architecture. The implementation is based on the original research paper and uses official pretrained weights where applicable. It supports fine-tuning on custom datasets and includes efficient training loops, adapter-based LoRA integration, and performance tracking.

---

##  Features

- ✅ Full model architecture built from scratch based on the PaliGemma paper
- ✅ Efficient training using custom training loops, learning rate schedulers, and configurable arguments
- ✅ Tested and fine-tuned on custom datasets with measurable performance gain (e.g., +10%)

---

## Model Overview

**PaliGemma** is a powerful vision-language model that combines image and text understanding. It uses:
- A vision encoder (e.g., ViT or Resampler)
- A text decoder (Gemma family)
- Multi-modal token fusion

---

## upcoming :
- will be adding a adapter for lora using peft and all as that is the only thing left and has been bugging me 
