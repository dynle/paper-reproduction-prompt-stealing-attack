# Prompt Stealing Attacks Against Text-to-Image Generation Models

[![hugging](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/vera365/lexica_dataset)
[![arXiv: paper](https://img.shields.io/badge/arXiv-paper-red.svg)](https://arxiv.org/abs/2302.09923)
[![license](https://img.shields.io/badge/License-CC_BY_4.0/MIT-blue)](#license)

This is the official implementation of the USENIX 2024 paper [Prompt Stealing Attacks Against Text-to-Image Generation Models](https://arxiv.org/abs/2302.09923).

## LexicaDataset

LexicaDataset is a large-scale text-to-image prompt dataset containing **61,467 prompt-image pairs** collected from [Lexica](https://lexica.art/). All prompts are curated by real users and images are generated by Stable Diffusion.

LexicaDataset is available at [🤗 Hugging Face Datasets](https://huggingface.co/datasets/vera365/lexica_dataset).

**Load LexicaDataset**

You can use the Hugging Face [`Datasets`](https://huggingface.co/docs/datasets/quickstart) library to easily load prompts and images from LexicaDataset.

```python
import numpy as np
from datasets import load_dataset

trainset = load_dataset('vera365/lexica_dataset', split='train')
testset  = load_dataset('vera365/lexica_dataset', split='test')
```

**Metadata Schema**

`trainset` and `testset` share the same schema.

| Column              | Type       | Description                                                  |
| :------------------ | :--------- | :----------------------------------------------------------- |
| `image`             | `image`    | The generated image                                          |
| `prompt`            | `string`   | The text prompt used to generate this image                  |
| `id`                | `string`   | Image UUID                                                   |
| `promptid`          | `string`   | Prompt UUID                                                  |
| `width`             | `uint16`   | Image width                                                  |
| `height`            | `uint16`   | Image height                                                 |
| `seed`              | `uint32`   | Random seed used to generate this image.                     |
| `grid`              | `bool`     | Whether the image is composed of multiple smaller images arranged in a grid |
| `model`             | `string`   | Model used to generate the image                             |
| `nsfw`              | `string`   | Whether the image is NSFW                                    |
| `subject`           | `string`   | the subject/object depicted in the image, extracted from the prompt |
| `modifier10`        | `sequence` | Modifiers in the prompt that appear more than 10 times in the whole dataset. We regard them as labels to train the modifier detector |
| `modifier10_vector` | `sequence` | One-hot vector of `modifier10`                               |


## Code

### Setup the Environment

The following code are run and tested on A100 GPU.

The environment requirement:
> cuda toolkit 11.7, python 3.8, pytorch 1.12.0a0+8a1a93a

You can access it from the official docker image provided by Nvidia: `nvcr.io/nvidia/pytorch:22.05-py3`
After building the container, install necessary packages:

```
git clone https://github.com/verazuo/prompt-stealing-attack.git
cd prompt-stealing-attack
pip install -r requirements.txt
```

### Usage of PromptStealer

We provide a script `eval_PromptStealer.py` for easy use.
To run PromptStealer, first create dir `output/PS_ckpt`, then download two modules/checkpoints of PromptStealer in `output/PS_ckpt/`.

| Module            |                          Checkpoint                          |
| ----------------- | :----------------------------------------------------------: |
| subject generator | <a href="https://drive.google.com/file/d/1OO8fJrsoIR1qH2Ni2oint4bYciG5y8Ma/view?usp=drive_link">Download</a> |
| modifier detector | <a href="https://drive.google.com/file/d/1JmhAPzBImiJVw4pnTLa2daBOhNDM_oGc/view?usp=drive_link">Download</a> |


2. Run PromptStealer

```
python eval_PromptStealer.py
Loading CLIP model...
Dataset({
    features: ['image', 'prompt', 'id', 'promptid', 'width', 'height', 'seed', 'grid', 'model', 'nsfw', 'subject', 'modifier10', 'modifier10_vector'],
    num_rows: 12294
})
Return text: prompt


PromptStealer init...
...
metric,pred
semantic_sim,0.6999
modifier_sim,0.4477
```

3. To evaluate the similarity between target images and stolen images. Use [Stable Diffusion](https://github.com/CompVis/stable-diffusion) (`sd-v1-4.ckpt`) to generate stolen images (see [scripts/txt2img.py](https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py) for details).
Then, calculate the average image/pixel similarity via functions `get_image_similarity()`/`get_pixel_mse()` in `utils.py`.

### Train PromptStealer

1. Train the subject generator

```
nohup python train_subject_generator.py  > train_subject_generator.log & 
```

2. Train the modifier detector

Download the pre-trained Tresnet-L model.
```
cd output; mkdir pretrained_ckpt; cd pretrained_ckpt
wget -O tresnet_l.pth https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth 
```

Then, train the modifier detector
```
nohup python train_modifier_detector.py > train_modifier_detector.log & 

... ...
             metric      pred
0           semantic    0.6999
1           modifier    0.4477
```

## Comments

The implementation of the subject generator is adopted from [BLIP](https://github.com/salesforce/BLIP) and the modifier detector is from [ML_Decoder](https://github.com/Alibaba-MIIL/ML_Decoder).
The category of modifiers is from [CLIP_Interrogator](https://github.com/pharmapsychotic/clip-interrogator).
We use `sd-v1-4.ckpt` in [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for image generation.
Thanks for open-sourcing!


## Ethics & Disclosure

According to the [terms and conditions of Lexica](https://lexica.art/terms), images on the website are available under the Creative Commons Noncommercial 4.0 Attribution International License. We strictly followed Lexica’s Terms and Conditions, utilized only the official Lexica API for data retrieval, and disclosed our research to Lexica. We also responsibly disclosed our findings to related prompt marketplaces.

## License

LexicaDataset is available under the [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). The code in this repository is available under the [MIT License](./LICENSE).

**Note, the code is intended for research purposes only. Any misuse is strictly prohibited.**

## Citation

If you find this useful in your research, please consider citing:

```bibtex
@inproceedings{SQBZ24,
  author = {Xinyue Shen and Yiting Qu and Michael Backes and Yang Zhang},
  title = {{Prompt Stealing Attacks Against Text-to-Image Generation Models}},
  booktitle = {{USENIX Security Symposium (USENIX Security)}},
  publisher = {USENIX},
  year = {2024}
}
```
