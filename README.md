# Git-RSCLIP

**Git-RSCLIP** is pre-trained on the Git-10M dataset (a global-scale remote sensing image-text pair dataset, consisting of **10 million image-text pairs**), available at [Github](https://github.com/Chen-Yang-Liu/Text2Earth)

The paper has been published in **IEEE Geoscience and Remote Sensing Magazine**: [IEEE](https://ieeexplore.ieee.org/document/10591792) | [ArXiv](https://arxiv.org/pdf/2501.00895)

## News 🔥
✅ 2025.06.01: **Git-RSCLIP** series downloads exceeded **60,000** times 🔥

## Model DownLoad Link
- **Large version**:[[🤗 Huggingface](https://huggingface.co/lcybuaa/Git-RSCLIP) | [🌊 Modelscope](https://modelscope.cn/models/lcybuaa1111/Git-RSCLIP)]

- **Base version**: [[🤗 Huggingface](https://huggingface.co/lcybuaa/Git-RSCLIP-base) | [🌊 Modelscope](https://modelscope.cn/models/lcybuaa1111/Git-RSCLIP-base)]

## Intended uses & limitations

You can use the raw model for tasks like zero-shot image classification and text-image retrieval.


### How to use

#### Use Git-RSCLIP to get image features

```python
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("lcybuaa/Git-RSCLIP")
processor = AutoProcessor.from_pretrained("lcybuaa/Git-RSCLIP")

url = "https://github.com/Chen-Yang-Liu/PromptCC/blob/main/Example/B/train_000051.png?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
  image_features = model.get_image_features(**inputs)
```


#### zero-shot image classification:

```python
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("lcybuaa/Git-RSCLIP")
processor = AutoProcessor.from_pretrained("lcybuaa/Git-RSCLIP")

url = "https://github.com/Chen-Yang-Liu/PromptCC/blob/main/Example/B/train_000051.png?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a remote sensing image of river", "a remote sensing image of houses and roads"]
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image) # these are the probabilities
top5_indices = torch.argsort(probs, descending=True)[:, :5].cpu().numpy()
top1_indices = top5_indices[:, 0]
print(f"the image 0 is '{top1_indices[0]}'")
```

For more code examples, we refer to the [documentation](https://huggingface.co/transformers/main/model_doc/siglip.html#).


## Training procedure

### Training data

Git-RSCLIP is pre-trained on the Git-10M dataset (a global-scale remote sensing image-text pair dataset, consisting of 10 million image-text pairs) [(Liu et al., 2024)](https://github.com/chen-yang-liu/Text2Earth).

### Preprocessing

Images are resized/rescaled to the same resolution (256x256) and normalized across the RGB channels with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).

Texts are tokenized and padded to the same length (64 tokens).


## Evaluation results

Evaluation of Git-RSCLIP compared to other CLIP is shown below (taken from the paper).

<img src="https://github.com/Chen-Yang-Liu/Text2Earth/blob/main/images/Git-RSCLIP.png?raw=true"
alt="drawing" width="1000"/>

### BibTeX entry and citation info

```bibtex
@ARTICLE{10988859,
  author={Liu, Chenyang and Chen, Keyan and Zhao, Rui and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Geoscience and Remote Sensing Magazine}, 
  title={Text2Earth: Unlocking text-driven remote sensing image generation with a global-scale dataset and a foundation model}, 
  year={2025},
  volume={},
  number={},
  pages={2-23},
  doi={10.1109/MGRS.2025.3560455}}
```
