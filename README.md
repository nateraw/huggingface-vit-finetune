# huggingface-vit-finetune

Huggingface does images now!

Well...they will soon. For now we gotta install `transformers` from master. 

```
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers.git@master --upgrade
python run.py
```

## Using trained models w/ `transformers`

Currently, the following models are available:
  - nateraw/vit-base-patch16-224-cifar10

```python
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png'
image = Image.open(requests.get(url, stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
preds = outputs.logits.argmax(dim=1)

classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]
classes[preds[0]]
```