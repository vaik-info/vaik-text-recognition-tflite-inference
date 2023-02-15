# vaik-text-recognition-tflite-inference

Inference by text recognition Tflite model


## Install

``` shell
pip install git+https://github.com/vaik-info/vaik-text-recognition-tflite-inference.git
```

## Usage

### Example
```python
import numpy as np
from PIL import Image
from vaik_text_recognition_tflite_inference.tflite_model import TfliteModel

classes = TfliteModel.char_json_read('/home/kentaro/Github/vaik-text-recognition-tflite-trainer/data/jpn_character.json')
model_path = '/home/kentaro/.vaik_text_recognition_pb_exporter/model.tflite'
model = TfliteModel(model_path, classes)

image1 = np.asarray(Image.open("/home/kentaro/Desktop/images/いわき_0333.png").convert('RGB'))

output, raw_pred = model.inference(image1)
```


#### Output

- output

```text
{'text': 'いわき', 'classes': [113, 155, 118], 'scores': 0.9999999991409891}
```