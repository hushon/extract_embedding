# extract_embedding
## Introduction
A library for extracting embeddings.

## Installation
```
pip install git+https://github.com/hushon/extract_embedding.git
```

## Examples
```python
from extract_embedding import ExtractEmbedding

def post_process_fn(data):
    return data.mean(dim=0)  # Example post-processing function

with ExtractEmbedding(layers=[model.fc, model.conv], extract_input=True, enable_grad=False, apply_func=post_process_fn) as extractor:
    output = model(input)
print(extractor.extracted_data[0].shape)  # First layer's input (after post-processing)
print(extractor.extracted_data[1].shape)  # Second layer's input (after post-processing)
```