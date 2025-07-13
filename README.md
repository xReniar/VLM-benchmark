# VLM Benchmark

## Setup environment
The following instruction setup the environment using `conda`:
```
conda create -n benchmark python=3.11
conda activate benchmark
pip3 install -r requirements.txt --no-cache-dir
```

## Check configurations files
- There are 2 configurations files that need to be checked before running experiments and they are `benchmark.yaml` and `models.yaml`.
- The `models.yaml` specifies what models are going to be used for inference. The structure is shown below:
```yaml
"model_name":
    model_id: ""
    type: ""
    parameters:
```
- `model_name`: name of the model you want to use
- `model_id`: huggingface model id
- `type`: can be `ImageText2Text` or `Vision2Text`, for custom types check `gemma3n.py` and others in the `./vlm/models/*` folder.
- `parameters`: this field is optional, but if present add at least one parameter (e.g `top_p`, `temperature`, etc..)