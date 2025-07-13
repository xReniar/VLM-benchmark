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
- The `models.yaml` specifies what models are going to be used for inference. An example structure is shown below:
```yaml
SmolVLM:
    model_id: HuggingFaceTB/SmolVLM-500M-Instruct
    type: Vision2Seq
    parameters:
        max_new_tokens: # this is None so the default hugginface value will be used
        repetition_penalty:
        temperature: 0.7
        top_k: 2
        top_p: 0.6
```
- `model_name`: name of the model you want to use (can be anything)
- `model_id`: huggingface model id
- `type`: can be `ImageText2Text` or `Vision2Seq`, for custom types check `gemma3n.py` and others in the `./vlm/models/*` folder.
- `parameters`: this field is optional, but if present add at least one parameter (e.g `top_p`, `temperature`, etc..)