import json
import os


for fn in os.listdir("raw"):
    print(f"preparing {fn}")
    json_fn: dict = json.load(open(os.path.join("raw", fn)))

    parsed_json_fn = {}
    for key in json_fn.keys():
        print(key)
        obj = json_fn[key]

        try:
            parsed_json_fn[key] = {
                "response": json.loads(obj["response"]) if isinstance(obj["response"], str) else {},
                "inference_time": obj["time"]
            }
        except:
            parsed_json_fn[key] = {
                "response": {},
                "inference_time": obj["time"]
            }

    with open(f"parsed/vllm_{fn}", "w") as f:
        json.dump(parsed_json_fn, f, indent=4)