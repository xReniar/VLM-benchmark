from vllm import LLM
from PIL import Image


def main():
    vlm = LLM(
        model="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        quantization="AWQ",
        dtype="float16"
    )

    img = Image.open("data/kie/images/0.png")
    prompt = "perform"

    conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {
            "role": "user",
            "content": [
                { "type": "image_pil", "image_pil": img },
                { "type": "text", "text": prompt }
            ],
        },
    ]

    outputs = vlm.chat(conversation)


    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

if __name__ == "__main__":
    main()