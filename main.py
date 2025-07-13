from vlm import VLM


x = VLM("SmolVLM")
response = x.predict("datasets/kie/images/0.png", "How much do I need to pay?")

print(response)