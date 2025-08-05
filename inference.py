from vlm import VLM


model = VLM("SmolVLM2-500M")


print(model.predict("data/kie/images/0.png", "How much do I have to pay?"))