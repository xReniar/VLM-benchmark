from vlm import VLM
from dataset import Task, SROIE, DocILE


x = VLM("Qwen2.5-VL")
response = x.predict("data/kie/images/0.png", "How much do I need to pay?")

print(response)

'''
for data in DocILE(task=Task.OCR, split="train"):
    image_path = data["image_path"]
    for entity in data["fields"]:
        print(entity)
'''

'''
for data in DocILE(task=Task.KIE, split="train"):
    print(data)
'''