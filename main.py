from vlm import VLM
from dataset import Task, SROIE, DocILE


#x = VLM("SmolVLM")
#response = x.predict("datasets/kie/images/0.png", "How much do I need to pay?")

for data in DocILE(task=Task.OCR, split="train"):
    image_path = data["image_path"]
    for entity in data["fields"]:
        print(entity)

'''
for data in DocILE(task=Task.KIE, split="train"):
    print(data)
'''