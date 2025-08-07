from unsloth import FastVisionModel
from training.dataset import SROIE, Data, DocILE
from PIL import Image
import json


system_message = """You are a highly advanced Vision Language Model (VLM), specialized in extracting visual data. 
Your task is to process and extract meaningful insights from images, leveraging multimodal understanding
to provide accurate and contextually relevant information."""


def format_data(sample: Data):
    global system_message
    pil_image = Image.open(sample.image_path)

    field_names = set([entity.label for entity in sample.entities])
    output_format = {field: ".." for field in field_names}

    prompt = "Extract the following {fields} from the above document. If a field is not present, return ''. Return the output in a valid JSON format like {output_format}" \
        .format(
            fields = list(field_names),
            output_format = output_format
        )
    
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                { "type": "image", "image": pil_image },
                { "type": "text", "text": prompt }
            ]
        },
        {
            "role": "assistant",
            "content": [{ 
                "type": "text",
                "text": json.dumps(sample.to_json("kie"))
            }]
        }
    ]

    return { "messages": conversation }

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

train_dataset = [format_data(sample) for sample in DocILE(tasks=["kie"], split="train")]
test_dataset = [format_data(sample) for sample in DocILE(tasks=["kie"], split="test")]


from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = train_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 60,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "training/single_test/qwen2.5-vl_docile/outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)

trainer_stats = trainer.train()
model.save_pretrained("training/single_test/qwen2.5-vl_docile/lora_model")  # Local saving
tokenizer.save_pretrained("training/single_test/qwen2.5-vl_docile/lora_model")