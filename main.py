import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from transformers.utils import quantization_config
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import trackio

def format_data(sample):
  system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

  return {
    "images": [sample["image"]],
    "messages": [
      {
        "role": "system",
        "content": [{"type": "text", "text": system_message}],
      },
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "image": sample["image"],
          },
          {
            "type": "text",
            "text": sample["query"],
          },
        ],
      },
      {
        "role": "assistant",
        "content": [{"type": "text", "text": sample["label"][0]}],
      },
    ],
  }


def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
  # Prepare the text input by applying the chat template
  text_input = processor.apply_chat_template(
      sample['messages'][1:2],  # Use the sample without the system message
      tokenize=False,
      add_generation_prompt=False
  )

  # Process the visual input from the sample
  image_inputs, _ = process_vision_info(sample['messages'])

  # Prepare the inputs for the model
  model_inputs = processor(
      text=[text_input],
      images=image_inputs,
      return_tensors="pt",
  ).to(device)  # Move inputs to the specified device

  # Generate text with the model
  generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

  # Trim the generated ids to remove the input ids
  trimmed_generated_ids = [
      out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
  ]

  # Decode the output text
  output_text = processor.batch_decode(
      trimmed_generated_ids,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=False
  )

  return output_text[0]  # Return the first decoded output text

def train():
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Training requires GPU.")
  print(f"CUDA available: {torch.cuda.is_available()}")
  print(f"CUDA device count: {torch.cuda.device_count()}")
  print(f"Current device: {torch.cuda.current_device()}")
  print(f"Device name: {torch.cuda.get_device_name()}")
  print(f"CUDA available: {torch.cuda.is_available()}")
  print(f"Device count: {torch.cuda.device_count()}")
  
  train_dataset, eval_dataset, test_dataset = load_dataset("chartqa", split=['train[:10%]', 'val[:10%]', 'test[:10%]'])

  train_dataset = [format_data(sample) for sample in train_dataset]
  eval_dataset = [format_data(sample) for sample in eval_dataset]
  test_dataset = [format_data(sample) for sample in test_dataset]

  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
  )
  local_model_path = "./Qwen2-VL-7B"
  processor = Qwen2VLProcessor.from_pretrained(local_model_path)
  model = Qwen2VLForConditionalGeneration.from_pretrained(
    local_model_path,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    pad_token_id=processor.tokenizer.pad_token_id
  )
  print(f"Model device: {model.device}")

  print("Tokenizer special tokens检查:")
  print(f"Pad token ID: {processor.tokenizer.pad_token_id}")
  print(f"Model config Pad token ID: {model.config.pad_token_id}")

  peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
  )
  # peft_model = get_peft_model(model, peft_config)
  # peft_model.print_trainable_parameters()

  training_args = SFTConfig(
    output_dir="qwen2-7b-instruct-trl-sft-ChartQA",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    gradient_accumulation_steps=16,  # Steps to accumulate gradients
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    max_length=None,
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Steps interval for saving
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub
    report_to="trackio",  # Reporting tool for tracking metrics
  )

  trackio.init(
    project="qwen2-7b-instruct-trl-sft-ChartQA",
    name="qwen2-7b-instruct-trl-sft-ChartQA",
    config=training_args,
    space_id=training_args.output_dir + "-trackio"
  )

  trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
  )

  trainer.train()
  trainer.save_model(training_args.output_dir)

  trackio.finish()
  
  # processor = Qwen2VLProcessor.from_pretrained(local_model_path, trust_remote_code=True, use_fast=False)
  # output = generate_text_from_sample(model, processor, train_dataset[0])
  # print(train_dataset[0])
  # print(output)

def evaluate():
  train_dataset, eval_dataset, test_dataset = load_dataset("chartqa", split=['train[:1%]', 'val[:1%]', 'test[:1%]'])

  train_dataset = [format_data(sample) for sample in train_dataset]
  # eval_dataset = [format_data(sample) for sample in eval_dataset]
  # test_dataset = [format_data(sample) for sample in test_dataset]
  
  local_model_path = "./Qwen2-VL-7B"
  model = Qwen2VLForConditionalGeneration.from_pretrained(
    local_model_path,
    device_map="auto",
    dtype=torch.bfloat16,
  )
  processor = Qwen2VLProcessor.from_pretrained(local_model_path)

  adapter_path = "qwen2-7b-instruct-trl-sft-ChartQA"
  model.load_adapter(adapter_path)
  model.set_adapter('default')

  # from peft import PeftModel
  # model = PeftModel.from_pretrained(model, adapter_path)

  # model.eval()

  output = generate_text_from_sample(model, processor, train_dataset[0])
  print(train_dataset[0])
  print('--------------------------------')
  print(output)
  print('--------------------------------')

if __name__ == "__main__":
  # train()
  evaluate()