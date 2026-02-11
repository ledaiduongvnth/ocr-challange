import torch 
from datasets import load_dataset
from transformers import LightOnOcrProcessor
from transformers import LightOnOcrForConditionalGeneration
from torch.utils.data import Dataset
from pdf2image import convert_from_path
from PIL import Image
import os
import shutil
from transformers import TrainingArguments
from transformers import Trainer
from torch.utils.data import Subset
from jiwer import cer, wer
import torch
from tqdm import tqdm

finevision_subset ="bill_2"
model_id = "lightonai/LightOnOCR-2-1B" # or any of the above checkpoints

device = "cuda:1" if torch.cuda.is_available() else "cpu"

processor = LightOnOcrProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"
print(f"Using device: {device}")

model = LightOnOcrForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map=device,
).to(device)

for param in model.model.language_model.parameters():
    param.requires_grad = False
print(f"Language model frozen: {param.requires_grad}")

#=====================================================================================================
BASE_DIR = "/media/drive-2t/hoangnv83/code/ocr/data/r2k"
outpth = "./model"
IMAGE_DIR = f"{BASE_DIR}/pdf"
HTML_DIR  = f"{BASE_DIR}/labels"
#==================================================================

class PDFHTMLDataset(Dataset):
    def __init__(
        self,
        pdf_dir,
        html_dir,
        ids=None,
        dpi=200,
    ):
        self.pdf_dir = pdf_dir
        self.html_dir = html_dir
        self.dpi = dpi

        if ids is None:
            self.ids = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(pdf_dir)
                if f.lower().endswith(".pdf")
            )
        else:
            self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        src_pdf = os.path.join(self.pdf_dir, f"{id_}.pdf")

        images = convert_from_path(
            src_pdf,
            dpi=self.dpi,
            first_page=1,
            last_page=1,
        )
        image = images[0].convert("RGB")

        with open(
            os.path.join(self.html_dir, f"{id_}.html"),
            "r",
            encoding="utf-8"
        ) as f:
            html_code = f.read().strip()

        return {
            "images": [image],
            "texts": [
                {
                    "assistant": html_code
                }
            ]
        }

full_ds = PDFHTMLDataset(
    pdf_dir=IMAGE_DIR,
    html_dir=HTML_DIR,
)

ids = full_ds.ids
n = len(ids)

train_ids = ids[:int(0.85 * n)]
val_ids   = ids[int(0.85 * n):int(0.95 * n)]
test_ids  = ids[int(0.95 * n):]

train_ds = PDFHTMLDataset(IMAGE_DIR, HTML_DIR, ids=train_ids)
val_ds   = PDFHTMLDataset(IMAGE_DIR, HTML_DIR, ids=val_ids)
test_ds  = PDFHTMLDataset(IMAGE_DIR, HTML_DIR, ids=test_ids)

# assistant start pattern: <|im_end|>\n<|im_start|>assistant\n
ASSISTANT_START_PATTERN = [151645, 198, 151644, 77091, 198]
MAX_LENGTH = 1024
LONGEST_EDGE = 700

def evaluate_model(model, dataset, num_samples=50, batch_size=8, description="Model"):
    model.eval()

    predictions = []
    ground_truths = []

    print(f"\nEvaluating {description} on {num_samples} samples...")

    for start_idx in tqdm(range(0, min(num_samples, len(dataset)), batch_size)):
        end_idx = min(start_idx + batch_size, num_samples, len(dataset))
        batch_samples = [dataset[i] for i in range(start_idx, end_idx)]

        batch_images = [[s["images"][0]] for s in batch_samples]
        batch_ground_truths = [s["texts"][0]["assistant"].strip() for s in batch_samples]

        messages = [{"role": "user", "content": [{"type": "image"}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts = [text] * len(batch_images)

        inputs = processor(text=texts,
                           images=batch_images,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=MAX_LENGTH,
                           size={"longest_edge": LONGEST_EDGE},
                           ).to(device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True)

        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[:, input_length:]
        batch_predictions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        batch_predictions = [p.strip() for p in batch_predictions]

        predictions.extend(batch_predictions)
        ground_truths.extend(batch_ground_truths)

    cer_score = cer(ground_truths, predictions) * 100
    wer_score = wer(ground_truths, predictions) * 100
    perfect_matches = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)

    print(f"CER: {cer_score:.2f}% | WER: {wer_score:.2f}% | Perfect: {perfect_matches}/{num_samples}")

    for i in range(min(3, len(predictions))):
        match = "✅" if predictions[i] == ground_truths[i] else "❌"
        print(f"{match} Sample {i+1}: '{predictions[i]}' vs '{ground_truths[i]}'")

    return {"cer": cer_score, "wer": wer_score, "perfect_matches": perfect_matches}
#***************************************************************************
from peft import PeftModel
print("\n" + "="*80)
print("BEFORE TRAINING")
print("="*80)

if isinstance(model, PeftModel):
    with model.disable_adapter():
        base_results = evaluate_model(model, test_ds, num_samples=100, batch_size=4, description="Base")
else:
    base_results = evaluate_model(model, test_ds, num_samples=100, batch_size=4, description="Base")

torch.cuda.empty_cache()

#***********************************************************************
def collate_fn(examples):
    batch_messages = []
    batch_images = []

    for example in examples:
        example_images = example["images"]
        example_texts = example["texts"]

        assert len(example_images) == 1, (
            f"Expected 1 image per sample, got {len(example_images)}"
        )
        assert len(example_texts) == 1, (
            f"Expected 1 text per sample, got {len(example_texts)}"
        )

        image = example_images[0].convert("RGB")
        batch_images.append(image)

        conversation = example_texts[0]
        # strip extra whitespaces and newlines to avoid tokenization issues
        assistant_text = conversation.get("assistant", "").strip()

        messages = [
            {"role": "user", "content": [{"type": "image"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]
        batch_messages.append(messages)

    if len(batch_images) == 0:
        return None

    texts = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        for messages in batch_messages
    ]

    inputs = processor(
        text=texts,
        images=batch_images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        size={"longest_edge": LONGEST_EDGE},  # reduce due to memory requirements
    )

    labels = inputs["input_ids"].clone()
    pad_token_id = processor.tokenizer.pad_token_id

    for i in range(len(labels)):
        full_ids = inputs["input_ids"][i].tolist()

        # find where assistant content starts (after the assistant marker)
        assistant_content_start = None
        for idx in range(len(full_ids) - len(ASSISTANT_START_PATTERN)):
            if (
                full_ids[idx : idx + len(ASSISTANT_START_PATTERN)]
                == ASSISTANT_START_PATTERN
            ):
                assistant_content_start = idx + len(ASSISTANT_START_PATTERN)
                break

        if assistant_content_start is None:
            print(f"Warning: Could not find assistant marker in sample {i}")
            print(f"Sample {i} failed. Text: {texts[i]}")
            labels[i, :] = -100
        else:
            # mask everything first
            labels[i, :] = -100

            # unmask from assistant content start to end
            # this trains on: assistant text + EOS
            for idx in range(assistant_content_start, len(full_ids)):
                if full_ids[idx] == pad_token_id:
                    break
                labels[i, idx] = inputs["input_ids"][i, idx]

        # mask padding tokens
        labels[i, inputs["input_ids"][i] == pad_token_id] = -100

    inputs["labels"] = labels

    # convert tensors to device with proper dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    return inputs

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('high')

output_dir = f"LightOnOCR-2-ft-{finevision_subset}"
use_bf16 = torch.cuda.is_available()

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=50,
    # max_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=4,
    learning_rate=6e-5,
    weight_decay=0.0,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=use_bf16,
    fp16=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    warmup_steps=10,
    lr_scheduler_type="linear",
)

print(f"Output directory: {output_dir}")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

val_ds_small = Subset(val_ds, range(100))
train_ds_small = Subset(train_ds, range(480))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds_small,
    eval_dataset=val_ds_small,
    data_collator=collate_fn,
)

print("Starting training...")
print(f"Number of training samples: {len(train_ds_small)}")
print(f"Number of validation samples: {len(val_ds_small)}")

trainer.train()

# import os
# import matplotlib.pyplot as plt

# # ===== PLOT =====
# save_dir = "plot_rs"
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "loss_curve.png")

# train_steps = []
# train_losses = []
# eval_steps = []
# eval_losses = []

# for entry in trainer.state.log_history:
#     if 'loss' in entry:
#         train_steps.append(entry['step'])
#         train_losses.append(entry['loss'])
#     if 'eval_loss' in entry:
#         eval_steps.append(entry['step'])
#         eval_losses.append(entry['eval_loss'])

# plt.figure(figsize=(10, 6))
# plt.plot(train_steps, train_losses, label='Training Loss', marker='o', linewidth=2)
# plt.plot(eval_steps, eval_losses, label='Validation Loss', marker='s', linewidth=2)
# plt.xlabel('Steps', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()

# plt.savefig(save_path, dpi=300)
# plt.show()

# print(f"Plot save in: {save_path}")

print("\n" + "="*80)
print("AFTER TRAINING")
finetuned_results = evaluate_model(model, test_ds, num_samples=100, batch_size=4, description="Finetuned")

print("\n" + "="*80)
print("COMPARISON")
print(f"{'Metric':<20} {'Base':<12} {'Finetuned':<12} {'Change':<12}")
print("-" * 56)
print(f"{'CER (%)':<20} {base_results['cer']:<12.2f} {finetuned_results['cer']:<12.2f} {base_results['cer']-finetuned_results['cer']:+.2f}")
print(f"{'WER (%)':<20} {base_results['wer']:<12.2f} {finetuned_results['wer']:<12.2f} {base_results['wer']-finetuned_results['wer']:+.2f}")
print(f"{'Perfect':<20} {base_results['perfect_matches']:<12} {finetuned_results['perfect_matches']:<12} {finetuned_results['perfect_matches']-base_results['perfect_matches']:+d}")
print("="*80)

trainer.save_model(outpth)
processor.save_pretrained(outpth)

print(f"Model saved to {outpth}")
