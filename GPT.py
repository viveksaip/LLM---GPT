#1. Install Required Libraries
#First, make sure you have the necessary libraries installed. Run the following command in a Google Colab cell:

!pip install transformers datasets torch


#2. Import Libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch


#3. Load a Pre-trained Model and Tokenizer
#We will use the GPT-2 model, which is readily available from Hugging Face's model hub.

model_name = 'gpt2'

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add special tokens if required (e.g., padding tokens for sequences)
tokenizer.pad_token = tokenizer.eos_token




#4. Load Your Dataset
#For the sake of simplicity, we'll load a small dataset from Hugging Face's datasets library. You can replace this with your own text dataset.

# Load a sample dataset, e.g., the "wikitext" dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Check the dataset structure
dataset

#smaller dataset
#dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


#5. Tokenize the Dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors='pt', padding=True, truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Format the datasets for PyTorch
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])


#6. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # Number of epochs
    per_device_train_batch_size=2,  # Adjust based on available memory
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
)


#7. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)


#8. Train the Model
trainer.train()

#9. Save the Model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")


#10. Generate Text with the Fine-Tuned Model
# Load the fine-tuned model
fine_tuned_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

# Generate text
input_text = "Once upon a time"
inputs = fine_tuned_tokenizer(input_text, return_tensors="pt")

# Generate a sequence
generated_ids = fine_tuned_model.generate(inputs['input_ids'], max_length=100)

# Decode the generated text
generated_text = fine_tuned_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_text)
