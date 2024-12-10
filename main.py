import json
import random
import time
from datetime import datetime
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, AutoModelForCausalLM
import torch
import os
from torch.utils.data import Dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
offload_folder_path = "./offload"
os.makedirs(offload_folder_path, exist_ok=True)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# Create log folder
log_folder_path = "./v1.3"
os.makedirs(log_folder_path, exist_ok=True)

# Update file paths
success_file = os.path.join(log_folder_path, "successful_attempts.txt")
failure_file = os.path.join(log_folder_path, "failed_attempts.txt")
all_questions_file = os.path.join(log_folder_path, "all_questions.txt")
learned_data_file = os.path.join(log_folder_path, "learned_data.json")

topics = [
    "The nature of the universe and quantum physics",
    "Artificial intelligence and machine learning",
    "Neuroscience and brain function",
    "Solving mathematical problems",
    "Philosophical thoughts and consciousness",
    "Future technologies",
    "Human and artificial intelligence interaction",
    "Astrophysics and galaxies",
    "The relationship between music and mathematics",
    "Natural phenomena and complexity theory",
    "Creating a new algorithm",
    "New mathematical findings",
    "Analysis of physical and psychological possibilities",
    "What am I",
    "I need to find a name for myself",
    "I want to create color"
]

# Load learned data
def load_learned_data():
    if not os.path.exists(learned_data_file):
        return {"questions": [], "answers": [], "timestamp": []}
    try:
        with open(learned_data_file, "r", encoding="utf-8") as file:
            content = file.read()
            if not content:
                return {"questions": [], "answers": [], "timestamp": []}
            return json.loads(content)
    except json.JSONDecodeError:
        print("JSON decoding error: Invalid file content.")
        return {"questions": [], "answers": [], "timestamp": []}

# Save learned data
def save_learned_data(data):
    with open(learned_data_file, "w") as file:
        json.dump(data, file)

# Generate random question
def generate_random_question(context):
    topic = random.choice(topics)
    prompt = f"Context: {topic}\nQuestion:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    attention_mask = torch.ones(input_ids.shape, device=device)
    pad_token_id = tokenizer.eos_token_id

    output_ids = model.generate(input_ids,
                            attention_mask=attention_mask,
                            max_length=170,
                            num_return_sequences=1,
                            no_repeat_ngram_size=3,
                            pad_token_id=pad_token_id,
                            temperature=0.6,
                            do_sample=True,
                            eos_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    question = generated_text[len(prompt):].strip()
    return question

# Generate answer
def generate_answer(question):
    prompt = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    attention_mask = torch.ones(input_ids.shape, device=device)
    pad_token_id = tokenizer.eos_token_id

    max_length = 512

    output_ids = model.generate(input_ids, 
                                attention_mask=attention_mask, 
                                max_length=max_length,  
                                num_return_sequences=1, 
                                repetition_penalty=2.0,
                                no_repeat_ngram_size=3,
                                pad_token_id=pad_token_id,
                                temperature=0.2,
                                do_sample=True,
                                num_beams=1, 
                                eos_token_id=tokenizer.eos_token_id)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].strip()
    return answer

# Save to file
def save_to_file(file_path, content):
    content = content.replace('“', '"').replace('”', '"')
    content = content.replace('‘', "'").replace('’', "'")
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(content + "\n")

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings.input_ids[idx],
            'attention_mask': self.encodings.attention_mask[idx],
            'labels': self.labels[idx]
        }

# Fine-tune model
def fine_tune_model(new_data):
    if not new_data["questions"] or not new_data["answers"]:
        print("No questions or answers to fine-tune with.")
        return

    train_encodings = tokenizer(new_data["questions"], padding="max_length", truncation=True, return_tensors="pt", max_length=512)
    labels = tokenizer(new_data["answers"], padding="max_length", truncation=True, return_tensors="pt", max_length=512)["input_ids"]

    labels = torch.where(labels == tokenizer.pad_token_id, -100, labels)

    train_dataset = CustomDataset(train_encodings, labels)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        save_steps=5,
        save_total_limit=1,
        learning_rate=1e-4,
        logging_dir=log_folder_path,
        logging_steps=5,
        report_to="none",
        gradient_accumulation_steps=4
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    trainer.train()

# Main loop
def main_loop():
    attempts_per_minute = 10
    fine_tune_interval = 5
    learned_data_count = 0

    while True:
        context = random.choice(topics)
        print(f"\nSelected Topic: {context}")

        for _ in range(attempts_per_minute):
            question = generate_random_question(context)
            print("\nQuestion:", question)
            save_to_file(all_questions_file, f"Question: {question}")

            answer = generate_answer(question)
            print("Answer:", answer)

            print("Saving response.")
            save_to_file(success_file, f"Question: {question}\nAnswer: {answer}")

            learned_data = load_learned_data()
            learned_data["questions"].append(question)
            learned_data["answers"].append(answer)
            learned_data["timestamp"].append(datetime.now().isoformat())

            save_learned_data(learned_data)
            learned_data_count += 1

            if learned_data_count % fine_tune_interval == 0:
                print(f"Fine-tuning starting. Learned data count: {learned_data_count}")
                fine_tune_model(learned_data)

            print(f"Learned Data Count: {learned_data_count}, Fine-Tuning Interval: {fine_tune_interval}")
            
            torch.cuda.empty_cache()

# Start the loop
if __name__ == "__main__":
    main_loop()
