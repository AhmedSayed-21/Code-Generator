
---

# 🧑‍💻 Code Generator with CodeT5 + LoRA

This project fine-tunes [Salesforce CodeT5](https://huggingface.co/Salesforce/codet5-base) using **LoRA (Low-Rank Adaptation)** to generate Python code from natural language instructions.
It includes data preprocessing, cleaning, training, evaluation, and inference with different prompting strategies (zero-shot, one-shot, few-shot).

---

## 🚀 Features

* Load and preprocess custom dataset (`train.csv`).
* Clean and normalize text (fix unicode, remove unwanted chars).
* Detect and fix Python code formatting using `parso` and `black`.
* Build HuggingFace `Dataset` with train/val/test split.
* Fine-tune `codet5-base` with LoRA using HuggingFace `Trainer`.
* Evaluate model performance on test set.
* Run inference with:

  * Zero-shot prompts
  * One-shot prompts
  * Few-shot prompts

---

## 📂 Project Structure

```
code-generator-1.ipynb   # Main notebook
train.csv                # Training dataset (input/output/instruction)
```

---

## ⚙️ Installation

Make sure you have Python 3.8+ and install dependencies:

```bash
pip install pandas matplotlib black parso joblib tqdm
pip install transformers datasets accelerate peft bitsandbytes
```

If running on **Kaggle** or **Colab**, GPU is recommended.

---

## 📊 Dataset

The dataset should contain at least these columns:

* `instruction`: Natural language instruction.
* `input`: Optional additional input.
* `output`: Target Python code.

The notebook will clean, normalize, and combine them into:

* `input_text`
* `target_text`

---

## 🏋️ Training

Training is handled with HuggingFace `Seq2SeqTrainer`:

```python
trainer.train()
```

The fine-tuned model will be saved at:

```
./codet5-lora-finetuned
```

---

## ✅ Evaluation

Run:

```python
metrics = trainer.evaluate(test_dataset)
print(metrics)
```

---

## 🤖 Inference

Examples with HuggingFace `pipeline`:

```python
from transformers import pipeline

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

prompt = """
Task: Write a Python function.

Problem:
Write a function to add two numbers.

Input:
a = 3, b = 5

Answer:
"""

output = pipe(prompt, max_new_tokens=128, num_beams=4)
print(output[0]["generated_text"])
```

---

## 📌 Example Outputs

* **Fibonacci generator**
* **Reverse string function**
* **Odd/Even checker**
* **Basic arithmetic functions**

---

## 🔧 Future Work

* Support multi-language code generation.
* Add evaluation metrics like BLEU/CodeBLEU.
* Package notebook into scripts for CLI usage.

---

## 📜 License

MIT License. Feel free to use, modify, and share.

 
