# Importing the necessary libraries
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments

# Loading the XLM-Roberta large model from Hugging Face
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-large')

# Loading the MNLI train set and the XNLI validation and test sets
# Assuming they are in the same format as the GLUE datasets
# See https://huggingface.co/datasets/glue for more details
from datasets import load_dataset
mnli_train = load_dataset('glue', 'mnli', split='train')
xnli_val = load_dataset('xnli', 'validation')
xnli_test = load_dataset('xnli', 'test')

# Combining the MNLI train set and the XNLI validation and test sets
# Shuffling the translations for the premise and hypothesis in the final epoch
# See https://huggingface.co/joeddav/xlm-roberta-large-xnli for more details
from random import shuffle
def shuffle_translations(example):
  languages = list(example['premise'].keys())
  shuffle(languages)
  example['premise'] = {lang: example['premise'][lang] for lang in languages}
  example['hypothesis'] = {lang: example['hypothesis'][lang] for lang in languages}
  return example

xnli_test = xnli_test.map(shuffle_translations) # Only shuffling the xnli_test dataset
combined_dataset = mnli_train.concatenate_datasets([xnli_val, xnli_test])

# Splitting the combined dataset into two parts
# The first part contains MNLI train set and XNLI validation set
# The second part contains only XNLI test set
first_part = combined_dataset.select(range(len(mnli_train) + len(xnli_val)))
second_part = combined_dataset.select(range(len(mnli_train) + len(xnli_val), len(combined_dataset)))

# Defining the training arguments
# See https://huggingface.co/transformers/main_classes/trainer.html for more details
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=4,              # number of training epochs
    per_device_train_batch_size=128, # batch size per device during training
    per_device_eval_batch_size=128,  # batch size for evaluation
    warmup_steps=len(first_part) * 0.1, # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

# Defining the trainer
trainer = Trainer(
    model=model,                         # the pretrained model
    args=training_args,                  # training arguments
)

# Training the model on the first part of the dataset for three epochs
trainer.train(train_dataset=first_part, max_steps=len(first_part) * 3)

# Training the model on the second part of the dataset for one epoch
trainer.train(train_dataset=second_part, max_steps=len(second_part))
