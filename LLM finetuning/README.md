# LLM Finetuning

## Requirements:
- Huggingface account
- Weights and Biases account

## Models tried
- gpt2
- distillgpt2

## Datasets
- wikitext
- ptb, Penn tree bank

## Changes made to the code
- changed `max_length` in the `tokenizer` and `model.generate()` to 512
- Since gpt model is not trained with pad tokens, I had to set `tokenizer.pad_token = tokenizer.eos_token`
- Extracting proper columns in the input data </br>
    ```tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])```
- In `TrainingArugments()`, evaluation_strategy was corrected to `eval_strategy`

## Files:
Please check [distilgpt2+wiki.ipynb](distilgpt2+wiki.ipynb), [gpt2+wiki.ipynb](gpt2+wiki.ipynb) for complete implementations.

## NOTE:
**PS**: I ran out of compute instances therefore, I could not complete more experiments.