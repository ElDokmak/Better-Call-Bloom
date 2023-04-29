<div>
<img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*JliDVAl4g8t6D4v49iedGA.png" >
<div>

# Credits 
All credits for the repo is for [Thomas Rochefort-Beaudoin](https://medium.com/@thomas.rochefort.beaudoin) (PhD student in mechanical engineering @polymtl. Who mostly write about money and AI) and his blog 
[Training a Language Model To Give (Non) Legal Advice](https://pub.towardsai.net/training-a-large-language-model-to-give-non-legal-advice-b9f6d7d11016) 
that gave me experience of how to fine tune LLM.

# The BLOOM model
BLOOM model from the awesome folks behind the BigScience initiative. To promote research inclusion in a field dominated by private research and unreleased models, the BLOOM initiative produced a completely open-source large language model of 176B parameters (the same scale as its private competitor GPT3 from OpenAI).
Multiple checkpoints are available on [HuggingFace](https://huggingface.co/bigscience).

BLOOM is an autoregressive language model built with a "decoder-only" transformer architecture for text-generation tasks. It was pretrained on the ROOTS corpus, which is constructed of over 1.6TB of text data encompassing 46 natural languages and 13 programming languages.

## The Pile Of Law dataset 
The Pile Of Law dataset is a corpus of 256 GB of legal texts ranging anywhere from the U.S. State codes to Bar exam outlines. 
containing a large set of Q&A written in plain, easy-to-understand English on which we can finetune our model.

```
from datasets import load_dataset
dataset = load_dataset("pile-of-law/pile-of-law",'r_legaladvice')
```

# Finetuning BLOOM-3B
Thomas limited myself to the 3B parameters BLOOM checkpoint, which is easily downloaded using the transformers library from HuggingFace:
```
from transformers import BloomTokenizerFast, BloomForCausalLM
tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-3b")
model = BloomForCausalLM.from_pretrained("bigscience/bloom-3b")
```
We will finetune BLOOM in a causal language modeling task where the model has to predict the next token in a sentence given the past tokens. To do so, we need to prepare the dataset so as to create 'blocks' of input texts corresponding to a single sequence.

We first remove the URL and timestamp from each sample and tokenize the text field. We can use the dataset.map function to efficiently apply a function to the datasets with multiple threads:

```
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_dataset = dataset.map(tokenize_function, 
                                batched=True, 
                                num_proc=8, 
                                remove_columns=["text","created_timestamp","downloaded_timestamp","url"])
```

Next, we create the "blocks" of texts that will represent our unique sequences of tokens. The block size can be tuned depending on the VRAM available on your GPU.
```
block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
 k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=8,
)
```
Training is even simpler:
```
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)
```

The Trainer API will run training for 3 epochs on the training dataset of r/legaladvice.

Pay Attention The model was trained on an A-100 with 40GB of VRAM, which allowed me to use a block_size for the samples of 200 and a batch size of 16.
if your specs is lower than that u will face a problem of running out of resources.

With these settings, finetuning for 3 epochs took about 26 hours on the single A100.

NOTE : When using pytorch optimizer fintuning for 3 epochs took about 16 hours whis is incredibly great.

The trained weights are available on [HuggingFace](https://huggingface.co/tomrb/bettercallbloom-3b).










