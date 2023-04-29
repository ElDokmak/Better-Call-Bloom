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



