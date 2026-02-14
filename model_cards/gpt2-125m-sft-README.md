---
license: openrail
language:
  - en
library_name: pytorch
pipeline_tag: text-generation
tags:
  - gpt2
  - tinystories
  - sft
  - instruction-tuning
  - story-generation
  - from-scratch
  - causal-lm
datasets:
  - 0rn0/tinystories-instruct-balanced
base_model:
  - 0rn0/gpt2-125m-tinystories
widget:
  - text: "Write a story about: a little girl and her dog at the park\nWith: happy ending\n\n### Story:\n"
    example_title: "Happy ending"
    output:
      text: "Once upon a time there was a little girl and her loyal dog. They were best friends and did everything together. One day they went to the park and they were having lots of fun.\nSuddenly, the girl saw a big seat. She wanted to sit on it, so she asked her loyal dog, \"Can I sit on the seat?\" Her dog barked happily and the girl hopped on the seat.\nThey were both very happy and they spent the whole day playing and laughing together. They even saw some other children playing in the park.\nAt the end of the day, the girl and her loyal dog went home. The girl was sure that her loyal dog was always there to protect her and make her feel safe. She hugged him and said, \"I love you, my loyal dog!\""
  - text: "Write a story about: a boy who lost his favorite toy\nWith: sad ending\n\n### Story:\n"
    example_title: "Sad ending"
    output:
      text: "The boy said, \"Yes, I understand. I lost my toy and I can't find it.\" The man said, \"Don't worry, I'll help you find it.\"\nThey looked and looked for the toy, but they could not find it. The boy was very sad. The man said, \"I'm sorry, I can't find your toy.\" The boy went home with a sad face, and the man went back to his house with a bad feeling."
---

# GPT-2 125M — TinyStories SFT

## Model Details
- **Architecture**: GPT-2 (custom implementation)
- **Parameters**: ~125M
- **Context Length**: 512 tokens
- **Embedding Dim**: 768
- **Attention Heads**: 12
- **Transformer Layers**: 12
- **Tokenizer**: GPT-2 BPE (tiktoken, vocab size 50,257)

## Training

### Pre-training
Pre-trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset for 3 epochs. Pre-trained weights available at [0rn0/gpt2-125m-tinystories](https://huggingface.co/0rn0/gpt2-125m-tinystories).

### Supervised Fine-Tuning (SFT)
Fine-tuned for 1 epoch on [0rn0/tinystories-instruct-balanced](https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced), a curated instruction-tuning dataset with balanced happy/sad endings (50-50 split, ~325K examples).

**SFT Hyperparameters:**
- Optimizer: AdamW
- Learning Rate: 3e-4
- Batch Size: 32
- Epochs: 1

## Usage

This is a **custom PyTorch model** (not transformers-compatible). Requires source code from the GitHub repository.

```bash
git clone https://github.com/aryandeore/monday-morning-moral-sft.git
cd monday-morning-moral-sft
uv sync
```

### Instruction Format
```
Write a story about: {topic}
With: {ending} ending

### Story:
```

Where `{ending}` is either `happy` or `sad`.

## Examples

### Happy ending
**Prompt:** `Write a story about: a little girl and her dog at the park`

> Once upon a time there was a little girl and her loyal dog. They were best friends and did everything together. One day they went to the park and they were having lots of fun.
> Suddenly, the girl saw a big seat. She wanted to sit on it, so she asked her loyal dog, "Can I sit on the seat?" Her dog barked happily and the girl hopped on the seat.
> They were both very happy and they spent the whole day playing and laughing together. They even saw some other children playing in the park.
> At the end of the day, the girl and her loyal dog went home. The girl was sure that her loyal dog was always there to protect her and make her feel safe. She hugged him and said, "I love you, my loyal dog!"

### Sad ending
**Prompt:** `Write a story about: a boy who lost his favorite toy`

> The boy said, "Yes, I understand. I lost my toy and I can't find it." The man said, "Don't worry, I'll help you find it."
> They looked and looked for the toy, but they could not find it. The boy was very sad. The man said, "I'm sorry, I can't find your toy." The boy went home with a sad face, and the man went back to his house with a bad feeling.

## Limitations
- Generates short stories only (~180 words)
- Trained on synthetic data, not human-authored stories
- Limited vocabulary and narrative complexity
- Binary ending classification (happy/sad only)
- Only fine-tuned for 1 epoch — could benefit from further training

## Links

- **Pre-trained model:** [0rn0/gpt2-125m-tinystories](https://huggingface.co/0rn0/gpt2-125m-tinystories)
- **30M SFT variant:** [0rn0/gpt2-30m-tinystories-sft](https://huggingface.co/0rn0/gpt2-30m-tinystories-sft)
- **Training dataset:** [0rn0/tinystories-instruct-balanced](https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced)
- **Demo:** [Tiny Tales GPT](https://tinytales.aryandeore.ai)
- **Collection:** [Tiny Stories](https://huggingface.co/collections/0rn0/tiny-stories)
