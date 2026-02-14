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
  - 0rn0/gpt2-30m-tinystories
widget:
  - text: "Write a story about: a little girl and her dog at the park\nWith: happy ending\n\n### Story:\n"
    example_title: "Happy ending"
    output:
      text: "Once there was a little girl and her dog, Ducky. Every day, Ducky and Ducky went to the park to play. One day, Ducky met a little girl who was only three years old.\nThe little girl asked Ducky, \"Do you want to play with me?\"\nDucky was very happy. \"Yes!\" he said, \"Let's!\"\nSo they ran around the park and splashed in the water. Ducky was having so much fun! Ducky wanted to jump over the fence, but Ducky said, \"No Ducky, I can't jump over the fence!\"\nThe little girl smiled and said, \"Don't worry, I'll show you how.\"\nAnd just as Ducky jumped, Ducky jumped one last time and landed back in the park. Ducky watched and smiled. He was so happy they were able to play together!"
  - text: "Write a story about: a boy who lost his favorite toy\nWith: sad ending\n\n### Story:\n"
    example_title: "Sad ending"
    output:
      text: "Once there was a boy who loved to play with his toy. He liked to play with it every day.\nOne day he was playing with his toy and he noticed something strange. He went to his mom and asked what was wrong. His mom said he had lost his toy. The boy wanted to help and he started to search.\nHe looked in the room the closet and found his toy. He was so happy and he took it out to play with it.\nBut then he realised he had lost his toy. He looked everywhere but he couldn't find it. He was so sad.\nHe started to cry. His mom tried to comfort him and said she would try and find it. But nothing worked. The boy was sad so he went back to bed without his toy."
---

# GPT-2 30M — TinyStories SFT

## Model Details
- **Architecture**: GPT-2 (custom implementation)
- **Parameters**: ~30M
- **Context Length**: 512 tokens
- **Embedding Dim**: 384
- **Attention Heads**: 6
- **Transformer Layers**: 6
- **Tokenizer**: GPT-2 BPE (tiktoken, vocab size 50,257)

## Training

### Pre-training
Pre-trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset for 6 epochs. Pre-trained weights available at [0rn0/gpt2-30m-tinystories](https://huggingface.co/0rn0/gpt2-30m-tinystories).

### Supervised Fine-Tuning (SFT)
Fine-tuned for 5 epochs on [0rn0/tinystories-instruct-balanced](https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced), a curated instruction-tuning dataset with balanced happy/sad endings (50-50 split, ~325K examples).

**SFT Hyperparameters:**
- Optimizer: AdamW
- Learning Rate: 5e-4
- Batch Size: 64
- Epochs: 5

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

> Once there was a little girl and her dog, Ducky. Every day, Ducky and Ducky went to the park to play. One day, Ducky met a little girl who was only three years old.
> The little girl asked Ducky, "Do you want to play with me?"
> Ducky was very happy. "Yes!" he said, "Let's!"
> So they ran around the park and splashed in the water. Ducky was having so much fun! Ducky wanted to jump over the fence, but Ducky said, "No Ducky, I can't jump over the fence!"
> The little girl smiled and said, "Don't worry, I'll show you how."
> And just as Ducky jumped, Ducky jumped one last time and landed back in the park. Ducky watched and smiled. He was so happy they were able to play together!

### Sad ending
**Prompt:** `Write a story about: a boy who lost his favorite toy`

> Once there was a boy who loved to play with his toy. He liked to play with it every day.
> One day he was playing with his toy and he noticed something strange. He went to his mom and asked what was wrong. His mom said he had lost his toy. The boy wanted to help and he started to search.
> He looked in the room the closet and found his toy. He was so happy and he took it out to play with it. He was so happy and he ran around it.
> But then he realised he had lost his toy. He looked everywhere but he couldn't find it. He was so sad.
> He started to cry. His mom tried to comfort him and said she would try and find it. But nothing worked. The boy was sad so he went back to bed without his toy. He was so sad that he couldn't keep it.

## Limitations
- Generates short stories only (~180 words)
- Trained on synthetic data, not human-authored stories
- Limited vocabulary and narrative complexity
- Binary ending classification (happy/sad only)

## Links

- **Pre-trained model:** [0rn0/gpt2-30m-tinystories](https://huggingface.co/0rn0/gpt2-30m-tinystories)
- **125M SFT variant:** [0rn0/gpt2-125m-tinystories-sft](https://huggingface.co/0rn0/gpt2-125m-tinystories-sft)
- **Training dataset:** [0rn0/tinystories-instruct-balanced](https://huggingface.co/datasets/0rn0/tinystories-instruct-balanced)
- **Demo:** [Tiny Tales GPT](https://tinytales.aryandeore.ai)
- **Collection:** [Tiny Stories](https://huggingface.co/collections/0rn0/tiny-stories)
