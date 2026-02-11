# Monday Morning Moral - Instruction Fine-Tuning

Fine-tune a pretrained TinyStories model to follow instructions and generate customized stories.

## Project Structure

```
├── models/                      # Model architecture (copied from pretraining)
│   ├── gpt2.py
│   ├── transformer.py
│   ├── multi_head_attention.py
│   ├── feed_forward.py
│   └── embeddings.py
├── data/
│   ├── prepare_dataset.py       # Parse + curate TinyStoriesInstruct
│   ├── instruction_dataset.py   # InstructionDataset (tokenizes at init)
│   └── collate.py               # custom_collate_fn (pad + mask)
├── utils/
│   ├── config.py                # Model config (copied from pretraining)
│   └── formatting.py            # format_instruction() template
├── finetune.py                  # Main training script
├── generate.py                  # Inference with instruction format
└── checkpoint.py                # Checkpoint utilities (copied from pretraining)
```

## Instruction Template

```
Write a story about: {summary}
With: {ending} ending

### Story:
{story}
```

At inference, the model receives everything up to `### Story:` and generates the story.

## Key Configuration

- **Model**: 30M parameter pretrained checkpoint
- **Context length**: 512 tokens
- **Hyperparameters**: AdamW lr=5e-5, batch_size=8, epochs=2
- **Dataset**: TinyStories-Instruct curated subset

## Next Steps

1. Run `data/prepare_dataset.py` to parse and upload curated dataset
2. Implement `data/instruction_dataset.py` for tokenization
3. Implement `data/collate.py` for batching
4. Implement `finetune.py` for training
5. Implement `generate.py` for inference
6. Build Gradio UI for demo
