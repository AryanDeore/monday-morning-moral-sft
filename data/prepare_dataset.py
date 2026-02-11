"""Parse raw TinyStoriesInstruct dataset and curate for fine-tuning.

This script:
1. Loads TinyStoriesInstruct from HuggingFace
2. Parses examples from raw text format
3. Extracts Summary, Features (ending type), and Story
4. Creates a curated dataset with explicit schema
5. Uploads to HuggingFace Hub
"""

from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, load_dataset
from typing import Generator


def parse_ending_type(features_str: str) -> str:
    """Extract ending type from Features field.

    Args:
        features_str: The Features field content

    Returns:
        "sad" if "BadEnding" in features, else "happy"
    """
    if features_str and "BadEnding" in features_str:
        return "sad"
    return "happy"


def parsed_examples_generator(split: str = "train"):
    """Stream raw TinyStoriesInstruct dataset and yield parsed examples.

    Accumulates rows until <|endoftext|>, extracts fields, yields dicts.

    Args:
        split: "train" or "test"

    Yields:
        Dict with keys: instruction, response, ending
    """
    raw_ds = load_dataset("roneneldan/TinyStoriesInstruct", split=split, streaming=True)

    current_example = []

    for row in raw_ds:
        text = row["text"]
        current_example.append(text)

        if "<|endoftext|>" in text:
            # End of example - parse and yield
            full_text = "\n".join(current_example)

            # Extract fields
            summary = None
            features = None
            story = None

            lines = full_text.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Summary:"):
                    summary = line.replace("Summary:", "").strip()
                elif line.startswith("Features:"):
                    features = line.replace("Features:", "").strip()
                elif line.startswith("Story:"):
                    # Story spans rest of lines until endoftext
                    story_lines = lines[i+1:]
                    story = "\n".join(story_lines).replace("<|endoftext|>", "").strip()

            # Yield if complete
            if summary and story:
                ending = parse_ending_type(features)
                instruction = f"Write a story about: {summary}\nWith: {ending} ending\n\n### Story:\n"

                yield {
                    "instruction": instruction,
                    "response": story,
                    "ending": ending,
                }

            current_example = []


def create_and_upload_dataset():
    """Create curated dataset and upload to HuggingFace Hub."""

    # Define schema
    features = Features({
        'instruction': Value('string'),
        'response': Value('string'),
        'ending': ClassLabel(names=['happy', 'sad']),
    })

    # Create train and test datasets
    print("Creating train dataset...")
    train_ds = Dataset.from_generator(
        parsed_examples_generator,
        gen_kwargs={"split": "train"},
        features=features
    )

    print("Creating test dataset...")
    test_ds = Dataset.from_generator(
        parsed_examples_generator,
        gen_kwargs={"split": "test"},
        features=features
    )

    # Test with small subset
    print("\nSample from train set:")
    sample = train_ds.select(range(min(3, len(train_ds))))
    for i, ex in enumerate(sample):
        print(f"\n--- Example {i} ---")
        print(f"Instruction: {ex['instruction'][:100]}...")
        print(f"Ending: {ex['ending']}")
        print(f"Response: {ex['response'][:100]}...")

    # Combine and push to hub
    ds = DatasetDict({"train": train_ds, "test": test_ds})

    print(f"\nPushing to hub...")
    print(f"Train size: {len(train_ds)}")
    print(f"Test size: {len(test_ds)}")

    ds.push_to_hub("aryandeore/tinystories-instruct-curated", max_shard_size="500MB")
    print("✓ Dataset uploaded!")


if __name__ == "__main__":
    create_and_upload_dataset()
