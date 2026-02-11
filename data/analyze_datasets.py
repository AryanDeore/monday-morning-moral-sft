"""Compare TinyStoriesInstruct datasets to choose best for fine-tuning."""

from datasets import load_dataset
from collections import Counter
from typing import Dict, Tuple


def analyze_dataset(repo: str, split: str, num_samples: int = 1000) -> Dict:
    """Analyze a single dataset split (handles both original and fork formats)."""
    print(f"\nAnalyzing {repo} - {split} split ({num_samples} samples)...")

    ds = load_dataset(repo, split=split, streaming=True)

    ending_types = Counter()
    summary_lengths = []
    story_lengths = []
    has_features = 0
    missing_fields = 0

    count = 0

    # Determine format by checking first row
    is_fork_format = False
    for first_row in ds.take(1):
        # Fork format: all fields in single text row
        # Original format: needs accumulation until endoftext
        is_fork_format = "Story:" in first_row["text"] and "<|endoftext|>" not in first_row["text"]

    if is_fork_format:
        # Fork format: complete example per row
        for row in ds:
            text = row["text"]

            # Extract fields
            summary = None
            features = None
            story = None

            lines = text.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("Summary:"):
                    summary = line.replace("Summary:", "").strip()
                elif line.startswith("Features:"):
                    features = line.replace("Features:", "").strip()
                elif line.startswith("Story:"):
                    story_lines = lines[i+1:]
                    story = "\n".join(story_lines).strip()

            # Count
            if summary and story:
                ending = "sad" if features and "BadEnding" in features else "happy"
                ending_types[ending] += 1
                summary_lengths.append(len(summary.split()))
                story_lengths.append(len(story.split()))
                if features:
                    has_features += 1
            else:
                missing_fields += 1

            count += 1
            if count >= num_samples:
                break
    else:
        # Original format: accumulate until endoftext
        current_example = []
        for row in ds:
            text = row["text"]
            current_example.append(text)

            if "<|endoftext|>" in text:
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
                        story_lines = lines[i+1:]
                        story = "\n".join(story_lines).replace("<|endoftext|>", "").strip()

                # Count
                if summary and story:
                    ending = "sad" if features and "BadEnding" in features else "happy"
                    ending_types[ending] += 1
                    summary_lengths.append(len(summary.split()))
                    story_lengths.append(len(story.split()))
                    if features:
                        has_features += 1
                else:
                    missing_fields += 1

                current_example = []
                count += 1

                if count >= num_samples:
                    break

    return {
        "repo": repo,
        "split": split,
        "total_examples": count,
        "complete_examples": count - missing_fields,
        "missing_fields": missing_fields,
        "ending_distribution": dict(ending_types),
        "avg_summary_length": sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0,
        "avg_story_length": sum(story_lengths) / len(story_lengths) if story_lengths else 0,
        "has_features_pct": (has_features / (count - missing_fields)) * 100 if count > missing_fields else 0,
    }


def compare_datasets():
    """Compare both datasets."""

    print("=" * 70)
    print("TINYSTORIES INSTRUCTION DATASET COMPARISON")
    print("=" * 70)

    results = {}

    # Analyze original dataset
    print("\n📊 ORIGINAL DATASET (roneneldan/TinyStoriesInstruct - 21M-218k)")
    try:
        train_orig = analyze_dataset("roneneldan/TinyStoriesInstruct", "train", 1000)
        val_orig = analyze_dataset("roneneldan/TinyStoriesInstruct", "validation", 500)
        results["original"] = {"train": train_orig, "validation": val_orig}
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Analyze fork dataset
    print("\n📊 FORK DATASET (skeskinen/TinyStories-Instruct-hf - 2.5M-25k)")
    try:
        train_fork = analyze_dataset("skeskinen/TinyStories-Instruct-hf", "train", 1000)
        val_fork = analyze_dataset("skeskinen/TinyStories-Instruct-hf", "validation", 500)
        results["fork"] = {"train": train_fork, "validation": val_fork}
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Print comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for dataset_name, splits in results.items():
        print(f"\n{dataset_name.upper()}:")
        total_train = splits["train"]["complete_examples"]
        total_val = splits["validation"]["complete_examples"]
        total = total_train + total_val

        print(f"  Total examples: {total:,} (train: {total_train:,}, val: {total_val:,})")
        print(f"  Train ending split: happy {splits['train']['ending_distribution'].get('happy', 0)} / sad {splits['train']['ending_distribution'].get('sad', 0)}")
        print(f"  Val ending split: happy {splits['validation']['ending_distribution'].get('happy', 0)} / sad {splits['validation']['ending_distribution'].get('sad', 0)}")
        print(f"  Avg summary length: {splits['train']['avg_summary_length']:.1f} words")
        print(f"  Avg story length: {splits['train']['avg_story_length']:.1f} words")

    # Evaluation criteria
    print("\n" + "=" * 70)
    print("EVALUATION CRITERIA FOR YOUR USE CASE")
    print("=" * 70)
    print("""
✓ Dataset Size: Larger is better (more diverse examples)
✓ Ending Balance: Need ~50% sad, 50% happy for balanced training
✓ Data Quality: Complete Summary + Story fields
✓ OOD Handling: Plan to add refusal examples separately:
  - "Write a poem" → "I'm trained to generate short stories"
  - "Who is the president?" → "I'm trained to generate short stories"
  - "What is 2+2?" → "I'm trained to generate short stories"

📌 Current ending split is ~89% happy, 11% sad - IMBALANCED
   Solution: Over-sample sad examples OR filter to equal split
    """)


if __name__ == "__main__":
    compare_datasets()
