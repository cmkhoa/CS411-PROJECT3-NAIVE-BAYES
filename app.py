"""Vietnamese sentiment classification with Multinomial Naive Bayes."""

import csv
import math
import re
import unicodedata
from collections import Counter


LABELS = ("positive", "negative")
model = None


def tokenize(text):
    """Keep accents; split into lowercase alphabetic tokens."""
    text = unicodedata.normalize("NFC", text).lower()
    return re.findall(r"[^\W\d_]+", text)


def read_data(filename):
    """Read UTF-8 CSV data and reject malformed rows."""
    with open(filename, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["text", "label"]:
            raise ValueError(f"{filename}: expected columns text,label")
        rows = []
        for row in reader:
            if (None in row or not row["text"] or not row["text"].strip()
                    or row["label"] not in LABELS):
                raise ValueError(f"{filename}, line {reader.line_num}: invalid text or label")
            rows.append((row["text"], row["label"]))
    if not rows:
        raise ValueError(f"{filename}: add your manually authored or collected sentences")
    return rows


def train_model(train_file="data/train.csv"):
    """Learn counts and priors from the training file only."""
    global model
    model = None
    rows = read_data(train_file)
    sentences = Counter(label for _, label in rows)
    if any(sentences[label] == 0 for label in LABELS):
        raise ValueError("Training data must include both positive and negative sentences")

    counts = {label: Counter() for label in LABELS}
    for text, label in rows:
        counts[label].update(tokenize(text))
    vocabulary = set(counts["positive"]) | set(counts["negative"])
    if not vocabulary:
        raise ValueError("Training data must contain alphabetic tokens")

    model = {
        "vocabulary": vocabulary,
        "counts": counts,
        "totals": {label: sum(counts[label].values()) for label in LABELS},
        "sentences": sentences,
        "priors": {label: sentences[label] / len(rows) for label in LABELS},
    }
    return model


def require_model():
    if model is None:
        raise RuntimeError("Call train_model() before predicting or evaluating")


def word_likelihood(word, label):
    """Laplace smoothing with alpha = 1 for a vocabulary token."""
    require_model()
    return (model["counts"][label][word] + 1) / (
        model["totals"][label] + len(model["vocabulary"])
    )


def score_text(text):
    """Count repeated tokens and ignore tokens outside the training vocabulary."""
    require_model()
    tokens = [word for word in tokenize(text) if word in model["vocabulary"]]
    return {
        label: math.log(model["priors"][label])
        + sum(math.log(word_likelihood(word, label)) for word in tokens)
        for label in LABELS
    }


def predict(text):
    """Choose by log score, then prior, then positive on an exact tie."""
    scores = score_text(text)
    return max(LABELS, key=lambda label: (
        scores[label], model["priors"][label], label == "positive"
    ))


def evaluate_model(test_file="data/test.csv"):
    """Evaluate without updating the model; positive is the positive class."""
    require_model()
    rows = read_data(test_file)
    matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for text, actual in rows:
        predicted = predict(text)
        if actual == "positive":
            key = "TP" if predicted == "positive" else "FN"
        else:
            key = "TN" if predicted == "negative" else "FP"
        matrix[key] += 1
    return {
        "accuracy": (matrix["TP"] + matrix["TN"]) / len(rows),
        "confusion_matrix": matrix,
    }


def explain(text):
    """Print token counts, smoothed likelihoods, and scores for the video."""
    require_model()
    tokens = tokenize(text)
    print("Tokens:", tokens)
    for word in tokens:
        if word not in model["vocabulary"]:
            print(f"  {word}: ignored (unknown)")
            continue
        for label in LABELS:
            numerator = model["counts"][label][word] + 1
            denominator = model["totals"][label] + len(model["vocabulary"])
            probability = word_likelihood(word, label)
            print(f"  {word} | {label}: {numerator}/{denominator}"
                  f" = {probability:.6f}, log = {math.log(probability):.6f}")
    print("Log scores:", score_text(text))
    print("Prediction:", predict(text))


def main():
    try:
        train_model()
        print("Vocabulary size:", len(model["vocabulary"]))
        for label in LABELS:
            print(f"{label}: sentences={model['sentences'][label]},"
                  f" tokens={model['totals'][label]}, prior={model['priors'][label]:.4f}")
        results = evaluate_model()
        print(f"Accuracy: {results['accuracy']:.2%}")
        print("Confusion matrix:", results["confusion_matrix"])
    except (OSError, ValueError) as error:
        print(f"Setup error: {error}")
        return 1

    print("Enter Vietnamese text; :trace TEXT shows the math; :quit exits.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text == ":quit":
            break
        if text.startswith(":trace "):
            explain(text[len(":trace "):])
        elif text:
            print(predict(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
