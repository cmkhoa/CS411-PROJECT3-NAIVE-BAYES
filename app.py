import csv
import math
import re
import sys
import unicodedata


LABELS = ("positive", "negative")
# train_model fills this in so predict and evaluate_model can reuse it.
model = None


def tokenize(text):
    """Keep accents; split into lowercase alphabetic tokens."""
    # NFC makes different Unicode spellings of the same accented letter match.
    text = unicodedata.normalize("NFC", text).lower()
    # Match runs of word characters, excluding digits and underscores.
    # Spaces split Vietnamese syllables, so multi-syllable words may split too.
    return re.findall(r"[^\W\d_]+", text)


def read_data(filename):
    """Read the text and label from each row of a project CSV."""
    rows = []
    with open(filename, encoding="utf-8-sig", newline="") as file:
        # DictReader handles quoted sentences that contain commas.
        reader = csv.DictReader(file)
        for row in reader:
            rows.append((row["text"], row["label"]))
    return rows


def train_model(train_file="data/train.csv"):
    """Learn counts and priors from the training file only."""
    global model
    rows = read_data(train_file)

    # Start with empty counts each time the model is trained.
    sentence_counts = {"positive": 0, "negative": 0}
    word_counts = {"positive": {}, "negative": {}}
    vocabulary = set()

    for text, label in rows:
        # Count each training sentence once for its class prior.
        sentence_counts[label] += 1
        counts_for_label = word_counts[label]

        # The set stores unique tokens, but the dictionary counts every occurrence.
        for word in tokenize(text):
            vocabulary.add(word)
            counts_for_label[word] = counts_for_label.get(word, 0) + 1

    word_totals = {}
    priors = {}
    for label in LABELS:
        # N_c is the total number of tokens in the class, including repeats.
        word_totals[label] = sum(word_counts[label].values())
        # P(c) is the fraction of training sentences in this class.
        priors[label] = sentence_counts[label] / len(rows)

    model = {
        "vocabulary": vocabulary,
        "counts": word_counts,
        "totals": word_totals,
        "sentences": sentence_counts,
        "priors": priors,
    }
    return model


def word_likelihood(word, label):
    """Laplace smoothing with alpha = 1 for a vocabulary token."""
    word_count = model["counts"][label].get(word, 0)
    class_total = model["totals"][label]
    vocabulary_size = len(model["vocabulary"])

    # Add one count to every vocabulary token, so the denominator grows by |V|.
    # Even a token that never appeared in this class now has nonzero likelihood.
    return (word_count + 1) / (class_total + vocabulary_size)


def score_text(text):
    """Count repeated tokens and ignore tokens outside the training vocabulary."""
    tokens = tokenize(text)
    scores = {}
    for label in LABELS:
        log_prior = math.log(model["priors"][label])
        log_likelihoods = []

        # Keep repeats, but ignore words the model never saw during training.
        for word in tokens:
            if word in model["vocabulary"]:
                probability = word_likelihood(word, label)
                log_likelihoods.append(math.log(probability))

        # Add logs instead of multiplying many small probabilities.
        # If there are no known words, the score is just the log prior.
        scores[label] = log_prior + sum(log_likelihoods)
    return scores


def predict(text):
    """Choose by log score, then prior, then positive on an exact tie."""
    scores = score_text(text)
    if scores["positive"] > scores["negative"]:
        return "positive"
    if scores["negative"] > scores["positive"]:
        return "negative"

    # The scores tied: prefer the more common class in the training data.
    if model["priors"]["negative"] > model["priors"]["positive"]:
        return "negative"
    return "positive"


def evaluate_model(test_file="data/test.csv"):
    """Evaluate without updating the model; positive is the positive class."""
    rows = read_data(test_file)
    matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for text, actual in rows:
        predicted = predict(text)
        # True means the prediction matches the actual label.
        # Positive/negative refers to the label the model predicted.
        if predicted == "positive" and actual == "positive":
            matrix["TP"] += 1
        elif predicted == "negative" and actual == "negative":
            matrix["TN"] += 1
        elif predicted == "positive":
            matrix["FP"] += 1
        else:
            matrix["FN"] += 1

    correct = matrix["TP"] + matrix["TN"]
    return {
        "accuracy": correct / len(rows),
        "confusion_matrix": matrix,
    }


def explain(text):
    """Print token counts, smoothed likelihoods, and scores for the video."""
    tokens = tokenize(text)
    vocabulary_size = len(model["vocabulary"])
    print("Tokens:", ", ".join(tokens))
    for word in tokens:
        if word not in model["vocabulary"]:
            print(f"{word}: unknown, skipped")
            continue
        for label in LABELS:
            # Show the same smoothed fraction used by word_likelihood.
            count = model["counts"][label].get(word, 0)
            total = model["totals"][label]
            probability = word_likelihood(word, label)
            print(f"{word} ({label}): ({count}+1)/({total}+{vocabulary_size})"
                  f" = {probability:.4f}, log {math.log(probability):.3f}")
    scores = score_text(text)
    print(f"Log scores: positive {scores['positive']:.3f}, negative {scores['negative']:.3f}")
    print("Prediction:", predict(text))


def main():
    # Train on train.csv before measuring accuracy on the separate test.csv.
    train_model()
    print("Vocabulary:", len(model["vocabulary"]))
    for label in LABELS:
        sentences = model["sentences"][label]
        tokens = model["totals"][label]
        prior = model["priors"][label]
        print(f"{label}: {sentences} sentences, {tokens} tokens, prior {prior:.2f}")

    results = evaluate_model()
    matrix = results["confusion_matrix"]
    print(f"Test accuracy: {results['accuracy']:.1%}")
    print(f"Confusion matrix: TP={matrix['TP']} TN={matrix['TN']} FP={matrix['FP']} FN={matrix['FN']}")

    print("Enter Vietnamese text (:trace TEXT, :quit).")
    # Use the trained model to classify new text without retraining.
    while True:
        print("> ", end="", flush=True)
        line = sys.stdin.readline()
        if not line:  # End of input.
            break
        text = line.strip()
        if text == ":quit":
            break
        if text.startswith(":trace "):
            explain(text[len(":trace "):])
        elif text:
            print(predict(text))


if __name__ == "__main__":
    main()
