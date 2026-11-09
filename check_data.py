"""Check submission dataset sizes, labels, and duplicate sentences."""

from collections import Counter

from app import read_data, tokenize


def check_data():
    errors = []
    seen = {}
    for filename, per_class in (("data/train.csv", 80), ("data/test.csv", 20)):
        try:
            rows = read_data(filename)
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue
        counts = Counter(label for _, label in rows)
        print(f"{filename}: {len(rows)} sentences, {dict(counts)}")
        if counts != Counter(positive=per_class, negative=per_class):
            errors.append(f"{filename}: expected {per_class} sentences per class")
        for line, (text, _) in enumerate(rows, start=2):
            key = tuple(tokenize(text))
            location = f"{filename}:{line}"
            if not key:
                errors.append(f"{location}: no alphabetic tokens")
            elif key in seen:
                errors.append(f"{location}: duplicate sentence of {seen[key]}")
            else:
                seen[key] = location
    for error in errors:
        print(f"ERROR: {error}")
    if not errors:
        print("Dataset checks passed. Verify Vietnamese language and sources yourself.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(check_data())
