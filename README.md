# Vietnamese Naive Bayes Sentiment Classifier

A standard-library-only Multinomial Naive Bayes classifier for Vietnamese text.
The CSV files contain 200 unchanged Vietnamese sentences from UIT-VSFC, selected
from the beginning of its training split as requested. See [Project_Guidelines.md](Project_Guidelines.md)
for the assignment rules.

## Prepare your data

Source: [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback),
a Vietnamese student-feedback corpus. Starting at source row 0 of its training
split, select the first 100 positive and first 100 negative entries, skipping
neutral entries. For each class, the first 80 entries go into training and the
next 20 into testing. Preserve source order within each output file. Original
wording is unchanged; sentiment 2 maps to `positive`, 0 to `negative`, and topic
labels are omitted. The selected entries end at source row 235 (zero-based).

The user reviewed and chose this source; AI assistance downloaded and extracted
the requested entries. No sentences were generated or rephrased. The assignment
requires manual collection without AI and does not explicitly address existing
labeled corpora; this selection method should be disclosed accurately. The source
card lists its license as unknown. Citation: Nguyen et al., *UIT-VSFC: Vietnamese
Students’ Feedback Corpus for Sentiment Analysis*, KSE 2018, pp. 19–24,
DOI: 10.1109/KSE.2018.8573337.

| File | Positive | Negative | Total |
| --- | ---: | ---: | ---: |
| `data/train.csv` | 80 | 80 | 160 |
| `data/test.csv` | 20 | 20 | 40 |

Use a spreadsheet editor or CSV writer to quote text containing commas correctly.
Keep the test set separate. Avoid duplicate sentences and near duplicates across
both files. Do not adjust the training data based on test predictions.

## Run

Run these commands from the project folder with Python 3. No installation is needed.

```sh
python3 check_data.py
python3 -m unittest -v
python3 app.py
```

The checker verifies sizes, labels, nonempty text, and duplicate token sequences.
It cannot establish authorship, sentiment correctness, or language authenticity.
The unit tests use temporary abstract tokens, never submission dataset sentences.
The algorithm unit tests run independently of the project CSV files.
The current dataset passes the checker. Evaluation on the selected 40 test entries
gives 85% accuracy: TP=17, TN=17, FP=3, FN=3. This small ordered subset does not
establish performance on the full corpus or other kinds of Vietnamese text.

The app displays vocabulary size, class counts, priors, accuracy, and the confusion
matrix. Enter Vietnamese sentences for predictions. Use `:trace ` followed by your
own sentence to see tokens, likelihood fractions, log scores, and the prediction.
Enter `:quit` to exit. The confusion matrix treats positive as the positive class:
TP/TN are correct predictions; FP/FN are incorrect positive/negative predictions.

## How it works

`tokenize()` normalizes Unicode to NFC, lowercases text, preserves accents, and
extracts alphabetic runs. Vietnamese uses spaces between syllables, so these tokens
are usually syllables rather than complete multi-syllable words. This is a simple
bag-of-syllables model, with no dictionary or external tokenizer. Explain this
choice during the interview. Punctuation and numbers are discarded; accents are
retained because they distinguish Vietnamese syllables.

`train_model()` learns class sentence counts, vocabulary, and token counts from
training data only. Repeated tokens count every time. Training again replaces the
previous model, so editing the training file updates all learned probabilities.

- Prior: `P(c) = sentences_in_class / total_training_sentences`
- Laplace likelihood: `P(w|c) = (count(w,c) + 1) / (total_tokens_in_class + vocabulary_size)`
- Score: `log(P(c)) + sum(log(P(w|c)))`, including repeated known tokens

`predict()` ignores unknown tokens. Ties use the higher prior, then `positive`.
Empty input or input containing only unknown tokens therefore uses the priors.
`evaluate_model()` returns accuracy as a fraction and a TP/TN/FP/FN dictionary;
it never updates training counts.

For inspecting individual counts during your video, open Python interactively:

```python
import app
app.train_model()
app.model["counts"]       # Token counts for each class
app.model["totals"]       # Total token occurrences for each class
app.model["priors"]
```

Choose two tokens from your own data. Inspect their counts in each class, then use
`app.word_likelihood(token, label)` or `app.explain(your_sentence)` to show the math.
The model loses word order, so negation, sarcasm, and multi-syllable expressions
can be difficult. Accuracy must be measured on your actual held-out sentences.

## Video and submission

Record a 5–7 minute video with your face visible throughout. Use your actual data:

1. Show files, class balance, sources, and Vietnamese language declaration.
2. Show tokenization, vocabulary size, total tokens per class, and two token counts.
3. Work through numeric priors and smoothed likelihoods from your training counts.
4. Trace your own input and include an unknown token to demonstrate ignoring it.
5. Modify training data live, rerun, and show changed counts and scores.
   Restore the original dataset afterward.
6. Show held-out evaluation and at least three new sentences you wrote yourself;
   explain a limitation or misclassification.

Before submitting:

- Fill in your identity, sources, accessible video URL, and duration in `video_link.txt`.
- Complete the [PETRA AI declaration](https://www.petraai.org/student) accurately,
  including AI assistance with implementation, tests, documentation, and dataset
  downloading/extraction from the user-selected source. Save the
  resulting image as `AI_declaration.png` in the project root.
- Run the data checker, tests, and app with your completed dataset.
- For the Vietnamese bonus, submit by **November 9, 2026, 11:59 PM** and
  [book the required interview](https://calendar.app.google/J5EejvwkKgtPE8oKA).
  The final late deadline is November 12, but the bonus requires November 9.
- Submit `app.py`, `data/train.csv`, `data/test.csv`, `video_link.txt`,
  `AI_declaration.png`, and `requirements.txt` to Gradescope Project 3, preserving
  the folder structure. Check the official autograder results afterward.

The video, declaration, interview booking, and Gradescope submission are still
pending. The classifier and populated dataset are ready for your review.
