# UIC CS 411: Artificial Intelligence
## Project 3: Naive Bayes Text Classifier from Scratch

This repository contains the implementation, dataset, and guidelines for **Project 3: Naive Bayes Text Classifier (Vietnamese Sentiment Analysis)**.

For complete project instructions, algorithm specifications, AI policy, dataset requirements, and video presentation rubric, refer to **[Project_Guidelines.md](Project_Guidelines.md)**.

---

### Important Dates

- **Project Released:** October 26, 2026, 12:00 AM
- **Submission Deadline:** November 9, 2026, 11:59 PM
- **Late Submission Deadline:** November 12, 2026, 11:59 PM

- **Late Policy**: Submissions are accepted without penalty until November 12 at 11:59 PM (Late Submission Deadline).
- However, to qualify for the Non-English Language Bonus (+30%), students must submit their project by the **Submission Deadline (November 9 at 11:59 PM)** and book an interview (36 interview slots are available on November 11, 18, and 21, on a first-come, first-served basis.).
- Submissions after November 12 at 11:59 PM will not be accepted.

---
## Getting Started

### 1. Set Up Python Virtual Environment (Optional)
```bash
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 2. Run Your Classifier
```bash
python app.py
```
This executes `app.py` to train on `data/train.csv` and evaluate on `data/test.csv`. The classifier scored **85%** (34 of 40 correct) on the held-out test set.

---

## Project Guidelines

Please read **[Project_Guidelines.md](Project_Guidelines.md)** carefully before starting. It covers:
- **AI Policy & Declaration:** Allowed vs. prohibited AI use; mandatory PETRA AI Student Declaration (`AI_declaration.png`).
- **Dataset Preparation:** 100 positive and 100 negative sentences (80/80 train, 20/20 test); AI generation of data is strictly prohibited. This project uses the first 100 entries of each class from the [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) training split, skipping neutral entries. The selection method is documented in `video_link.txt`.
- **Algorithm Specifications:** Bag-of-words counting, class priors, Laplace smoothing, log-space scoring, and tie-breaking rules.
- **Video Presentation Rubric (30 marks):** 5–7 minute video recording walkthrough (no written report).
- **+30% Non-English Language Bonus:** Guidelines and interview booking link for non-English language submissions.

---

## Submission Instructions

Submit all files listed in the directory structure (including `app.py`, `data/train.csv`, `data/test.csv`, `video_link.txt`, `AI_declaration.png`, and `requirements.txt`) to **Gradescope** under **Project 3**.

For complete requirements and the evaluation rubric, refer to [Project_Guidelines.md](Project_Guidelines.md).
