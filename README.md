# NLP Product Classification on Etsy Marketplace Data

**Academic Research Project — University College Cork, Ireland**
*A Systematic Approach to Feature Reduction and Encoding for Efficient Text Classification*

---

## Overview

This project builds a scalable NLP classification pipeline on real Etsy marketplace data, tackling a dual classification problem:

- **Top-level category prediction** — 12–15 product categories (e.g. Home & Living, Clothing, Jewellery)
- **Bottom-level category prediction** — 2,900 granular product sub-categories

Using only the product listing title and structured attributes (material, style, occasion, etc.), the system automatically classifies products — mimicking a real e-commerce search, tagging, and recommendation problem.

---

## Key Results

| Task | Model | Accuracy | Weighted F1 |
|---|---|---|---|
| Top Category | Logistic Regression | **80.50%** | **80.14%** |
| Bottom Category | Random Forest (300 estimators) | — | — |

### Top Category — Per Class Performance (Selected)

| Category | Precision | Recall | F1-Score |
|---|---|---|---|
| Category 12 (best) | 0.92 | 0.91 | **0.92** |
| Category 6 | 0.83 | 0.85 | 0.84 |
| Category 5 | 0.90 | 0.91 | 0.90 |
| Category 8 | 0.73 | 0.91 | 0.81 |
| Category 1 (weakest) | 0.83 | 0.52 | 0.64 |

---

## Pipeline — 9 Stages

```
Raw Text (product titles)
        ↓
1. Text Preprocessing (NLP)
   · Lowercasing
   · Punctuation removal
   · Tokenisation (NLTK)
   · Stop word removal
   · Lemmatisation
        ↓
2. TF-IDF Vectorisation
   · Converts text to numerical feature matrix
   · Emphasises rare but important terms
        ↓
3. Chi-Squared Feature Selection (χ²)
   · Retains top 2,000 most predictive terms
   · Reduces dimensionality significantly
        ↓
4. Ordinal Encoding
   · Encodes 11 categorical fields (material, style,
     occasion, pattern, etc.) into numerical values
        ↓
5. Feature Merging (hstack)
   · TF-IDF features + encoded categorical features
   · Combined into unified sparse feature matrix
        ↓
6. Best-K Feature Selection
   · Final refinement of combined feature set
        ↓
7. Model Training
   · Logistic Regression → Top Category
   · Random Forest       → Bottom Category
        ↓
8. Batch Inference
   · Predictions generated in batches (memory efficient)
        ↓
9. Evaluation
   · Accuracy, Precision, Recall, Weighted F1-Score
```

---

## Why Two Different Models?

**Logistic Regression** was chosen for top-category prediction because the problem has only 12–15 classes — LR with the liblinear solver handles high-dimensional sparse text features efficiently at this scale.

**Random Forest (300 estimators)** was chosen for bottom-category prediction because there are 2,900 classes — Logistic Regression struggles with extreme multi-class problems at this scale, while Random Forest's ensemble approach handles it more robustly.

---

## Dataset

| Property | Value |
|---|---|
| Format | Parquet |
| Rows | 25,514 product listings |
| Columns | 15 |
| Primary text feature | `title` (product listing title) |
| Categorical features | material, style, occasion, pattern, + 7 others |
| Top categories | 12–15 classes |
| Bottom categories | ~2,900 classes |

### Category Distribution (from EDA)
- **Home & Living** dominates with 50,000+ listings
- **Craft Supplies & Tools** and **Clothing** follow (~33K and ~30K)
- **Pet Supplies, Books & Music, Bags & Purses** are underrepresented (<7K each)
- This class imbalance was a key challenge — addressed using weighted F1-score evaluation

### Word Cloud Insights
Most frequent terms in product titles: **custom, vintage, gift, personalized, handmade** — reflecting Etsy's artisan marketplace nature. Domain-specific stopwords (free, new, shipping) were excluded from analysis.

---

## Tech Stack

- **Language:** Python
- **NLP:** NLTK (tokenisation, lemmatisation, stop word removal)
- **Features:** Scikit-learn TF-IDF, Chi-Squared, Ordinal Encoder
- **Models:** Logistic Regression, Random Forest
- **Data:** Pandas, NumPy, SciPy (sparse matrices)
- **Visualisation:** Matplotlib, WordCloud

---

## Repository Contents

```
Etsy-Market-place/
│
├── ETSY_Prediction.ipynb                        # Full pipeline notebook
├── A_Systematic_Approach_to_Feature_...pdf      # Research paper
└── README.md
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Suhasgoli99/Etsy-Market-place.git
cd Etsy-Market-place
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn nltk matplotlib wordcloud pyarrow jupyter
```

**3. Download NLTK data (first time only)**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

**4. Open the notebook**
```bash
jupyter notebook ETSY_Prediction.ipynb
```

> **Note:** The dataset is in Parquet format. Ensure `pyarrow` or `fastparquet` is installed to read it.

---

## Key Findings

1. **Feature engineering matters more than model complexity** — a carefully designed NLP pipeline with Logistic Regression outperforms complex models on sparse text data
2. **Chi-Squared feature selection** is highly effective at reducing TF-IDF dimensionality without significant accuracy loss
3. **Class imbalance** (Home & Living dominating) affects per-category performance — Category 8 shows high recall but lower precision due to its large representation in the dataset
4. **Combining text and categorical features** via sparse matrix operations is computationally efficient and improves classification over text-only approaches

---

## Research Paper

The full methodology, literature review, results, and analysis are documented in the included research paper:
**"A Systematic Approach to Feature Reduction and Encoding for Efficient Text Classification"**
*Janakiram Suhas Goli — University College Cork, 2024*

---

## Author

**Janakiram Suhas Goli**
MSc Data Science & Analytics — University College Cork, Ireland
[github.com/Suhasgoli99](https://github.com/Suhasgoli99) | suhasgoli99@gmail.com
