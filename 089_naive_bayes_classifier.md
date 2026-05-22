# Naive Bayes Classifier Part 8: Bag-of-Words Text Classification

The most prominent real-world application of the Naive Bayes classifier is text classification (e.g., spam filtering, sentiment analysis, document tagging). To feed text into the classifier, we must convert it into numeric vectors using the **Bag-of-Words (BoW)** representation and apply **Multinomial Naive Bayes**.

---

## 1. Bag-of-Words Representation

The Bag-of-Words model simplifies text by discarding grammar, sentence structure, and word order, focusing entirely on **word frequencies**.

1. **Vocabulary Construction**: We compile a list of all unique words (tokens) appearing across all training documents. Let this vocabulary be $V$.
2. **Vectorization**: Each document is represented as a vector of length $|V|$, where index $j$ holds the frequency count of word $w_j$ in that document.

---

## 2. Multinomial Naive Bayes Formulation

In **Multinomial Naive Bayes**, we model the text generation process as drawing words from a multinomial distribution for each class.

```mermaid
flowchart TD
    Doc["Input Text Document"] --> Tokenize["Tokenize and lower-case words"]
    Tokenize --> BoW["Convert to Word Count Vector f_w"]
    BoW --> SumLogs["Sum log-likelihoods: ∑ f_w * log P("w | c") + log P("c")"]
    SumLogs --> OutputClass["Predict class with highest score"]
```

### Word Likelihood with Laplace Smoothing

For class $c$, the likelihood of word $w$ is computed as:
$$P(w \mid c) = \frac{N_{wc} + \alpha}{N_c + \alpha \cdot |V|}$$

Where:

- $N_{wc}$ is the total number of times word $w$ appears in all training documents of class $c$.
- $N_c$ is the total count of **all** words in all training documents of class $c$:
  $$N_c = \sum_{w' \in V} N_{w'c}$$
- $|V|$ is the size of the vocabulary.
- $\alpha$ is the smoothing parameter ($\alpha = 1$ for Laplace).

### Document Classification Rule

For a test document $d$ represented as word counts $f_w$, the classification score in log-space is:
$$\log P(c \mid d) \propto \log P(c) + \sum_{w \in V} f_w \cdot \log P(w \mid c)$$

---

## 3. Python Implementation from Scratch

The following runnable Python script implements a complete text preprocessor (lowercase, punctuation removal, tokenization) and a `CustomMultinomialNB` classifier from scratch. It trains on a mock spam dataset, predicts on test documents, and verifies that the output matches Scikit-Learn's `CountVectorizer` + `MultinomialNB` pipeline exactly.

```python
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Custom Text Preprocessor & Classifier from Scratch
def tokenize(text):
    # Convert to lowercase and split by non-word characters
    words = re.findall(r'\b\w+\b', text.lower())
    return words

class CustomMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.vocabulary = None
        self.class_priors = {}
        # Likelihoods dict: class -> word -> probability
        self.word_likelihoods = {}
        self.classes = None

    def fit(self, documents, y):
        # Build vocabulary
        all_words = []
        tokenized_docs = [tokenize(doc) for doc in documents]
        for tokens in tokenized_docs:
            all_words.extend(tokens)
        self.vocabulary = sorted(list(set(all_words)))
        vocab_size = len(self.vocabulary)

        self.classes = np.unique(y)
        n_samples = len(documents)

        # Initialize counting containers
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
            self.word_likelihoods[c] = {}

            # Filter docs in class c
            docs_in_class = [tokenized_docs[i] for i in range(n_samples) if y[i] == c]

            # Count word occurrences
            word_counts = {word: 0 for word in self.vocabulary}
            total_words_in_class = 0
            for doc in docs_in_class:
                for word in doc:
                    if word in word_counts:
                        word_counts[word] += 1
                        total_words_in_class += 1

            # Compute smoothed word likelihoods P(w | c)
            for word in self.vocabulary:
                num = word_counts[word] + self.alpha
                den = total_words_in_class + self.alpha * vocab_size
                self.word_likelihoods[c][word] = num / den

    def predict_log_proba(self, test_documents):
        preds_log_probs = []
        for doc in test_documents:
            tokens = tokenize(doc)
            # Count frequencies in test doc
            doc_word_counts = {}
            for token in tokens:
                if token in self.vocabulary:
                    doc_word_counts[token] = doc_word_counts.get(token, 0) + 1

            sample_log_probs = {}
            for c in self.classes:
                log_prob = np.log(self.class_priors[c])
                for word, count in doc_word_counts.items():
                    log_prob += count * np.log(self.word_likelihoods[c][word])
                sample_log_probs[c] = log_prob
            preds_log_probs.append(sample_log_probs)
        return preds_log_probs

    def predict(self, test_documents):
        log_probs = self.predict_log_proba(test_documents)
        preds = []
        for doc_probs in log_probs:
            preds.append(max(doc_probs, key=doc_probs.get))
        return np.array(preds)

# 2. Setup training dataset
train_docs = [
    "Free offer click here now",
    "Meeting scheduled for tomorrow morning",
    "Claim your free money prize today",
    "Can we review the project project status tomorrow?",
    "Earn money fast from home offer"
]
# Labels: 1 = Spam, 0 = Ham
y_train = np.array([1, 0, 1, 0, 1])

# 3. Train both Custom and Sklearn models
custom_nb = CustomMultinomialNB(alpha=1.0)
custom_nb.fit(train_docs, y_train)

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', lowercase=True)
X_train_vectorized = vectorizer.fit_transform(train_docs).toarray()

sklearn_nb = MultinomialNB(alpha=1.0)
sklearn_nb.fit(X_train_vectorized, y_train)

# 4. Predict on test documents
test_docs = [
    "Meeting tomorrow project",
    "Claim free prize now"
]
X_test_vectorized = vectorizer.transform(test_docs).toarray()

custom_preds = custom_nb.predict(test_docs)
sklearn_preds = sklearn_nb.predict(X_test_vectorized)

# Calculate and compare joint log likelihoods
custom_log_probs = custom_nb.predict_log_proba(test_docs)
sklearn_log_probs = sklearn_nb._joint_log_likelihood(X_test_vectorized)

print("=== Predictions Comparison ===")
print("Custom Predictions: ", custom_preds)
print("Sklearn Predictions:", sklearn_preds)

print("\n=== Word Likelihood Verification P(free | Spam) ===")
# Sklearn logs features. e^log = probability
free_idx = vectorizer.vocabulary_['free']
sklearn_prob_free_spam = np.exp(sklearn_nb.feature_log_prob_[1, free_idx])
custom_prob_free_spam = custom_nb.word_likelihoods[1]['free']
print(f"Custom likelihood P(free|Spam):  {custom_prob_free_spam:.6f}")
print(f"Sklearn likelihood P(free|Spam): {sklearn_prob_free_spam:.6f}")

# Assert matching joint log likelihoods, probabilities, and predictions
assert np.all(custom_preds == sklearn_preds), "Predictions do not match!"
assert np.isclose(custom_prob_free_spam, sklearn_prob_free_spam), "Word probabilities do not match!"
for i in range(len(test_docs)):
    for c_idx, c in enumerate(custom_nb.classes):
        assert np.isclose(custom_log_probs[i][c], sklearn_log_probs[i, c_idx]), "Joint log likelihoods do not match!"
print("\n[SUCCESS] Custom Multinomial Naive Bayes text classifier matches Scikit-Learn exactly!")
```

---

- **Next Topic**: [090_naive_bayes_part_9.md](file:///Users/prime/Developer/ml/090_naive_bayes_part_9.md) - Gaussian Naive Bayes for continuous data.
