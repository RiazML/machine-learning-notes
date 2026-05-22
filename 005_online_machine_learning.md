# Online Machine Learning (Incremental Learning)

To address the latency, storage, and cost limitations of Batch Learning, modern systems often employ **Online Machine Learning** (also known as **Incremental Learning**).

---

## 1. What is Online Machine Learning?

In Online Learning, the model is not trained on a static dataset all at once. Instead, it ingests data continuously, either as **individual data points** or in **small groups called mini-batches**. As new data arrives in production, the model analyzes it, updates its mathematical parameters (weights) instantly, and immediately discards the data (unless needed for backup).

```
                      [Continuous Streaming Data]
                                   │
                                   ▼
┌──────────────────┐       ┌───────────────┐
│ Live Predictions │ ◄──── │ Active Model  │
└──────────────────┘       └───────▲───────┘
                                   │ Update weights instantly
                               ┌───┴───┐
                               │ Train │
                               └───────┘
```

---

## 2. Key Concepts in Online Learning

### A. Learning Rate ($\eta$)

The learning rate is a hyperparameter that controls how quickly the model adapts to new data.

- **High Learning Rate**: The model updates its weights aggressively in response to new data. It will adapt to new trends rapidly, but it will also **forget old patterns quickly** (referred to as _high plasticity_).
- **Low Learning Rate**: The model updates its weights slowly. It is more stable and retains historical memory well, but it will be sluggish to adapt to sudden changes in the data stream (referred to as _high stability_).

### B. Out-of-Core Learning

Online learning algorithms can be used to train models on datasets that are too large to fit into a computer's RAM (Out-of-Core Learning).

- Instead of loading a 500GB dataset into 64GB of RAM, the system reads the data from disk in small chunks of 100MB.
- The model trains on one chunk, updates its weights, discards the chunk from memory, and then loads the next chunk. This allows training on infinite datasets using limited hardware.

---

## 3. Real-World Use Cases

1. **Smart Keyboards (Gboard, SwiftKey)**: When you type on your smartphone, the keyboard learns your personal slang, abbreviations, and writing style in real-time. It updates its prediction model locally on your device without needing to retrain on a central server.
2. **YouTube/Netflix Home Feed**: When you watch a new type of video (e.g., suddenly watching a cooking video), the homepage algorithm updates your recommendations within minutes. It does not wait for a weekly batch retraining cycle.
3. **Financial Fraud Detection**: Credit card usage patterns shift dynamically. Online models adapt to the latest transaction signatures immediately to flag fraudulent activities.

---

## 4. Risks & Challenges of Online Learning

While highly responsive, Online Learning introduces unique operational risks:

### A. Model Poisoning (Bad Data Ingestion)

If the input data stream in production deteriorates or becomes malicious, the model will learn from it and degrade instantly.

- **Example: Chatbot Corruption**. In 2016, Microsoft released an AI chatbot named _Tay_ on Twitter. It was designed to learn from interactions with users in real-time. Within 24 hours, malicious users coordinated to feed it offensive and inflammatory statements. The chatbot's online model ingested this data and immediately began outputting highly offensive tweets, forcing Microsoft to shut it down.
- **Mitigation**: Systems must implement strict input anomaly detection, data cleaning filters, and the ability to roll back the model's weights to a previous safe state.

### B. Catastrophic Forgetting

If an online model is exposed to a long sequence of data from a new distribution, it may completely overwrite its weights, erasing the mathematical relationships it learned from previous distributions.

- **Example**: A recommendation engine exposed only to summer apparel data for three months might completely forget how to recommend winter jackets.

---

## Summary Comparison: Batch vs. Online Learning

| Dimension           | Batch Learning                           | Online Learning                                                 |
| :------------------ | :--------------------------------------- | :-------------------------------------------------------------- |
| **Data Ingestion**  | Static dataset, processed in bulk.       | Continuous stream, processed point-by-point or in mini-batches. |
| **Model Updates**   | Offline, manual redeployment.            | Online, automatic real-time weight updates.                     |
| **RAM Requirement** | High (must load the whole dataset).      | Low (only loads the current data point/mini-batch).             |
| **Adapts to Drift** | Slow (requires full retraining).         | Fast (adapts dynamically to real-time changes).                 |
| **Risk Profile**    | Low (fully validated before deployment). | High (vulnerable to model poisoning and forgetting).            |
| **Hardware Focus**  | Centralized heavy GPU clusters.          | Lightweight edge processors, streaming servers.                 |
