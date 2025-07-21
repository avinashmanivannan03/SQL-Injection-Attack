# SQL Injection Attack Detection using Deep Learning

## Overview

This project presents a comprehensive study and implementation of various deep learning and machine learning models for the detection of SQL Injection (SQLi) attacks. SQLi remains a critical vulnerability in web applications, allowing attackers to manipulate database queries to gain unauthorized access, exfiltrate sensitive data, or corrupt the database.

The primary objective of this project is to develop and evaluate an AI-driven system capable of automatically and accurately detecting SQLi attempts in real-time.

This repository explores and compares the performance of four advanced models:

- Graph Neural Network (GNN)
- Contrastive Learning with Siamese Networks
- Transformer (BERT) + PCA + SVM (Hybrid Approach)
- Bidirectional LSTM (BiLSTM) with Attention Mechanism

---

## Table of Contents

- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training the Models](#training-the-models)
  - [Making Predictions](#making-predictions)
- [Results and Conclusion](#results-and-conclusion)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Dataset Description

The project utilizes the SQL Injection Dataset available on Kaggle. This dataset comprises website traffic data, where each entry is a SQL query string. The queries are labeled as either malicious (`1`) or benign (`0`).

**Dataset Source**: Kaggle SQL Injection Dataset

### Data Distribution

| Dataset File              | Benign Queries (Label 0) | Malicious Queries (Label 1) | Total    |
|---------------------------|---------------------------|------------------------------|----------|
| trainingdata.csv          | 42,360                    | 55,915                       | 98,275   |
| testingdata.csv           | 13,134                    | 11,573                       | 24,707   |
| testinglongdata_500.csv   | 100                       | 400                          | 500      |
| testinglongdatav2.csv     | 20                        | 300                          | 320      |

---

## Project Structure

The codebase is organized into the following Jupyter Notebooks:

- `CYBER_DATASET_DESCRIPTION.ipynb`: Exploratory Data Analysis (EDA) and statistics.
- `cyber-gnn-final.ipynb`: Implementation of Graph Neural Network (GNN).
- `CYBER_CONTRASTIVE_LEARNING_WORKING.ipynb`: Contrastive learning using a Siamese Network.
- `CYBER_TRANSFORMER+AUTOENCODER.ipynb`: Transformer + PCA + One-Class SVM model.
- `CYBER_BILSTM+ATTENTION_MECHANISM.ipynb`: BiLSTM with Attention Mechanism.

---

## Models Implemented

### 1. Graph Neural Network (GNN)

- **Notebook**: `cyber-gnn-final.ipynb`
- **Approach**: SQL queries are parsed into token-based graphs. GCN layers extract relational structure and semantics.
- **Strength**: Captures syntax and structural dependencies within queries.
- **Accuracy**: 96.10%

### 2. Contrastive Learning with Siamese Networks

- **Notebook**: `CYBER_CONTRASTIVE_LEARNING_WORKING.ipynb`
- **Approach**: Learns vector representations using triplet loss to separate benign and malicious queries.
- **Strength**: Effective for generalizing to zero-day and previously unseen attacks.
- **Accuracy**: 95.37%

### 3. Transformer + PCA + SVM (Hybrid Approach)

- **Notebook**: `CYBER_TRANSFORMER+AUTOENCODER.ipynb`
- **Approach**: Embeddings from Sentence-BERT reduced using PCA, then passed to One-Class SVM.
- **Strength**: Lightweight inference; hybrid approach combining semantic understanding with anomaly detection.
- **Accuracy**: 63.16%

### 4. BiLSTM with Attention Mechanism

- **Notebook**: `CYBER_BILSTM+ATTENTION_MECHANISM.ipynb`
- **Approach**: BiLSTM models sequence flow; attention layers highlight important tokens.
- **Strength**: Captures contextual token dependencies.
- **Accuracy**: 53.15%

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- CUDA-enabled GPU (recommended)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sql-injection-detection.git
   cd sql-injection-detection
   ````

2. Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   **Sample `requirements.txt`:**

   ```
   pandas
   numpy
   torch
   torch-geometric
   networkx
   nltk
   gensim
   sqlparse
   scikit-learn
   tensorflow
   transformers
   imbalanced-learn
   matplotlib
   ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/) and extract all CSV files into the root directory.

---

## Usage

### Training the Models

1. Open any notebook using Jupyter Notebook, VS Code, or Google Colab.
2. Run all cells sequentially.
3. Models are trained and stored automatically for prediction/inference.

### Making Predictions

A simplified example using the GNN model:

```python
import torch
import sqlparse
import networkx as nx
from gensim.models import Word2Vec
from torch_geometric.data import Data

model = torch.load('gnn_model.pth')
model.eval()

w2v_model = Word2Vec.load("word2vec.model")

def tokenize_sql(query):
    parsed = sqlparse.parse(query)
    return [t.value.lower() for s in parsed for t in s.tokens if t.value.strip()]

def build_graph_for_prediction(query, word2vec_model):
    tokens = tokenize_sql(query)
    # Add actual graph construction logic here...
    pass  # Placeholder

def predict(query_string):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    graph_data = build_graph_for_prediction(query_string, w2v_model).to(device)
    
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index, graph_data.batch)
        pred = out.argmax(dim=1).item()
    
    return "Malicious (SQL Injection)" if pred == 1 else "Benign"

if __name__ == "__main__":
    test_query_1 = "SELECT * FROM users WHERE id = '1' OR '1'='1'"
    test_query_2 = "SELECT product_name FROM products WHERE category = 'Gifts'"

    print(f"Query: '{test_query_1}' -> Prediction: {predict(test_query_1)}")
    print(f"Query: '{test_query_2}' -> Prediction: {predict(test_query_2)}")
```

---

## Results and Conclusion

| Model                            | Description                                                             | Accuracy |
| -------------------------------- | ----------------------------------------------------------------------- | -------- |
| Graph Neural Network (GNN)       | Captures structural and syntactic token relationships in SQL queries    | 96.10%   |
| Contrastive Learning             | Learns discriminative embeddings via Siamese Network with Triplet Loss  | 95.37%   |
| Transformer + PCA + SVM (Hybrid) | Sentence-BERT for embeddings, PCA for reduction, SVM for classification | 63.16%   |
| BiLSTM with Attention            | Sequence model with contextual awareness and attention weighting        | 53.15%   |

### Observations

* GNNs are most effective due to their ability to encode structural semantics.
* Contrastive learning generalizes well, especially for previously unseen (zero-day) attacks.
* Transformer-based hybrid models need careful tuning to avoid underfitting.
* Sequence-only models like BiLSTM fall short in detecting complex SQLi patterns.

---

## Future Work

Several enhancements are proposed:

1. **Model Ensembling**
   Combine GNN and Contrastive Learning outputs for improved robustness using:

   * Voting ensembles
   * Weighted fusion
   * Meta-learning

2. **Zero-Day Attack Simulation**
   Generate adversarial or novel attack payloads to test generalization.

3. **API Deployment**
   Deploy the models via RESTful API using FastAPI or Flask for real-time SQLi monitoring.

4. **Model Explainability**
   Apply LIME, SHAP, or attention heatmaps for interpretability.

5. **Active Learning Framework**
   Integrate human feedback loops to retrain on edge-case queries and reduce false positives.

---

## Contributing

We welcome contributions from developers, researchers, and enthusiasts. Whether it's model improvement, documentation, or bug fixes—your help is appreciated.

### How to Contribute

1. **Fork the repository**
   Click the **Fork** button on the top-right of this repository.

2. **Clone your fork**

   ```bash
   git clone https://github.com/your-username/sql-injection-detection.git
   cd sql-injection-detection
   ```

3. **Create a new feature branch**

   ```bash
   git checkout -b feature/AmazingFeature
   ```

4. **Make your changes** and commit

   ```bash
   git commit -m "Add: AmazingFeature"
   ```

5. **Push the branch**

   ```bash
   git push origin feature/AmazingFeature
   ```

6. **Open a Pull Request** on GitHub. Provide a clear description and link any related issues.

---



