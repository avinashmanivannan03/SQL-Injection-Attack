# SQL Injection Attack Detection using Deep Learning

## Overview

This project presents a comprehensive study and implementation of various deep learning and machine learning models for the detection of SQL Injection (SQLi) attacks. SQLi remains a critical vulnerability in web applications, allowing attackers to manipulate database queries to gain unauthorized access, exfiltrate sensitive data, or corrupt the database. The primary objective of this project is to develop and evaluate an AI-driven system capable of automatically and accurately detecting SQLi attempts in real-time.

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
- [Contributing](#contributing)  
- [License](#license)

---

## Dataset Description

The project utilizes the **SQL Injection Dataset** available on Kaggle. This dataset comprises website traffic data, where each entry is a SQL query string. The queries are labeled as either malicious (`1`) or benign (`0`).

**Dataset Source**: Kaggle SQL Injection Dataset

### Data Distribution

| Dataset File             | Benign Queries (Label 0) | Malicious Queries (Label 1) | Total   |
|--------------------------|---------------------------|------------------------------|---------|
| `trainingdata.csv`       | 42,360                    | 55,915                       | 98,275  |
| `testingdata.csv`        | 13,134                    | 11,573                       | 24,707  |
| `testinglongdata_500.csv`| 100                       | 400                          | 500     |
| `testinglongdatav2.csv`  | 20                        | 300                          | 320     |

---

## Project Structure

The project's codebase is organized into the following Jupyter notebooks:

- **`CYBER_DATASET_DESCRIPTION.ipynb`**  
  Provides a detailed exploratory data analysis (EDA) of the datasets.

- **`cyber-gnn-final.ipynb`**  
  Implements a Graph Neural Network (GNN) for SQLi detection.

- **`CYBER_CONTRASTIVE_LEARNING_WORKING.ipynb`**  
  Implements contrastive learning using a Siamese Network with Triplet Loss.

- **`CYBER_TRANSFORMER+AUTOENCODER.ipynb`**  
  Implements a hybrid model using BERT, PCA, and an Autoencoder.

- **`CYBER_BILSTM+ATTENTION_MECHANISM.ipynb`**  
  Implements a BiLSTM with attention mechanism for classification.

---

## Models Implemented

### 1. Graph Neural Network (GNN)

- **Notebook**: `cyber-gnn-final.ipynb`  
- **Methodology**: SQL queries modeled as graphs with GCNs learning structural syntax.  
- **Why GNN?** Captures complex structural relationships.  
- **Accuracy**: **96.10%**

---

### 2. Contrastive Learning

- **Notebook**: `CYBER_CONTRASTIVE_LEARNING_WORKING.ipynb`  
- **Methodology**: Siamese Network with Triplet Loss learns distance-based embeddings.  
- **Why Contrastive Learning?** Generalizes well to zero-day attacks.  
- **Accuracy**: **95.37%**

---

### 3. Transformer + PCA + SVM (Hybrid Approach)

- **Notebook**: `CYBER_TRANSFORMER+AUTOENCODER.ipynb`  
- **Methodology**: Sentence Transformer for embeddings + PCA + One-Class SVM.  
- **Why Hybrid?** Leverages Transformer power with efficient anomaly detection.  
- **Accuracy**: **63.16%**

---

### 4. BiLSTM with Attention Mechanism

- **Notebook**: `CYBER_BILSTM+ATTENTION_MECHANISM.ipynb`  
- **Methodology**: Sequence modeling of SQL queries with focus on important tokens.  
- **Accuracy**: **53.15%**

---

## Getting Started

### Prerequisites

- Python 3.9 or higher  
- Pip and optionally Conda  
- CUDA-enabled GPU (recommended)

---

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/sql-injection-detection.git
    cd sql-injection-detection
    ```

2. **Set up virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    **requirements.txt** should include:
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
    imblearn
    matplotlib
    ```

4. **Download Dataset**:
    - Download from [Kaggle](https://www.kaggle.com/)
    - Extract `.csv` files into project root directory.

---

## Usage

### Training the Models

- Open any notebook (e.g., `cyber-gnn-final.ipynb`) in **Jupyter Lab**, **Colab**, or **VS Code**.
- Run cells sequentially. Notebooks are self-contained.
- Trained models will be saved for later inference.

---

## Making Predictions

Here's a sample prediction script for the **GNN model**:

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
    # Graph building logic here
    pass  # Placeholder

def predict(query_string):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    graph_data = build_graph_for_prediction(query_string, w2v_model).to(device)
    
    with torch.no_grad():
        out = model(graph_data.x, graph_data.edge_index, graph_data.batch)
        pred = out.argmax(dim=1).item()
        
    return "Malicious (SQL Injection)" if pred == 1 else "Benign"

if __name__ == '__main__':
    test_query_1 = "SELECT * FROM users WHERE id = '1' OR '1'='1'"
    test_query_2 = "SELECT product_name FROM products WHERE category = 'Gifts'"
    
    print(f"Query: '{test_query_1}' -> Prediction: {predict(test_query_1)}")
    print(f"Query: '{test_query_2}' -> Prediction: {predict(test_query_2)}")

## 📊 Results and Conclusion

After rigorous experimentation and evaluation, the performance of all implemented models was analyzed based on their accuracy on the test datasets. The findings are summarized below:

| Model                                 | Description                                                                                   | Accuracy   |
|--------------------------------------|-----------------------------------------------------------------------------------------------|------------|
| **Graph Neural Network (GNN)**       | Captures the structural and syntactic relationships between tokens in a SQL query.           | **96.10%** |
| **Contrastive Learning**             | Learns discriminative embeddings using Siamese Network with Triplet Loss.                    | **95.37%** |
| **Transformer + PCA + SVM (Hybrid)** | Uses Sentence-BERT for embedding, PCA for dimensionality reduction, and SVM for classification. | 63.16%     |
| **BiLSTM with Attention**            | Sequence-based modeling using bidirectional LSTM layers and attention mechanism.             | 53.15%     |

### ✅ Key Takeaways:

- **Graph Neural Network (GNN)** emerged as the most effective model, demonstrating that capturing the graph-based structure of SQL queries is critical to accurately identifying complex and obfuscated SQL Injection attacks.

- **Contrastive Learning with Siamese Networks** performed nearly as well as GNN and showed excellent potential in generalizing to **zero-day attacks** due to its metric learning approach.

- The **Transformer-based hybrid approach**, while promising in theory, showed relatively poor performance likely due to noise and lack of optimization in the anomaly detection setup with One-Class SVM.

- **BiLSTM with Attention**, although good at modeling sequences, did not perform competitively—highlighting that pure sequence modeling may be insufficient for detecting subtle SQLi patterns.

---

### 🚀 Future Work

To further enhance the robustness, accuracy, and generalization capabilities of the system, the following directions are proposed:

- **Ensemble Learning**:  
  Combine the strengths of the top two models—**GNN** and **Contrastive Learning**—in an ensemble architecture. Potential strategies include:
  - Majority voting
  - Confidence-weighted ensemble
  - Meta-learning (stacked generalization)

- **Zero-Day Attack Simulation**:  
  Introduce synthetically generated SQLi queries with novel obfuscation and payloads to evaluate and improve model robustness.

- **Real-Time API Deployment**:  
  Wrap the best-performing models into a lightweight, RESTful API for integration into web application firewalls (WAFs) or Intrusion Detection Systems (IDS).

- **Explainability**:  
  Integrate tools such as LIME, SHAP, or attention heatmaps to explain why a query is classified as malicious, aiding cybersecurity analysts.

- **Active Learning Pipeline**:  
  Allow human-in-the-loop training to continually improve model performance on unseen data through user feedback loops.

---

## 🤝 Contributing

We welcome contributions from developers, researchers, and cybersecurity enthusiasts to make this project even better! Whether it's improving model performance, fixing bugs, enhancing documentation, or adding new features—your contributions are highly valued.

### 💡 How to Contribute

1. **Fork the Repository**  
   Click on the **Fork** button in the upper right-hand corner of this repository's page to create your own copy.

2. **Clone your forked repository**
   ```bash
   git clone https://github.com/your-username/sql-injection-detection.git
   cd sql-injection-detection

