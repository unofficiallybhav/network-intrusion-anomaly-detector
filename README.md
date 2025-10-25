# 🧠 Network Intrusion Anomaly Detector

An **AI-driven anomaly detection system** for identifying malicious network activity using a mix of **unsupervised learning**, **supervised classifiers**, and **feature selection** techniques.  
This project demonstrates how modern ML and explainable AI methods can be used for **cyber threat hunting** in network intrusion datasets.

---

## 🚀 Project Overview

This repository implements a modular **Network Intrusion Detection System (NIDS)** pipeline capable of detecting anomalies in network traffic using both supervised and unsupervised approaches.

The project is built around a **scalable, research-grade layout**, separating data handling, model training, and analysis into clean modules.

### 🧩 Core Features

- **Data preprocessing pipeline**: Cleans, encodes, and scales network flow data.  
- **Unsupervised anomaly detection**: Implements Isolation Forest, One-Class SVM, and Local Outlier Factor.  
- **Supervised classification**: Uses LightGBM, Random Forest, and XGBoost for attack classification.  
- **Dimensionality reduction**: PCA and feature selection via mutual information and feature importance.  
- **Parameter tuning**: Automated hyperparameter optimization with Optuna/GridSearchCV.  
- **Evaluation metrics**: ROC-AUC, F1-score, confusion matrix, and feature importance plots.  
- **Extensible structure**: Plug-and-play support for new algorithms or data sources.

---

## 📁 Project Structure
network-intrusion-anomaly-detector/
│
├── data/ # Raw and processed datasets
│ ├── raw/
│ └── processed/
│
├── notebooks/ # Experiment notebooks (EDA, model training, etc.)
│ ├── eda.ipynb
│ ├── unsupervised_models.ipynb
│ ├── supervised_models.ipynb
│ └── model_results.ipynb
│
├── outputs/ # Saved models, plots, reports, and metrics
│
├── src/ # Core source code
│ ├── preprocessing.py # Data cleaning and feature encoding
│ ├── feature_selection.py # Feature importance and reduction
│ ├── unsupervised_models.py# Isolation Forest, One-Class SVM, etc.
│ ├── supervised_models.py # LightGBM, Random Forest, etc.
│ ├── tuning.py # Hyperparameter tuning
│ ├── utils.py # Helper functions and metrics
│ └── main.py # Orchestrator script
│
├── requirements.txt # Dependencies
├── LICENSE # License (MIT/Apache recommended)
└── README.md # Project documentation



## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/unofficiallybhav/network-intrusion-anomaly-detector.git
cd network-intrusion-anomaly-detector

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

🧮 Usage
Run the main pipeline
python src/main.py


You can configure your run by editing parameters inside main.py — for example, choose between supervised or unsupervised workflows.

Run experiments via notebooks

If you prefer interactive experimentation:

jupyter notebook notebooks/


eda.ipynb — Exploratory data analysis

unsupervised_models.ipynb — Isolation Forest, One-Class SVM, LOF

supervised_models.ipynb — LightGBM, Random Forest, etc.

model_results.ipynb — ROC, F1, confusion matrix, feature importances

📊 Example Results
Model	            ROC-AUC	        F1-Score
Isolation Forest	0.692	        0.58
One-Class SVM	    0.631	        0.49
LightGBM	        0.91	        0.87

Feature importance and dimensionality reduction plots are automatically saved in /outputs/plots/.

🧠 Future Enhancements

 Deep learning autoencoders for unsupervised anomaly detection

 Self-supervised ViT-based flow visualization module

 Multi-agent cyber threat hunting integration

 Docker container for deployment

 REST API endpoint for live detection

🧰 Tech Stack

Python 3.10+

Pandas, NumPy, Scikit-learn

LightGBM, XGBoost

Optuna / GridSearchCV

Matplotlib, Seaborn

JupyterLab

🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a PR or an issue.

📜 License

This project is licensed under the MIT License — see the LICENSE
 file for details.

🧩 Author

Bhavyy Khurana
AI & Cybersecurity Enthusiast | Building autonomous threat-hunting systems

🔗 GitHub
 • LinkedIn

🌟 Acknowledgements

NSL-KDD and CICIDS datasets

Scikit-learn documentation

LightGBM and Optuna teams

Research inspiration from DARPA Intrusion Detection Evaluation Dataset papers
