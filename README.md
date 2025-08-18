#  LLM-POWERED CONTRACT RISK DETECTION AND CLAUSE-BASED INQUIRY SYSTEM

This repository contains end-to-end experiments with **Legal-BERT** and **Longformer** on three widely used legal NLP datasets:

- **LEDGAR** → Multi-label clause classification  
- **ContractNLI** → Natural language inference on contracts  
- **CUAD** → Extractive question answering for contracts  

The project includes training scripts, evaluation pipelines, and a Streamlit dashboard for visualizing results.

---

## Repository Structure

```
├── code/
│ ├── eda/ # Exploratory data analysis scripts
│ ├── main/ # evaluation + dashboard
│ │ ├── main.py # Runs evaluation for LEDGAR, ContractNLI, CUAD
│ │ └── dashboard.py # Streamlit dashboard for visualizing results
│ ├── model/ # Training scripts for each dataset
│ └── pre-processing/ # Cleaning & tokenization scripts
│
├── datasets/ # Tokenized inputs and processed data
│
├── results/
│ ├── evaluation_results.csv # Final metrics from the evaluation pipeline
│ └── figures/ # Exported charts
│
├── MRP_Report.pdf # Final project report
├── README.md # This file
└── Requirements.txt # Python dependencies
```

---

##  Quickstart

### 1. Clone repository
```bash
git clone https://github.com/NikStar2/MRP
```

### 2. Install dependencies
```bash
pip install -r Requirements.txt
```

### 3. Train models
Each dataset has its own script. Example:

```bash
python train_ledgar.py
python train_contractnli.py
python train_cuad.py
```

Trained model checkpoints are saved as:
- `best_ledgar.pt`
- `best_contractnli.pt`
- `best_cuad.pt`

### 4. Evaluate models
Run evaluation across all datasets:
```bash
python main.py
```
This generates **evaluation_results.csv**.

### 5. Launch dashboard
```bash
streamlit run dashboard.py
```
You’ll get an interactive visualization of metrics.


## Datasets

- **LEDGAR**: Contract clause classification dataset  
- **ContractNLI**: Contract-based natural language inference  
- **CUAD**: Contract Understanding Atticus Dataset  

Make sure you preprocess and tokenize them into the expected `.jsonl` or `.pt` formats before training.

---

## Notes
- Scripts use **gradient checkpointing**, **mixed precision training**, and **AdamW with warmup**.
- ContractNLI supports **oversampling of Contradiction class** and **layer freezing**.
- CUAD pipeline manually handles token type ids and span filtering.

---
