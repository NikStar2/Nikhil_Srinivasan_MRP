# ⚖️ Legal NLP Benchmark — LEDGAR, ContractNLI, CUAD

This repository contains end-to-end experiments with **Legal-BERT** and **Longformer** on three widely used legal NLP datasets:

- **LEDGAR** → Multi-label clause classification  
- **ContractNLI** → Natural language inference on contracts  
- **CUAD** → Extractive question answering for contracts  

The project includes training scripts, evaluation pipelines, and a Streamlit dashboard for visualizing results.

---

## 📂 Repository Structure

```
├── train_ledgar.py              # Fine-tune Legal-BERT on LEDGAR
├── train_contractnli_single.py  # Fine-tune Longformer on ContractNLI
├── train_cuad.py                # Fine-tune Legal-BERT on CUAD
├── main.py                      # Combined evaluation across datasets
├── dashboard_mrp_two_page_fixed.py  # Streamlit dashboard
├── evaluation_results.csv       # Saved metrics after running main.py
├── Requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Quickstart

### 1. Clone repository
```bash
git clone https://github.com/yourusername/legal-nlp-benchmark.git
cd legal-nlp-benchmark
```

### 2. Install dependencies
```bash
pip install -r Requirements.txt
```

### 3. Train models
Each dataset has its own script. Example:

```bash
python train_ledgar.py
python train_contractnli_single.py
python train_cuad.py
```

Trained models are saved as:
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
streamlit run dashboard_mrp_two_page_fixed.py
```
You’ll get an interactive visualization of metrics.

---

## 📊 Results

Below are the key metrics after training and evaluation.  
(Values may differ depending on random seed and environment.)

### Unified F1 across models
![Unified F1](01_unified_f1_lollipop.png)

### Micro vs Macro F1 (classification)
![Slope Micro vs Macro](02_slope_micro_macro_f1.png)

### Micro Precision / Recall / F1
![Micro metrics](03_grouped_micro_metrics.png)

### Macro metrics heatmap
![Macro heatmap](04_heatmap_macro.png)

### Accuracy (classification tasks)
![Accuracy](06_accuracy_bars.png)

### CUAD EM vs F1
![CUAD](08_cuad_em_f1.png)

### Evaluation results table
![Results table](10_results_table.png)

---

## 📑 Datasets

- **LEDGAR**: Contract clause classification dataset  
- **ContractNLI**: Contract-based natural language inference  
- **CUAD**: Contract Understanding Atticus Dataset  

Make sure you preprocess and tokenize them into the expected `.jsonl` or `.pt` formats before training.

---

## 📌 Notes
- Scripts use **gradient checkpointing**, **mixed precision training**, and **AdamW with warmup**.
- ContractNLI supports **oversampling of Contradiction class** and **layer freezing**.
- CUAD pipeline manually handles token type ids and span filtering.

---

## 🖼️ Dashboard Preview

The Streamlit dashboard provides:
- Unified F1 comparisons  
- Micro vs Macro breakdowns  
- Heatmaps and slope graphs  
- QA-specific metrics (EM & F1)  

Run it with:
```bash
streamlit run dashboard_mrp_two_page_fixed.py
```

---

## 📄 License
MIT License. Free to use with attribution.
