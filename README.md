# YouTube Comment Sentiment Analyzer

A DistilBERT-based NLP project that classifies YouTube comments as **negative**, **neutral**, or **positive** and provides an interactive Streamlit interface for reviewing and correcting predictions.

## Highlights

- Fetches real comments through the YouTube Data API
- Performs batch inference with a fine-tuned DistilBERT classifier
- Displays confidence scores and class probabilities
- Supports search, filtering, and sorting by engagement or confidence
- Provides a three-column review board for manual relabeling
- Exports minimal and full CSV datasets for future model refinement
- Includes Version 1 and Version 2 training pipelines

## Model performance

The Version 2 model was evaluated on a balanced unseen test set:

| Metric | Result |
|---|---:|
| Accuracy | 0.8080 |
| Macro F1 | 0.8088 |
| Weighted F1 | 0.8088 |
| Average confidence | 0.9519 |

Per-class F1 scores were approximately **0.83 negative**, **0.77 neutral**, and **0.82 positive**.

## Project structure

```text
youtube-sentiment-analyzer/
├── src/
│   └── app.py
├── training/
│   ├── training_model_v1.py
│   └── training_model_v2.py
├── docs/
│   └── project_report.pdf
├── data/
├── results/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the YouTube API key

Copy `.env.example` to `.env`, then add your key:

```env
YOUTUBE_API_KEY=your_real_key
MODEL_PATH=./results/checkpoint-2031
```

Load these variables in your terminal before running the app, or configure them through your IDE.

PowerShell example:

```powershell
$env:YOUTUBE_API_KEY="your_real_key"
$env:MODEL_PATH="./results/checkpoint-2031"
```

### 4. Add the trained checkpoint

Place the fine-tuned Hugging Face checkpoint in:

```text
results/checkpoint-2031/
```

Model weights are intentionally excluded from Git because they are large. The path can be changed with the `MODEL_PATH` environment variable.

### 5. Run the application

```bash
streamlit run src/app.py
```

## Training

The training scripts expect CSV files with the columns:

```text
comment,sentiment
```

Place the datasets in `data/` using the filenames referenced by each script, then run:

```bash
python training/training_model_v2.py
```

Version 2 includes dynamic padding, reproducible seeds, macro-F1 model selection, early stopping, and validation at each epoch.

## Documentation

The complete academic report is available at [`docs/project_report.pdf`](docs/project_report.pdf). It documents dataset construction, class balancing, training, evaluation, and the interactive review application.

## Privacy and reproducibility

- API keys and local environment files are excluded from Git.
- Raw datasets and model checkpoints are excluded by default.
- Only publish datasets when you have the right to redistribute them.
