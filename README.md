# Emergency Information Extractor

This project aims to extract critical emergency-related information from spoken or written input. The pipeline will:

- Convert speech to text
- Preprocess and normalize the text
- Detect language
- Run named entity recognition (NER) to find entities like locations, people, and key resources
- Classify the type of emergency
- Estimate severity

## Project Structure

- `data/raw` – Original, unprocessed data (audio, transcripts, etc.)
- `data/processed` – Cleaned and preprocessed data ready for modeling
- `models` – Trained model artifacts and checkpoints
- `notebooks` – Jupyter notebooks for experiments and EDA
- `src` – Core source code for the pipeline
- `app` – Application interface (e.g., UI or API)

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your data in `data/raw` and start experimenting in `notebooks/`.

