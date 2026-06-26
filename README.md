# Content Annotation App

Streamlit app for coding a fixed sample of Bluesky posts. Each coder works through the same posts and downloads a personal annotations CSV when finished.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/annotation_app.py
```

The deployed sample is configured in `config/config.json` and stored at:

```text
data/processed/YYYY-MM-DD/enriched_results_sample.csv
```

## Deploy

Deploy `app/annotation_app.py` from this repository on Streamlit Community Cloud. No secrets or external storage are required.

Each coder must use the prominent download button before closing the app and send the resulting CSV to the project owner. Place returned files in `annotations/raw/`, then merge them with:

```bash
python scripts/merge_annotations.py
```

The app's server-side annotation files are temporary on Streamlit Community Cloud; the downloaded CSV is the durable copy.
