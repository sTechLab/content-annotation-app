# Content Annotation App

Streamlit app for coding Bluesky posts. Sample mode uses the same fixed 98 posts for every coder; Full mode uses all posts in a stable coder-specific random order and includes preset tags and labels.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/annotation_app.py
```

The deployed datasets are configured in `config/config.json` and stored at:

```text
data/processed/YYYY-MM-DD/enriched_results_sample.csv
data/processed/YYYY-MM-DD/enriched_results.csv
```

## Deploy

Deploy `app/annotation_app.py` from this repository on Streamlit Community Cloud. No secrets or external storage are required.

Each coder must use the prominent download button before closing the app and send the resulting CSV to the project owner. Place returned files in `annotations/raw/`, then merge them with:

```bash
python scripts/merge_annotations.py
```

The app's server-side annotation files are temporary on Streamlit Community Cloud; the downloaded CSV is the durable copy.
