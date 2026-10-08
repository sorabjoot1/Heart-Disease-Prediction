# Heart Disease Prediction

## Files
- `heart_disease_classification.ipynb` – Tasks 1–7 (explore, preprocess, EDA, CV, 5 models, evaluation, comparison). Saves the best model.
- `heart.csv` – dataset
- `model/best_model.pkl` – best trained model (full pipeline: preprocessing + SVM)
- `model/model_info.json` – model name, feature order, test metrics
- `app/app.py` + `app/templates/index.html` – Flask web frontend/backend
- `requirements.txt`

## Run
```
pip install -r requirements.txt
python app/app.py
```
Open http://127.0.0.1:5000 in your browser.

To retrain: `jupyter notebook heart_disease_classification.ipynb` and Run All (re-creates `model/`).
Use the same scikit-learn version as in requirements.txt, since pickle files depend on it.
