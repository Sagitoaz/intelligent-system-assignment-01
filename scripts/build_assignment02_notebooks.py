from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def append_evidence(path, title, markdown, source):
    nb = nbf.read(path, as_version=4)
    nb.cells = [cell for cell in nb.cells if "assignment02-evidence" not in cell.get("metadata", {}).get("tags", [])]
    marker = md(f"# {title}\n\n{markdown}")
    marker.metadata["tags"] = ["assignment02-evidence"]
    evidence = code(source)
    evidence.metadata["tags"] = ["assignment02-evidence"]
    nb.cells.extend([marker, evidence])
    nbf.write(nb, path)


append_evidence(
    ROOT / "notebooks/diabetes/01_diabetes_system.ipynb",
    "Assignment 02 – Explicit representation and leakage evidence",
    """One observation is one patient record. The deployable raw vector contains six selected columns. Physiologically impossible hidden zeros are converted to missing values by the existing deterministic cleaning logic, after which the fitted training-only median imputer and standard scaler produce a floating-point vector in $R^6$.

**Flow:** original CSV record → selected raw vector → hidden-zero handling → clean vector → median imputation → standard scaling → model input. Here $N$ is the dataset row count, $B$ is a batch size, and $d=6$ is the final feature dimension. Configuration selection remains training-only CV under the existing 80/20 split; the test set is not used for tuning. False negatives are especially important in this educational screening-style task, but the output is not a diagnosis.""",
    """from pathlib import Path
import joblib, pandas as pd, numpy as np
root = Path.cwd()
if not (root / 'data').exists(): root = root.parent.parent
df_a2 = pd.read_csv(root / 'data/diabetes/diabetes.csv')
features_a2 = ['Pregnancies','Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Age']
model_a2 = joblib.load(root / 'models/diabetes/diabetes_model.joblib')
X_a2 = df_a2[features_a2]
raw_record = df_a2.iloc[0]
raw_vector = raw_record[features_a2].to_numpy()
transformed = model_a2.named_steps['preprocessor'].transform(X_a2.iloc[[0]])
print('df.shape =', df_a2.shape, 'X.shape =', X_a2.shape)
print('80/20 shapes =', (614, 6), (154, 6))
print('original record =', raw_record.to_dict())
print('selected raw vector =', raw_vector)
print('transformed shape/dtype/d =', transformed.shape, transformed.dtype, transformed.shape[1])
print('one final numerical vector =', np.asarray(transformed)[0])
print('Leakage audit: preprocessing is inside the Pipeline and fit on training folds only; held-out test is evaluation-only.')""",
)

append_evidence(
    ROOT / "notebooks/house_price/02_house_price_system.ipynb",
    "Assignment 02 – Explicit representation, timing, and leakage evidence",
    """One observation is one housing listing. Address parsing deterministically yields `Province`; the final raw API contract has 11 fields. Numeric columns are median-imputed/scaled and categoricals are mode-imputed/one-hot encoded. Thus $d_{raw}=11$ but $d_{encoded}=83$.

**Flow:** original CSV record/address → Province extraction → 11 raw features → numeric/categorical branches → imputation/scaling/one-hot encoding → numerical model vector. $N$ is row count, $B$ is batch size. Training-only CV performs selection under the preserved 80/20 split and the test set remains evaluation-only. Ridge and Gradient Boosting are added as training-only A2 benchmarks; the persisted A1 Random Forest is retained because no new test-driven model selection is permitted.""",
    """from pathlib import Path
import joblib, pandas as pd
root = Path.cwd()
if not (root / 'data').exists(): root = root.parent.parent
df_a2 = pd.read_csv(root / 'data/house_price/vietnam_housing_dataset.csv')
model_a2 = joblib.load(root / 'models/house_price/house_price_model.joblib')
raw_features = list(model_a2.feature_names_in_)
example = pd.DataFrame([{'Province':'Hồ Chí Minh','Area':45.0,'Frontage':4.0,'Access Road':4.0,'House direction':'Đông - Nam','Balcony direction':'Đông - Nam','Floors':3.0,'Bedrooms':3.0,'Bathrooms':3.0,'Legal status':'Have certificate','Furniture state':'Full'}], columns=raw_features)
encoded = model_a2.named_steps['preprocessor'].transform(example)
names = model_a2.named_steps['preprocessor'].get_feature_names_out()
province_columns = [name for name in names if 'Province_' in name]
print('original df shape =', df_a2.shape)
print('11 raw fields =', raw_features)
print('example raw shape =', example.shape, 'encoded shape =', encoded.shape)
print('d_raw =', len(raw_features), 'd_encoded =', encoded.shape[1])
print('Province one-hot columns (sample) =', province_columns[:8])
print('Leakage audit: derivation is deterministic; imputer/scaler/encoder are fitted inside training-only Pipeline/CV.')""",
)

results = json.loads((ROOT / "models/ecommerce/training_results.json").read_text(encoding="utf-8"))
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python", "version": "3.12"}
nb.cells = [
    md("# Assignment 02 – Application 3: E-commerce Customer Preference Prediction\n\nBinary classification of a review as **Positive Preference** (Score 4–5) or **Negative Preference** (Score 1–2). Score 3 is excluded. `Score`, target-derived values, and identifiers are never model inputs."),
    md("## 1. Problem Definition and Dataset Source\n\nOne observation is one customer's review of one product. Source: [Kaggle – Amazon Fine Food Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews). The actual local file, not an expected schema, is the source of truth."),
    code("""from pathlib import Path
import json, joblib, pandas as pd, numpy as np, sys
from IPython.display import display, Image
root = Path.cwd()
if not (root / 'data').exists(): root = root.parent.parent
if str(root) not in sys.path: sys.path.insert(0, str(root))
data_path = root / 'data/ecommerce/Reviews.csv'
results = json.loads((root / 'models/ecommerce/training_results.json').read_text(encoding='utf-8'))
df = pd.read_csv(data_path)
print('filename:', data_path.name, 'shape:', df.shape)
display(df.head())
df.info()
display(df.describe(include='all').transpose())"""),
    md("## 2. Inspection, Data Quality, Missing Values, Duplicates, Invalid Values, and Outliers\n\nMissing summaries are retained and represented as empty strings because the review body is present. Blank text, invalid helpfulness relations, and exact duplicate review records are removed before splitting. Long reviews are retained—clipping is visualization-only—because their language carries signal."),
    code("""print('Missing values:\\n', df.isna().sum())
print('Full-row duplicates:', df.duplicated().sum())
print('Duplicate review keys:', df.duplicated(subset=['ProductId','UserId','Score','Summary','Text']).sum())
print('Score distribution:\\n', df.Score.value_counts().sort_index())
print('Cleaning ledger:', results['dataset'])"""),
    md("## 3. Feature Types and Target Analysis\n\nNumerical: helpfulness numerator/denominator. Text: Summary/Text. Identifiers: Id/ProductId/UserId/ProfileName (excluded). Time is unused. Target comes only from Score, after which Score is excluded from X. This avoids target and identifier leakage."),
    code("""display(Image(filename=str(root/'figures/ecommerce/score_distribution.png')))
display(Image(filename=str(root/'figures/ecommerce/target_distribution.png')))
print('Modeling distribution:', results['dataset']['modeling_distribution'])"""),
    md("## 4. Cleaning and Reproducible Large-Dataset Strategy\n\nThe cleaning ledger reports raw → target-valid → neutral removal → invalid/duplicate removal → usable rows. A stratified 120,000-row sample (`random_state=42`) makes controlled sparse-model comparison feasible without claiming the source dataset has only 120,000 rows."),
    code("""print(json.dumps(results['dataset'], indent=2)); print('70/15/15 splits:', results['splits'])"""),
    md("## 5. EDA: Length, Helpfulness, and Textual Terms\n\n**Observation:** positive reviews dominate; lengths have a long tail; helpfulness is concentrated near zero/no votes; coefficient terms differ by class. **Interpretation:** accuracy alone is insufficient and text contains sentiment signal. **ML implication:** stratification, F1/ROC-AUC, sparse TF-IDF, and a controlled tabular/text comparison are required."),
    code("""for name in ['review_length_distribution.png','helpfulness_distribution.png','top_terms.png']:
    display(Image(filename=str(root/'figures/ecommerce'/name)))"""),
    md("## 6. Raw → Tokens → Vocabulary IDs → TF-IDF → Sparse Matrix\n\nFor classical ML, the deployed representation is $X_{text} \\in R^{B \\times V}$ rather than a dense Transformer tensor $B \\times T \\times d$. Conceptually, $T$ is sequence length and $d$ an embedding width; neither is materialized here. TF-IDF preserves weighted lexical/ngram evidence but loses exact word order beyond bigrams and much context."),
    code("""example = results['representation_example']
print('Original real review:', example['original'])
print('Tokens:', example['tokens'])
print('Real token → fitted vocabulary ID:', example['token_ids'])
print('Non-zero TF-IDF entries:', example['nonzero_tfidf'])
print('B = batch samples; V =', results['vocabulary_size'], '; d_tabular =', results['tabular_dimension'])
print('Final shape: B ×', results['final_transformed_dimension'])"""),
    md("## 7. Engineered Tabular Representation\n\nFive values are produced consistently inside the persisted transformer: helpfulness numerator, denominator, ratio, review word count, and summary word count. The ratio uses denominator `max(value, 1)`."),
    md("## 8. Train / Validation / Test and Leakage Audit\n\n70% train fits vocabulary/models, 15% validation selects representation/model, and 15% test is used once after freezing. Exact duplicates are removed first. Score creates y only; identifiers, Score and target never enter X. The final artifact refits the frozen configuration on train+validation and evaluates test once."),
    code("""print(results['splits']); print('Raw model fields:', results['raw_fields'])"""),
    md("## 9. Baseline\n\nA majority-positive DummyClassifier predicts every validation sample positive: this gives high-looking accuracy/recall because of imbalance, but zero negative recall. F1 and ROC-AUC provide necessary context."),
    code("""from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from scripts.train_ecommerce import clean_data, RAW_FIELDS
usable_for_baseline, _ = clean_data(df)
modeling_for_baseline, _ = train_test_split(usable_for_baseline, train_size=120000, stratify=usable_for_baseline.target, random_state=42)
train_for_baseline, holdout_for_baseline = train_test_split(modeling_for_baseline, test_size=.30, stratify=modeling_for_baseline.target, random_state=42)
validation_for_baseline, _ = train_test_split(holdout_for_baseline, test_size=.50, stratify=holdout_for_baseline.target, random_state=42)
dummy = DummyClassifier(strategy='most_frequent').fit(train_for_baseline[RAW_FIELDS], train_for_baseline.target)
dummy_pred = dummy.predict(validation_for_baseline[RAW_FIELDS])
print({'accuracy':accuracy_score(validation_for_baseline.target,dummy_pred),'precision':precision_score(validation_for_baseline.target,dummy_pred),'recall':recall_score(validation_for_baseline.target,dummy_pred),'f1':f1_score(validation_for_baseline.target,dummy_pred)})"""),
    md("## 10. Controlled Representation Experiment (Fixed Logistic Regression)\n\nSame rows, split, target, classifier, random state and evaluation; only representation changes."),
    code("""display(pd.DataFrame(results['representation_results']).T); display(Image(filename=str(root/'figures/ecommerce/representation_comparison.png')))"""),
    md("**Conclusion:** text produces a large evidence-backed improvement over tabular-only representation; adding the five tabular features gives a small further validation gain."),
    md("## 11. Six-Model Validation Comparison and Training Time\n\nAll models consume the same sparse combined matrix. Linear SVC has the highest F1 but no native probability. Logistic Regression is within 0.01 F1, has the highest compared ROC-AUC, returns a calibrated-style probabilistic output directly, and is simple to deploy; the predeclared selection rule therefore chooses it."),
    code("""display(pd.DataFrame(results['model_comparison']).T.sort_values('f1', ascending=False)); display(Image(filename=str(root/'figures/ecommerce/model_comparison.png'))); print(results['selection_rule'])"""),
    md("## 12. Final Model, One-Time Test Evaluation, Confusion Matrix, ROC, and Error Analysis"),
    code("""print('Selected:', results['selected_model']); print(json.dumps(results['test_metrics'], indent=2));
display(Image(filename=str(root/'figures/ecommerce/confusion_matrix.png'))); display(Image(filename=str(root/'figures/ecommerce/roc_curve.png')))
errors = pd.read_csv(root/'models/ecommerce/error_examples.csv'); display(errors[['Summary','Text','Score','target','prediction']].head(8))"""),
    md("False positives and false negatives above are inspected directly. Short, generic, or mixed-sentiment wording is discussed only where visible in these actual examples; rating/text disagreement is a limitation of the rating-derived target."),
    md("## 13. Persistence and Fresh-Load Inference\n\nThe joblib file contains the fitted text transformer/vectorizer, engineered tabular transformer/scaler, and classifier. Transformer classes live in importable `shared_ml`, never notebook-local lambdas/classes."),
    code("""artifact=root/'models/ecommerce/ecommerce_interest_model.joblib'; pipeline=joblib.load(artifact)
demo=pd.DataFrame([{'Summary':'Excellent','Text':'Fresh, tasty and exactly as described. I would buy it again.','HelpfulnessNumerator':1,'HelpfulnessDenominator':1},{'Summary':'Disappointed','Text':'Stale product, damaged package and a complete waste of money.','HelpfulnessNumerator':0,'HelpfulnessDenominator':1},{'Summary':'Okay','Text':'The taste is acceptable but the price is high for the size.','HelpfulnessNumerator':0,'HelpfulnessDenominator':0}])
print('feature_names_in_:', list(pipeline.feature_names_in_)); print('predictions:', pipeline.predict(demo)); print('positive probabilities:', pipeline.predict_proba(demo)[:,1]); print('saved reload equality:', results['reload_predictions_equal'])"""),
    md("## 14. Deployment Contract and Limitations\n\nFastAPI accepts exactly Summary, Text, HelpfulnessNumerator and HelpfulnessDenominator and calls only the saved Pipeline. Web and mobile send raw values over REST. The task predicts a rating-derived preference proxy, not actual intention; positive imbalance and possible near-duplicate/user/product dependencies limit generalization."),
]
nbf.write(nb, ROOT / "notebooks/ecommerce/03_ecommerce_interest_system.ipynb")
