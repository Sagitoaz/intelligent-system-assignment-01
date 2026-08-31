from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/house_price/02_house_price_system.ipynb"
nb = nbf.read(path, as_version=4)
nb.cells = [cell for cell in nb.cells if "assignment02-house-models" not in cell.get("metadata", {}).get("tags", [])]
intro = nbf.v4.new_markdown_cell("""## Assignment 02 model-family alignment: Ridge and Gradient Boosting

The original five valid A1 families are preserved. Ridge Regression and Gradient Boosting are added as training-only three-fold CV benchmarks on the same final 11-field representation. Timing uses `time.perf_counter()`. These additions do not use the held-out test set for selection and do not replace the already frozen Random Forest artifact.""")
intro.metadata["tags"] = ["assignment02-house-models"]
benchmark = nbf.v4.new_code_cell("""from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate
import time

a2_cv = KFold(n_splits=3, shuffle=True, random_state=42)
a2_candidates = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
}
a2_rows = []
for name, estimator in a2_candidates.items():
    pipeline = create_model_pipeline(estimator, final_numeric_features, final_categorical_features)
    started = time.perf_counter()
    scores = cross_validate(
        pipeline, X_train_final, y_train, cv=a2_cv,
        scoring={'MAE':'neg_mean_absolute_error','RMSE':'neg_root_mean_squared_error','R2':'r2'},
        n_jobs=-1,
    )
    elapsed = time.perf_counter() - started
    a2_rows.append({
        'Model': name,
        'CV MAE': -scores['test_MAE'].mean(),
        'CV RMSE': -scores['test_RMSE'].mean(),
        'CV R2': scores['test_R2'].mean(),
        'Training Time (s)': elapsed,
    })
a2_house_model_results = pd.DataFrame(a2_rows)
print(a2_house_model_results.round(4).to_string(index=False))
print('Selection note: training-only evidence; persisted Random Forest remains unchanged.')""")
benchmark.metadata["tags"] = ["assignment02-house-models"]
nb.cells.extend([intro, benchmark])
nbf.write(nb, path)
