from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data/ecommerce/Reviews.csv")
print("ECOMMERCE_SHAPE", df.shape)
print("ECOMMERCE_DTYPES")
print(df.dtypes.to_string())
print("ECOMMERCE_HEAD")
print(df.head(3).to_string(index=False))
print("ECOMMERCE_MISSING")
print(df.isna().sum().to_string())
print("FULL_DUPLICATES", int(df.duplicated().sum()))
print("REVIEW_DUPLICATES", int(df.duplicated(subset=["ProductId", "UserId", "Score", "Summary", "Text"]).sum()))
print("SCORES")
print(df["Score"].value_counts(dropna=False).sort_index().to_string())
print("INVALID_HELPFULNESS", int(((df.HelpfulnessNumerator < 0) | (df.HelpfulnessDenominator < 0) | (df.HelpfulnessNumerator > df.HelpfulnessDenominator)).sum()))

for name in ["diabetes", "house_price"]:
    path = ROOT / "models" / name / ("diabetes_model.joblib" if name == "diabetes" else "house_price_model.joblib")
    model = joblib.load(path)
    print("MODEL", name, type(model), list(model.named_steps), list(getattr(model, "feature_names_in_", [])))
    pre = model.named_steps.get("preprocessor")
    if pre is not None:
        try:
            print("TRANSFORMED_DIM", name, pre.transform(pd.DataFrame([{c: 1 for c in model.feature_names_in_}])).shape)
        except Exception as exc:
            print("TRANSFORM_PROBE_ERROR", name, repr(exc))

for path in [ROOT / "notebooks/diabetes/01_diabetes_system.ipynb", ROOT / "notebooks/house_price/02_house_price_system.ipynb"]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    heads = []
    executed = 0
    code = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            text = "".join(cell["source"])
            heads += [line for line in text.splitlines() if line.startswith("#")]
        else:
            code += 1
            executed += cell.get("execution_count") is not None
    print("NOTEBOOK", path.relative_to(ROOT), "cells", len(nb["cells"]), "code", code, "executed", executed)
    print("HEADINGS")
    print("\n".join(heads))
