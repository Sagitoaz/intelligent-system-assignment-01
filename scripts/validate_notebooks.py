import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
paths = [
    root / "notebooks/diabetes/01_diabetes_system.ipynb",
    root / "notebooks/house_price/02_house_price_system.ipynb",
    root / "notebooks/ecommerce/03_ecommerce_interest_system.ipynb",
]
for path in paths:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    unexecuted = [index for index, cell in enumerate(code) if cell.get("execution_count") is None]
    errors = [output for cell in code for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    assert not unexecuted, f"{path}: unexecuted code cells {unexecuted}"
    assert not errors, f"{path}: {len(errors)} error outputs"
    print(path.relative_to(root), f"PASS ({len(code)} executed code cells)")
