"""BRFSS feature order for the teaching-oriented one-dimensional sequence."""

TARGET = "Diabetes_binary"
FEATURE_ORDER = (
    "BMI", "HighBP", "HighChol", "CholCheck", "HeartDiseaseorAttack", "Stroke",
    "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth",
    "DiffWalk", "Sex", "Age", "Education", "Income",
)
SCALED_FEATURES = ("BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income")


def validate_schema(columns):
    missing = sorted(set(FEATURE_ORDER + (TARGET,)) - set(columns))
    if missing:
        raise ValueError(f"Missing BRFSS columns: {missing}")
    return True
