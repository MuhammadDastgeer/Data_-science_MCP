from mcp import MCPServer
import os, uuid, json
import pandas as pd
import matplotlib.pyplot as plt
import nbformat as nbf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Optional Deep Learning
try:
    from tensorflow import keras
    from tensorflow.keras import layers
except:
    keras = None

import sympy as sp

# -----------------------------
# Storage
# -----------------------------
STORAGE = os.path.join(os.getcwd(), "storage")
os.makedirs(STORAGE, exist_ok=True)

DATASETS = {}
MODELS = {}

# -----------------------------
# MCP Server
# -----------------------------
server = MCPServer("dastgeer-data-science-mcp")

# -----------------------------
# Dataset Upload
# -----------------------------
@server.tool()
async def upload_dataset(file_bytes: bytes, filename: str):
    """Upload dataset file and store information."""
    dataset_id = str(uuid.uuid4())
    ext = os.path.splitext(filename)[1] or ".csv"
    path = os.path.join(STORAGE, f"{dataset_id}{ext}")

    with open(path, "wb") as f:
        f.write(file_bytes)

    df = pd.read_csv(path)
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "col_names": df.columns.tolist()
    }

    DATASETS[dataset_id] = {"path": path, "info": info}
    return {"dataset_id": dataset_id, "info": info}


# -----------------------------
# EDA Report
# -----------------------------
@server.tool()
def eda_report(dataset_id: str):
    if dataset_id not in DATASETS:
        raise ValueError("Dataset not found")

    path = DATASETS[dataset_id]["path"]
    df = pd.read_csv(path)

    report = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "describe": df.describe(include="all").to_dict()
    }

    # Save sample plots
    try:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        saved_plots = []

        for c in num_cols[:3]:
            plt.figure()
            df[c].hist()
            plot_path = os.path.join(STORAGE, f"eda_{uuid.uuid4().hex}_{c}.png")
            plt.title(c)
            plt.savefig(plot_path)
            plt.close()
            saved_plots.append(plot_path)

        report["plots"] = saved_plots

    except Exception:
        pass

    return report


# -----------------------------
# ML Training
# -----------------------------
@server.tool()
def train_ml(dataset_id: str, target_column: str, model_name: str = "logreg"):
    if dataset_id not in DATASETS:
        raise ValueError("Dataset not found")

    path = DATASETS[dataset_id]["path"]
    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError("Invalid target column")

    X = df.drop(columns=[target_column]).select_dtypes(include=["number"])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_name == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "svc":
        model = SVC(probability=True)
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=100)
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    model_id = str(uuid.uuid4())
    model_path = os.path.join(STORAGE, f"ml_model_{model_id}.joblib")
    joblib.dump(model, model_path)

    MODELS[model_id] = {"type": "ml", "path": model_path}

    return {"model_id": model_id, "accuracy": acc, "report": report, "path": model_path}


# -----------------------------
# DL Training
# -----------------------------
@server.tool()
def train_dl(dataset_id: str, target_column: str, epochs: int = 5, model_config: dict = None):

    if keras is None:
        return {"error": "TensorFlow not installed"}

    if dataset_id not in DATASETS:
        raise ValueError("Dataset not found")

    path = DATASETS[dataset_id]["path"]
    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise ValueError("Invalid target column")

    X = df.drop(columns=[target_column]).select_dtypes(include=["number"]).fillna(0).values
    y = df[target_column].astype("int").values

    num_features = X.shape[1]
    num_classes = len(set(y))

    model = keras.Sequential()
    model.add(layers.Input(shape=(num_features,)))

    layers_cfg = model_config.get("layers", [64, 32]) if model_config else [64, 32]
    for n in layers_cfg:
        model.add(layers.Dense(n, activation="relu"))

    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    history = model.fit(
        X, y, epochs=epochs, validation_split=0.1, batch_size=32, verbose=0
    )

    model_id = str(uuid.uuid4())
    model_path = os.path.join(STORAGE, f"dl_model_{model_id}")
    model.save(model_path)

    MODELS[model_id] = {"type": "dl", "path": model_path}

    return {"model_id": model_id, "history": history.history, "path": model_path}


# -----------------------------
# Notebook Generator
# -----------------------------
@server.tool()
def generate_notebook(title: str, code_cells: list):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [nbf.v4.new_markdown_cell(f"# {title}")]

    for c in code_cells:
        nb["cells"].append(nbf.v4.new_code_cell(c))

    safe_title = title.replace(" ", "_")
    path = os.path.join(STORAGE, f"{safe_title}.ipynb")

    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    return {"notebook_path": path}


# -----------------------------
# Notes Generator
# -----------------------------
@server.tool()
def create_notes(topic: str):
    text = f"""
# Notes: {topic}

## Summary
Auto-generated notes for **{topic}**.

## Key Points
- Concept 1
- Concept 2
- Concept 3

## Example
Add your examples here.
"""
    return {"notes": text}


# -----------------------------
# Math Explanation Tool
# -----------------------------
@server.tool()
def explain_math(expression: str):
    try:
        expr = sp.sympify(expression)
        return {
            "original": str(expr),
            "simplified": str(sp.simplify(expr)),
            "derivative": str(sp.diff(expr)),
            "integral": str(sp.integrate(expr)),
        }
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Helpers
# -----------------------------
@server.tool()
def list_datasets():
    return list(DATASETS.keys())


@server.tool()
def list_models():
    return list(MODELS.keys())


# -----------------------------
# Run MCP Server
# -----------------------------
def run():
    server.run()


if __name__ == "__main__":
    run()
