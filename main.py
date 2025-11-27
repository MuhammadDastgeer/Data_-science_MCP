# File: fastmcp_data_science_mcp.py
from fastmcp import FastMCP
import os
import tempfile
import uuid
import json
import io

# Data tools
import pandas as pd
import nbformat as nbf
import joblib
import matplotlib.pyplot as plt

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Optional DL
try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    keras = None

# Math/notes
import sympy as sp

# ------------------------------
# Storage / DB-like paths
# ------------------------------
TMP = tempfile.gettempdir()
STORAGE_DIR = os.path.join(TMP, "ds_mcp_storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

DATASETS = {}   # dataset_id -> {"path":..., "info":...}
MODELS = {}     # model_id -> {"type":"ml"/"dl", "path":..., "meta":...}

CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "ds_categories.json")

print(f"DS MCP storage folder: {STORAGE_DIR}")

# ------------------------------
# Create FastMCP instance
# ------------------------------
mcp = FastMCP("DataScienceMCP")

# ------------------------------
# Helper functions
# ------------------------------
def _save_bytes_to_path(b: bytes, filename: str):
    path = os.path.join(STORAGE_DIR, filename)
    with open(path, "wb") as f:
        f.write(b)
    return path

def _read_csv_from_path(path: str):
    return pd.read_csv(path)

def _safe_filename(prefix="file", ext=".csv"):
    return f"{prefix}_{uuid.uuid4().hex}{ext}"

# ------------------------------
# Initialization check (write access)
# ------------------------------
def _init_storage_test():
    try:
        test_path = os.path.join(STORAGE_DIR, "test_write.txt")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        print("Storage writable:", STORAGE_DIR)
    except Exception as e:
        print("Storage write test failed:", str(e))
        raise

_init_storage_test()

# ------------------------------
# Tools
# ------------------------------
@mcp.tool()
async def upload_dataset(file_bytes: bytes, filename: str):
    """
    Upload dataset bytes (CSV/JSON/Excel). Returns dataset_id and basic info.
    file_bytes: raw bytes of file
    filename: original filename (to get extension)
    """
    try:
        ext = os.path.splitext(filename)[1].lower() or ".csv"
        dataset_id = str(uuid.uuid4())
        safe_name = f"{dataset_id}{ext}"
        path = _save_bytes_to_path(file_bytes, safe_name)

        # try to load as csv
        try:
            df = _read_csv_from_path(path)
        except Exception:
            # attempt to load json
            try:
                df = pd.read_json(path)
            except Exception:
                raise ValueError("Unsupported dataset format or invalid file content")

        info = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "col_names": df.columns.tolist()
        }
        DATASETS[dataset_id] = {"path": path, "info": info}
        return {"status": "success", "dataset_id": dataset_id, "info": info}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def eda_report(dataset_id: str, max_plots: int = 3):
    """
    Return basic EDA (dtypes, missing, describe) and save a few histogram plots.
    """
    try:
        if dataset_id not in DATASETS:
            raise ValueError("dataset not found")
        path = DATASETS[dataset_id]["path"]
        df = _read_csv_from_path(path)

        report = {
            "shape": list(df.shape),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "describe": df.describe(include="all").to_dict()
        }

        # Save up to max_plots numeric histograms
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        plots = []
        for c in num_cols[:max_plots]:
            plt.figure()
            df[c].hist()
            plt.title(c)
            plot_name = f"eda_{dataset_id}_{c}_{uuid.uuid4().hex}.png"
            plot_path = os.path.join(STORAGE_DIR, plot_name)
            plt.savefig(plot_path)
            plt.close()
            plots.append(plot_path)
        if plots:
            report["plots"] = plots
        return {"status": "success", "report": report}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def train_ml(dataset_id: str, target_column: str, model_name: str = "logreg"):
    """
    Train a simple ML model. Supports: logreg, rf, svc
    Returns model_id and metrics.
    """
    try:
        if dataset_id not in DATASETS:
            raise ValueError("dataset not found")
        path = DATASETS[dataset_id]["path"]
        df = _read_csv_from_path(path)

        if target_column not in df.columns:
            raise ValueError("target column not found in dataset")

        X = df.drop(columns=[target_column]).select_dtypes(include=["number"])
        y = df[target_column]

        if X.shape[1] == 0:
            raise ValueError("No numeric features available for training. Convert/cast features to numeric.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_name == "logreg":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "rf":
            model = RandomForestClassifier(n_estimators=100)
        elif model_name == "svc":
            model = SVC(probability=True)
        else:
            model = LogisticRegression(max_iter=1000)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        report = classification_report(y_test, preds, output_dict=True)

        model_id = str(uuid.uuid4())
        model_path = os.path.join(STORAGE_DIR, f"ml_model_{model_id}.joblib")
        joblib.dump(model, model_path)

        MODELS[model_id] = {"type": "ml", "path": model_path, "meta": {"model_name": model_name, "dataset_id": dataset_id}}

        return {"status": "success", "model_id": model_id, "accuracy": acc, "report": report, "path": model_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def train_dl(dataset_id: str, target_column: str, epochs: int = 5, layers_cfg: list = None):
    """
    Train a simple Keras MLP. layers_cfg is a list like [64,32]
    """
    try:
        if keras is None:
            return {"status": "error", "message": "TensorFlow/Keras not installed in environment."}
        if dataset_id not in DATASETS:
            raise ValueError("dataset not found")
        path = DATASETS[dataset_id]["path"]
        df = _read_csv_from_path(path)

        if target_column not in df.columns:
            raise ValueError("target column not found in dataset")

        X = df.drop(columns=[target_column]).select_dtypes(include=["number"]).fillna(0).values
        y = df[target_column].astype("int").values

        if X.shape[1] == 0:
            raise ValueError("No numeric features for DL training.")

        num_features = X.shape[1]
        num_classes = len(set(y))

        model = keras.Sequential()
        model.add(layers.Input(shape=(num_features,)))
        layers_cfg = layers_cfg or [64, 32]
        for n in layers_cfg:
            model.add(layers.Dense(int(n), activation="relu"))
        if num_classes == 2:
            model.add(layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
        else:
            model.add(layers.Dense(num_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"

        model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        hist = model.fit(X, y, epochs=int(epochs), validation_split=0.1, batch_size=32, verbose=0)

        model_id = str(uuid.uuid4())
        model_path = os.path.join(STORAGE_DIR, f"dl_model_{model_id}")
        model.save(model_path)

        MODELS[model_id] = {"type": "dl", "path": model_path, "meta": {"layers": layers_cfg, "dataset_id": dataset_id}}

        return {"status": "success", "model_id": model_id, "history": hist.history, "path": model_path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def generate_notebook(title: str, code_cells: list):
    """
    Create a .ipynb file from a list of code cells (strings).
    """
    try:
        nb = nbf.v4.new_notebook()
        nb["cells"] = [nbf.v4.new_markdown_cell(f"# {title}")]
        for c in code_cells:
            nb["cells"].append(nbf.v4.new_code_cell(c))
        safe_title = title.replace(" ", "_")
        path = os.path.join(STORAGE_DIR, f"{safe_title}_{uuid.uuid4().hex}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        return {"status": "success", "notebook_path": path}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def create_notes(topic: str):
    """
    Generate simple notes for a topic (you can later improve with LLM).
    """
    try:
        text = f\"\"\"# Notes: {topic}

## Summary
Auto-generated summary for **{topic}**.

## Key points
- Point 1
- Point 2

## Example
Add example code or math here.
\"\"\"
        return {"status": "success", "notes": text}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def explain_math(expression: str):
    """Symbolic simplify / derivative / integral using sympy."""
    try:
        expr = sp.sympify(expression)
        return {
            "status": "success",
            "original": str(expr),
            "simplified": str(sp.simplify(expr)),
            "derivative": str(sp.diff(expr)),
            "integral": str(sp.integrate(expr))
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ------------------------------
# Listing & model download helper
# ------------------------------
@mcp.tool()
def list_datasets():
    rows = []
    for k,v in DATASETS.items():
        rows.append({"dataset_id": k, **v["info"]})
    return {"status": "success", "datasets": rows}

@mcp.tool()
def list_models():
    rows = []
    for k,v in MODELS.items():
        rows.append({"model_id": k, "type": v.get("type"), "meta": v.get("meta"), "path": v.get("path")})
    return {"status": "success", "models": rows}

# resource: categories
@mcp.resource("ds:///categories", mime_type="application/json")
def categories():
    try:
        default = {"categories": ["Datasets", "Models", "Notebooks", "Notes", "Other"]}
        try:
            with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return json.dumps(default, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

# ------------------------------
# Run server
# ------------------------------
if __name__ == "__main__":
    # Run with default transport (http) on port 8000
    mcp.run(transport="http", host="0.0.0.0", port=8000)
