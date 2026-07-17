# LEAP Atmospheric Physics Using AI — ClimSim

This project contains a Google Colab notebook for experimenting with machine-learning models on the **LEAP — Atmospheric Physics Using AI (ClimSim)** dataset.

The notebook performs the following operations:

1. Configures the Kaggle API.
2. Downloads the competition dataset.
3. Reads the training and test data from the downloaded ZIP archive.
4. Selects atmospheric state and forcing variables.
5. Expands vertically resolved variables into 60 atmospheric levels.
6. Normalizes the input and target features.
7. Trains a fully connected neural network using TensorFlow/Keras.
8. Applies custom post-processing weights to the predictions.
9. Generates a `submission.csv` file.

## Project Structure

A recommended project structure is:

```text
leap-climsim/
├── LEAP_Atmospheric_Physics_using_AI_ClimSim.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
└── outputs/
    └── .gitkeep
```

The competition dataset should not be committed to the Git repository.

## Environment

The notebook was designed primarily for Google Colab.

It uses:

* Python
* pandas
* NumPy
* scikit-learn
* TensorFlow/Keras
* Kaggle CLI
* Git
* Git LFS
* ZIP file processing

## Requirements

Install the Python dependencies with:

```bash
pip install \
    kaggle \
    pandas \
    numpy \
    scikit-learn \
    tensorflow
```

A possible `requirements.txt` is:

```text
kaggle
numpy
pandas
scikit-learn
tensorflow
```

Git and Git LFS are only necessary when the project source code or model artifacts need to be stored in a repository.

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
```

## Kaggle API Configuration

The notebook uses the Kaggle command-line interface to download the dataset.

First, create a Kaggle API token from the Kaggle account settings. This downloads a file named:

```text
kaggle.json
```

In Google Colab, upload it with:

```python
from google.colab import files

files.upload()
```

Then move it to the Kaggle configuration directory:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

The `chmod 600` command prevents other users on the system from reading the API credentials.

Do not commit `kaggle.json` to Git.

A recommended `.gitignore` entry is:

```gitignore
kaggle.json
.kaggle/
```

## Dataset Download

The notebook downloads the ClimSim competition dataset with:

```bash
kaggle competitions download \
    -c leap-atmospheric-physics-ai-climsim
```

The generated file is:

```text
leap-atmospheric-physics-ai-climsim.zip
```

In the notebook execution, the downloaded archive was approximately 72.5 GB.

Because of its size, the archive should remain outside the Git repository. Users should download it directly from Kaggle when setting up the project.

## Dataset Files

The notebook expects the ZIP archive to contain:

```text
train.csv
test.csv
```

The files are read directly from the archive:

```python
import zipfile
import pandas as pd


zip_path = "/content/leap-atmospheric-physics-ai-climsim.zip"

with zipfile.ZipFile(zip_path, "r") as archive:
    with archive.open("train.csv") as train_file:
        train_df = pd.read_csv(train_file)

    with archive.open("test.csv") as test_file:
        test_df = pd.read_csv(test_file)
```

This avoids extracting the CSV files to disk, but the complete DataFrames are still loaded into memory.

## Input Variables

The model uses atmospheric state variables, surface variables, radiative properties, and atmospheric composition variables.

The original input groups are:

```python
input_columns = [
    "state_t",
    "state_q0001",
    "state_q0002",
    "state_q0003",
    "state_u",
    "state_v",
    "state_ps",
    "pbuf_SOLIN",
    "pbuf_LHFLX",
    "pbuf_SHFLX",
    "pbuf_TAUX",
    "pbuf_TAUY",
    "pbuf_COSZRS",
    "cam_in_ALDIF",
    "cam_in_ALDIR",
    "cam_in_ASDIF",
    "cam_in_ASDIR",
    "cam_in_LWUP",
    "cam_in_ICEFRAC",
    "cam_in_LANDFRAC",
    "cam_in_OCNFRAC",
    "cam_in_SNOWHLAND",
    "pbuf_ozone",
    "pbuf_CH4",
    "pbuf_N2O"
]
```

The following variables contain 60 vertical atmospheric levels:

```text
state_t
state_q0001
state_q0002
state_q0003
state_u
state_v
pbuf_ozone
pbuf_CH4
pbuf_N2O
```

Each vertically resolved variable is expanded into columns from level `0` through level `59`.

Example:

```text
state_t_0
state_t_1
...
state_t_59
```

After expansion, the model uses **556 input features**:

* 540 vertically resolved features
* 16 scalar features

## Target Variables

The predicted variables are:

```python
target_columns = [
    "ptend_t",
    "ptend_q0001",
    "ptend_q0002",
    "ptend_q0003",
    "ptend_u",
    "ptend_v",
    "cam_out_NETSW",
    "cam_out_FLWDS",
    "cam_out_PRECSC",
    "cam_out_PRECC",
    "cam_out_SOLS",
    "cam_out_SOLL",
    "cam_out_SOLSD",
    "cam_out_SOLLD"
]
```

The following target variables contain 60 vertical levels:

```text
ptend_t
ptend_q0001
ptend_q0002
ptend_q0003
ptend_u
ptend_v
```

After expansion, the model predicts **368 output values**:

* 360 vertically resolved outputs
* 8 scalar outputs

## Training and Validation Split

The training data is divided into training and validation sets:

```python
from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The split allocates:

* 80% of the rows to training
* 20% of the rows to validation

Using `random_state=42` makes the train-validation split reproducible.

## Feature Scaling

The notebook uses separate `StandardScaler` instances for inputs and targets:

```python
from sklearn.preprocessing import StandardScaler


scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
```

For each feature, standardization approximately applies:

```text
scaled value = (original value - training mean) / training standard deviation
```

The scalers are fitted only on the training set to avoid using validation statistics during training.

## Neural Network Architecture

The notebook creates a fully connected neural network with the following architecture:

```text
Input: 556 features

Dense layer: 512 units
LeakyReLU
Dropout: 20%

Dense layer: 256 units
LeakyReLU
Dropout: 20%

Dense layer: 128 units
LeakyReLU

Output layer: 368 units
Linear activation
```

The corresponding Keras implementation is:

```python
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential


model = Sequential([
    Dense(
        512,
        input_shape=(X_train_scaled.shape[1],)
    ),
    LeakyReLU(negative_slope=0.01),
    Dropout(0.2),

    Dense(256),
    LeakyReLU(negative_slope=0.01),
    Dropout(0.2),

    Dense(128),
    LeakyReLU(negative_slope=0.01),

    Dense(
        y_train_scaled.shape[1],
        activation="linear"
    )
])
```

A linear output activation is used because this is a multi-output regression problem.

## Model Compilation

The model is compiled with:

```python
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)
```

The optimization configuration uses:

* Adam optimizer
* Mean squared error as the training loss
* Mean absolute error as an additional metric

## Model Training

The notebook trains the neural network with:

```python
model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=50,
    batch_size=32,
    validation_data=(
        X_val_scaled,
        y_val_scaled
    )
)
```

The configured training parameters are:

```text
Epochs: 50
Batch size: 32
Validation set: 20% of the original training data
```

## Prediction

After training, predictions are generated for the validation set:

```python
y_val_pred_scaled = model.predict(
    X_val_scaled
)
```

The scaled predictions are converted back to their original units:

```python
y_val_pred = scaler_y.inverse_transform(
    y_val_pred_scaled
)
```

The same process is applied to the test dataset:

```python
X_test_scaled = scaler_X.transform(
    test_df[input_columns_expanded]
)

y_test_pred_scaled = model.predict(
    X_test_scaled
)

y_test_pred = scaler_y.inverse_transform(
    y_test_pred_scaled
)
```

## Custom Prediction Weighting

The notebook calculates the standard deviation of each target using the training dataset:

```python
std_devs = y_train.std()
inverse_std_devs = 1 / std_devs
```

It then applies custom post-processing to the predicted values.

### Zeroed atmospheric levels

For the following variables, levels `0` through `11` are set to zero:

```text
ptend_q0001
ptend_q0002
ptend_q0003
ptend_u
ptend_v
```

Additionally, levels `12` through `14` of `ptend_q0002` are set to zero.

### Inverse standard-deviation multiplication

Each remaining prediction is multiplied by the inverse standard deviation of its corresponding target:

```python
weighted_predictions[column] *= inverse_std_devs[column]
```

The function is:

```python
def apply_weighting(predictions, inverse_std_devs):
    weighted_predictions = predictions.copy()

    for variable in [
        "ptend_q0001",
        "ptend_q0002",
        "ptend_q0003",
        "ptend_u",
        "ptend_v"
    ]:
        for level in range(12):
            column = f"{variable}_{level}"

            if column in weighted_predictions:
                weighted_predictions[column] = 0.0

    for level in range(12, 15):
        column = f"ptend_q0002_{level}"

        if column in weighted_predictions:
            weighted_predictions[column] = 0.0

    for column in weighted_predictions.columns:
        if column in inverse_std_devs:
            weighted_predictions[column] *= (
                inverse_std_devs[column]
            )

    return weighted_predictions
```

This is a custom post-processing heuristic. It should not automatically be assumed to reproduce the official competition evaluation metric.

## Validation Metric

The notebook calculates a mean squared error between the original validation targets and the post-processed predictions:

```python
from sklearn.metrics import mean_squared_error


weighted_mse = mean_squared_error(
    y_val,
    y_val_pred_weighted
)

print(
    f"Validation Weighted MSE: {weighted_mse}"
)
```

Because only the predictions are multiplied by the inverse standard deviations, this calculation is not equivalent to applying a conventional weighted MSE to the errors.

A conventional weighted metric would normally apply weights to the squared differences:

```text
weight × (actual - prediction)²
```

The metric implementation should be reviewed against the competition's official scoring logic.

## Submission Generation

The final predictions are written to:

```text
submission.csv
```

Using:

```python
y_test_pred_weighted.to_csv(
    "submission.csv",
    index=False
)
```

The output contains the expanded target columns but does not explicitly copy an identifier column or start from the competition's sample submission file.

Before submitting the file, verify that:

* Every required column is present
* The column order matches the expected format
* Any required identifier column is included
* The number of rows matches the test dataset
* The prediction transformations match the official evaluation requirements

## Running the Notebook

### Google Colab

1. Upload the notebook to Google Colab.
2. Enable an appropriate runtime.
3. Upload `kaggle.json`.
4. Configure the Kaggle credentials.
5. Download the competition dataset.
6. Run the preprocessing and training cell.
7. Download the generated `submission.csv`.

A GPU runtime can be enabled through:

```text
Runtime → Change runtime type → Hardware accelerator → GPU
```

### Local Environment

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Configure the Kaggle credentials:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Download the dataset:

```bash
kaggle competitions download \
    -c leap-atmospheric-physics-ai-climsim
```

Open the notebook:

```bash
jupyter notebook
```

## Git Repository Integration

The notebook contains an attempted workflow for moving the downloaded ZIP archive into a Git repository:

```bash
git clone https://github.com/gabriel-ac6/LEAP-dataset.git
mv leap-atmospheric-physics-ai-climsim.zip LEAP-dataset/
cd LEAP-dataset
git add leap-atmospheric-physics-ai-climsim.zip
git commit -m "Added Leap Atmospheric Physics AI ClimSim data"
git push
```

This approach should not be used for the full competition archive.

The dataset is extremely large and should be downloaded from Kaggle rather than stored in GitHub.

A repository should contain:

* Notebook source code
* Python scripts
* Configuration examples
* Documentation
* Small sample files
* Model definitions
* Evaluation code

It should not contain:

* The complete competition ZIP
* `train.csv`
* `test.csv`
* Kaggle credentials
* Temporary model checkpoints
* Large generated prediction files

Recommended `.gitignore`:

```gitignore
# Credentials
kaggle.json
.env

# Dataset
*.zip
data/*
!data/.gitkeep
train.csv
test.csv

# Model artifacts
models/*
!models/.gitkeep
*.keras
*.h5
*.ckpt

# Outputs
outputs/*
!outputs/.gitkeep
submission.csv

# Notebook artifacts
.ipynb_checkpoints/

# Python
__pycache__/
*.py[cod]

# Virtual environment
.venv/
venv/
```

## Google Colab Shell Behavior

In Google Colab, each command beginning with `!` runs in a separate shell.

Therefore, this sequence does not reliably preserve the directory change:

```python
!cd LEAP-dataset
!git add file.zip
!git commit -m "Commit"
```

The `git add` command may run outside `LEAP-dataset`.

Use a single shell command:

```bash
cd LEAP-dataset &&
git add . &&
git commit -m "Update project" &&
git push
```

In Colab:

```python
!cd LEAP-dataset && \
 git add . && \
 git commit -m "Update project" && \
 git push
```

Alternatively, use the Colab directory magic:

```python
%cd /content/LEAP-dataset
```

Unlike `!cd`, `%cd` updates the notebook's current working directory.

## Known Issues

### Entire dataset loaded into memory

The notebook reads the complete `train.csv` and `test.csv` files into pandas DataFrames.

Given the dataset size, this can exceed the available Colab RAM.

Potential alternatives include:

* Reading the CSV in chunks
* Loading a subset for initial experiments
* Using Parquet files
* Using TensorFlow input pipelines
* Using Dask or Polars
* Training incrementally
* Using memory-mapped or streaming datasets

### Large scaled arrays

`StandardScaler` creates additional NumPy arrays for:

* Training inputs
* Validation inputs
* Training targets
* Validation targets
* Test inputs
* Predictions

This can substantially increase peak memory usage.

### No missing-value handling

The notebook does not check for:

* Missing values
* Infinite values
* Constant columns
* Zero target standard deviations

A zero standard deviation would produce an infinite inverse standard deviation:

```python
inverse_std_devs = 1 / std_devs
```

A safer implementation is:

```python
inverse_std_devs = (
    1 / std_devs.replace(0, np.nan)
).fillna(0)
```

### No training callbacks

The current training process always runs for 50 epochs.

It does not use:

* Early stopping
* Learning-rate reduction
* Model checkpoints
* TensorBoard logging

Example:

```python
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2
    ),
    ModelCheckpoint(
        "best_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]
```

### Model and scalers are not saved

The notebook generates a submission but does not persist:

* The trained model
* The input scaler
* The target scaler
* The expanded input columns
* The expanded target columns

Example:

```python
import joblib


model.save("climsim_model.keras")
joblib.dump(scaler_X, "scaler_X.joblib")
joblib.dump(scaler_y, "scaler_y.joblib")
```

### Partial reproducibility

The train-validation split has a fixed random state, but NumPy and TensorFlow seeds are not configured.

Example:

```python
import random
import numpy as np
import tensorflow as tf


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

### Post-processing requires validation

The weighting function directly modifies predicted values.

This may change their physical units and does not necessarily correspond to the official scoring formula.

The official metric and sample submission format should be implemented separately from the physical predictions.

### No physical constraints

The neural network is a generic dense regressor.

It does not explicitly enforce:

* Conservation laws
* Atmospheric energy balance
* Moisture constraints
* Vertical relationships
* Physical bounds
* Cross-variable consistency

## Suggested Improvements

Recommended next steps include:

1. Build a memory-efficient data pipeline.
2. Train initially on a manageable data subset.
3. Reproduce the official evaluation metric exactly.
4. Use the official sample submission as the output template.
5. Separate prediction generation from metric weighting.
6. Save the model and preprocessing objects.
7. Add early stopping and model checkpoints.
8. Add experiment tracking.
9. Evaluate errors separately by variable and atmospheric level.
10. Investigate architectures that preserve vertical atmospheric structure.

## Security Notes

* Never commit `kaggle.json`.
* Do not display Kaggle credentials in notebook output.
* Use SSH keys or secure tokens for Git operations.
* Do not store credentials directly in notebook cells.
* Remove notebook outputs that may contain secrets before publishing.
* Revoke any credential that was accidentally committed.

## Output Files

The notebook may generate:

```text
leap-atmospheric-physics-ai-climsim.zip
submission.csv
```

Recommended handling:

| File                         | Commit to Git                  |
| ---------------------------- | ------------------------------ |
| Notebook                     | Yes                            |
| README                       | Yes                            |
| Requirements                 | Yes                            |
| Small configuration examples | Yes                            |
| Competition ZIP              | No                             |
| Training CSV                 | No                             |
| Test CSV                     | No                             |
| Kaggle API credentials       | No                             |
| Trained model                | Only when appropriately stored |
| Submission CSV               | Usually no                     |

## License and Data Usage

The source code and the competition dataset may be subject to different licenses and usage rules.

Users should review the competition rules and dataset terms before redistributing data, trained artifacts, or derived outputs.
