"""
=============================================================================
CODVEDA INTERNSHIP - MACHINE LEARNING

TASK 1: Random Forest Classifier        (churn-bigml-80/20.csv)
TASK 2: SVM for Classification          (iris.csv)
TASK 3: Neural Network (MLP/Keras)      (churn-bigml-80/20.csv)

Run: python level3_all_tasks.py
Dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn
Optional:     tensorflow (Task 3 will auto-fallback to sklearn MLP if absent)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
)

# =============================================================================
# SHARED PREPROCESSING HELPER
# =============================================================================

def preprocess_churn(df):
    """Clean and encode the churn dataset."""
    df = df.copy()
    df.drop(columns=["State", "Area code"], inplace=True)
    le = LabelEncoder()
    for col in ["International plan", "Voice mail plan"]:
        df[col] = le.fit_transform(df[col])   # No->0, Yes->1
    df["Churn"] = df["Churn"].astype(int)      # False->0, True->1
    return df


# =============================================================================
# TASK 1: RANDOM FOREST CLASSIFIER
# =============================================================================

def run_task1():
    print("\n")
    print("=" * 70)
    print("  TASK 1: RANDOM FOREST CLASSIFIER")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading Data...")
    train_df = pd.read_csv("churn-bigml-80.csv")
    test_df  = pd.read_csv("churn-bigml-20.csv")
    print(f"    Train: {train_df.shape}  |  Test: {test_df.shape}")
    print(f"    Churn distribution (train):\n{train_df['Churn'].value_counts()}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2] Preprocessing...")
    train_df = preprocess_churn(train_df)
    test_df  = preprocess_churn(test_df)

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    X_test  = test_df.drop(columns=["Churn"])
    y_test  = test_df["Churn"]
    print(f"    Missing values: {X_train.isnull().sum().sum()}")

    # ── 3. Baseline model ─────────────────────────────────────────────────────
    print("\n[3] Training Baseline Random Forest (100 trees, default params)...")
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_base.fit(X_train, y_train)
    y_pred_base = rf_base.predict(X_test)
    print(f"    Accuracy : {accuracy_score(y_test, y_pred_base):.4f}")
    print(f"    F1-Score : {f1_score(y_test, y_pred_base):.4f}")

    # ── 4. Hyperparameter tuning ──────────────────────────────────────────────
    print("\n[4] Hyperparameter Tuning via GridSearchCV (5-fold CV)...")
    param_grid = {
        "n_estimators"     : [100, 200],
        "max_depth"        : [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"    Best Params   : {grid.best_params_}")
    print(f"    Best CV F1    : {grid.best_score_:.4f}")
    best_rf = grid.best_estimator_

    # ── 5. Cross-validation ───────────────────────────────────────────────────
    print("\n[5] Cross-Validation on Best Model:")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring=metric)
        print(f"    {metric.capitalize():12s}: {scores.mean():.4f} +/- {scores.std():.4f}")

    # ── 6. Final evaluation ───────────────────────────────────────────────────
    print("\n[6] Final Evaluation on Test Set:")
    y_pred = best_rf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ── 7. Visualisations ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Task 1: Random Forest Classifier", fontsize=14, fontweight="bold")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    axes[0].set_title("Confusion Matrix (Test Set)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Feature importance
    importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
    importances.sort_values().plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Feature Importance (Mean Decrease in Impurity)")
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("task1_random_forest_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("    Saved: task1_random_forest_results.png")

    # Top 5 features
    top5 = importances.sort_values(ascending=False).head(5)
    print("\n    Top 5 Most Important Features:")
    for feat, score in top5.items():
        print(f"      {feat:35s}: {score:.4f}")

    print("""
    EXPLANATION - RANDOM FOREST:
    -----------------------------------------------------------------------
    A Random Forest trains N decision trees, each on a random bootstrap
    sample of the data with a random subset of features (bagging). Final
    prediction = majority vote across all trees.

    WHY IT WORKS:
      - Individual trees are high variance (overfit). Averaging many trees
        reduces variance without increasing bias. This is the bias-variance
        tradeoff in action.

    HYPERPARAMETERS TUNED:
      n_estimators      - More trees = more stable, but diminishing returns.
      max_depth         - Limits tree depth; shallower = less overfitting.
      min_samples_split - Minimum data points needed to split a node.

    FEATURE IMPORTANCE:
      Measures how much each feature reduces impurity (Gini) when used as
      a split. Higher = more predictive. E.g., "Total day minutes" is
      typically the top churn predictor.

    CROSS-VALIDATION:
      5-fold CV splits training data into 5 parts, trains on 4, evaluates
      on 1, and rotates. Mean score = unbiased generalisation estimate.
    """)


# =============================================================================
# TASK 2: SUPPORT VECTOR MACHINE (SVM)
# =============================================================================

def run_task2():
    print("\n")
    print("=" * 70)
    print("  TASK 2: SUPPORT VECTOR MACHINE (SVM) FOR CLASSIFICATION")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading Data (Iris dataset)...")
    df = pd.read_csv("iris.csv")
    print(f"    Shape   : {df.shape}")
    print(f"    Classes : {df['species'].unique()}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2] Preprocessing (binary: versicolor vs virginica for clean 2D demo)...")
    # For decision boundary visualisation we keep 2 features & 2 classes
    df_bin = df[df["species"].isin(["versicolor", "virginica"])].copy()
    le = LabelEncoder()
    df_bin["target"] = le.fit_transform(df_bin["species"])

    X = df_bin[["petal_length", "petal_width"]].values
    y = df_bin["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y)
    print(f"    Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"    Classes: {le.classes_[0]}=0, {le.classes_[1]}=1")

    # ── 3. Linear SVM ─────────────────────────────────────────────────────────
    print("\n[3] Training SVM - Linear Kernel...")
    svm_lin = SVC(kernel="linear", probability=True, random_state=42)
    svm_lin.fit(X_train, y_train)
    y_pred_lin = svm_lin.predict(X_test)
    y_prob_lin = svm_lin.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_lin, target_names=le.classes_))
    print(f"    AUC (Linear): {roc_auc_score(y_test, y_prob_lin):.4f}")

    # ── 4. RBF SVM ────────────────────────────────────────────────────────────
    print("\n[4] Training SVM - RBF Kernel...")
    svm_rbf = SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42)
    svm_rbf.fit(X_train, y_train)
    y_pred_rbf = svm_rbf.predict(X_test)
    y_prob_rbf = svm_rbf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred_rbf, target_names=le.classes_))
    print(f"    AUC (RBF): {roc_auc_score(y_test, y_prob_rbf):.4f}")

    # ── 5. Comparison table ───────────────────────────────────────────────────
    print("\n[5] Kernel Comparison:")
    comp = pd.DataFrame({
        "Kernel"   : ["Linear", "RBF"],
        "Accuracy" : [accuracy_score(y_test, y_pred_lin),
                      accuracy_score(y_test, y_pred_rbf)],
        "Precision": [precision_score(y_test, y_pred_lin),
                      precision_score(y_test, y_pred_rbf)],
        "Recall"   : [recall_score(y_test, y_pred_lin),
                      recall_score(y_test, y_pred_rbf)],
        "AUC"      : [roc_auc_score(y_test, y_prob_lin),
                      roc_auc_score(y_test, y_prob_rbf)],
    })
    print(comp.to_string(index=False))

    # ── 6. Decision boundary helper ───────────────────────────────────────────
    def plot_boundary(model, X, y, ax, title):
        h = 0.02
        x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x0_min, x0_max, h),
                             np.arange(x1_min, x1_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=60)
        ax.set_title(title)
        ax.set_xlabel("Petal Length (scaled)")
        ax.set_ylabel("Petal Width (scaled)")

    # ── 7. Visualisations ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Task 2: SVM Classification (Iris)", fontsize=14, fontweight="bold")

    plot_boundary(svm_lin, X_train, y_train, axes[0, 0], "Linear Kernel - Decision Boundary")
    plot_boundary(svm_rbf, X_train, y_train, axes[0, 1], "RBF Kernel - Decision Boundary")

    for ax, y_p, title in zip(
        [axes[1, 0], axes[1, 1]], [y_pred_lin, y_pred_rbf],
        ["Confusion Matrix - Linear", "Confusion Matrix - RBF"]
    ):
        cm = confusion_matrix(y_test, y_p)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("task2_svm_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob_lin, name="Linear Kernel", ax=ax2)
    RocCurveDisplay.from_predictions(y_test, y_prob_rbf, name="RBF Kernel",    ax=ax2)
    ax2.set_title("ROC Curves: Linear vs RBF Kernel")
    plt.savefig("task2_svm_roc.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("    Saved: task2_svm_results.png | task2_svm_roc.png")

    print("""
    EXPLANATION - SVM:
    -----------------------------------------------------------------------
    SVM finds the decision boundary (hyperplane) that maximises the margin
    between the two classes. Data points on the margin boundary are called
    "support vectors" - they define the boundary.

    LINEAR KERNEL:
      Decision boundary is a straight line (in 2D) or flat hyperplane (nD).
      Best when data is linearly separable. C controls regularisation:
        - Low C  = wider margin, more misclassifications allowed (underfits)
        - High C = narrow margin, fewer misclassifications (can overfit)

    RBF KERNEL (Radial Basis Function):
      Implicitly maps data to infinite-dimensional space using the kernel
      trick. Creates non-linear (curved) decision boundaries. Controlled by:
        - C     : same regularisation trade-off as linear
        - gamma : how far a single training point's influence reaches.
                  High gamma = tight boundaries (overfit risk).

    WHY SCALE?: SVM uses Euclidean distance. Unscaled features with large
    values will dominate the distance calculation and bias the model.

    AUC-ROC: plots TPR vs FPR across all thresholds.
      1.0 = perfect classifier, 0.5 = random guessing.
    """)


# =============================================================================
# TASK 3: NEURAL NETWORK WITH KERAS (FALLBACK: sklearn MLP)
# =============================================================================

def run_task3():
    print("\n")
    print("=" * 70)
    print("  TASK 3: NEURAL NETWORK FOR CLASSIFICATION")
    print("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Loading Data...")
    train_df = pd.read_csv("churn-bigml-80.csv")
    test_df  = pd.read_csv("churn-bigml-20.csv")

    train_df = preprocess_churn(train_df)
    test_df  = preprocess_churn(test_df)

    X_all   = train_df.drop(columns=["Churn"]).values
    y_all   = train_df["Churn"].values
    X_test  = test_df.drop(columns=["Churn"]).values
    y_test  = test_df["Churn"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.15, random_state=42, stratify=y_all)

    # Scale (critical for neural networks)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    n_features = X_train.shape[1]
    print(f"    Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    print(f"    Features: {n_features}")

    # ── 2. Try Keras, fallback to sklearn MLP ─────────────────────────────────
    use_keras  = False
    train_loss = []
    val_loss   = []
    train_acc  = []
    val_acc    = []

    try:
        import tensorflow as tf
        keras = tf.keras
        layers = tf.keras.layers
        use_keras = True
        print("\n[2] TensorFlow detected. Building Keras Neural Network...")
        print(f"    TensorFlow version: {tf.__version__}")

        # Architecture
        model = keras.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        model.summary()

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True)

        print("\n    Training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100, batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

        train_loss = history.history["loss"]
        val_loss   = history.history["val_loss"]
        train_acc  = history.history["accuracy"]
        val_acc    = history.history["val_accuracy"]

        y_prob = model.predict(X_test).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

    except ImportError:
        print("\n[2] TensorFlow not installed. Using scikit-learn MLPClassifier...")
        print("    NOTE: To use TensorFlow/Keras, run: pip install tensorflow")

        # sklearn MLP - equivalent architecture
        # hidden_layer_sizes=(64, 32) = two hidden layers
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=42,
            verbose=False
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Extract learning curves
        train_loss = model.loss_curve_
        val_acc    = model.validation_scores_       # accuracy, not loss
        val_loss   = [1 - s for s in val_acc]       # approximate val_loss
        train_acc  = [1 - l for l in train_loss]    # approximate train_acc

    # ── 3. Evaluation ─────────────────────────────────────────────────────────
    print("\n[3] Evaluation on Test Set:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"    AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # ── 4. Visualisations ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Task 3: Neural Network Classifier (Churn)", fontsize=14, fontweight="bold")

    # Loss curve
    epochs = range(1, len(train_loss) + 1)
    axes[0].plot(epochs, train_loss, label="Train Loss",      color="steelblue")
    axes[0].plot(epochs, val_loss,   label="Validation Loss", color="tomato", linestyle="--")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy curve
    axes[1].plot(epochs, train_acc, label="Train Accuracy",      color="steelblue")
    axes[1].plot(epochs, val_acc,   label="Validation Accuracy", color="tomato", linestyle="--")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    axes[2].set_title("Confusion Matrix (Test Set)")
    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig("task3_neural_network_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("    Saved: task3_neural_network_results.png")

    print("""
    EXPLANATION - NEURAL NETWORK:
    -----------------------------------------------------------------------
    ARCHITECTURE:
      Input Layer  (17 neurons) - one per feature
           |
      Dense Layer  (64 neurons, ReLU activation)
           |
      Dropout (30%) - randomly disables neurons during training
           |
      Dense Layer  (32 neurons, ReLU activation)
           |
      Dropout (20%)
           |
      Output Layer (1 neuron, Sigmoid) -> outputs probability [0, 1]

    ACTIVATION FUNCTIONS:
      ReLU (Rectified Linear Unit): f(x) = max(0, x)
        Used in hidden layers. Solves vanishing gradient problem.
      Sigmoid: f(x) = 1 / (1 + e^-x)
        Outputs probability between 0 and 1. Used for binary output.

    TRAINING (BACKPROPAGATION):
      1. Forward pass: input -> predictions
      2. Compute loss: binary_crossentropy = -[y*log(p) + (1-y)*log(1-p)]
      3. Backward pass: compute gradients of loss w.r.t. weights
      4. Update weights via Adam optimizer (adaptive learning rate)
      5. Repeat for each mini-batch (32 samples at a time)

    DROPOUT:
      During each training step, randomly zero out X% of neurons.
      This forces the network to learn redundant representations
      and prevents co-adaptation -> reduces overfitting.

    EARLY STOPPING:
      Monitor validation loss after each epoch. Stop training if
      no improvement for 10 consecutive epochs, then restore the
      weights from the best epoch. Prevents overfitting automatically.

    ADAM OPTIMIZER:
      Adaptive Moment Estimation. Maintains per-parameter learning rates
      based on gradients. Faster and more robust than plain SGD.
    """)


# =============================================================================
# MAIN - RUN ALL TASKS
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  CODVEDA INTERNSHIP - MACHINE LEARNING - LEVEL 3 (ADVANCED)")
    print("=" * 70)
    print("  Running all 3 tasks...")

    run_task1()   # Random Forest
    run_task2()   # SVM
    run_task3()   # Neural Network

    print("\n" + "=" * 70)
    print("  ALL TASKS COMPLETE!")
    print("  Output plots saved:")
    print("    task1_random_forest_results.png")
    print("    task2_svm_results.png")
    print("    task2_svm_roc.png")
    print("    task3_neural_network_results.png")
    print("=" * 70)