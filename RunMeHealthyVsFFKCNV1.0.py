import os
import glob
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# -------------------------------
# Global settings and preset class weights for each scenario
# -------------------------------

IMAGE_SIZE = (512, 512)
BATCH_SIZE = 5
EPOCHS = 150

# For this binary task, only "normal" and "ffkcn" images are used.
# Set the weights for each external validation scenario.
EXTERNAL_WEIGHTS = {
    "scenario1": {0: 1.0, 1: 1.0},
    "scenario2": {0: 1.0, 1: 1.0},
    "scenario3": {0: 1.0, 1: 1.0}
}

# Set the weights for the k-fold cross validation scenario.
KFOLD_WEIGHTS = {0: 1.0, 1: 1.0}

# These are the class names that will be shown in plots.
BINARY_CLASS_NAMES = ["normal", "ffkcn"]

# -------------------------------
# Helper function: Find optimal threshold
# -------------------------------

def find_optimal_threshold(y_true, y_probs, step=0.01):
    """
    Iterates over thresholds between 0 and 1 to find the one that minimizes
    the absolute difference between the true positive rate and the true negative rate.
    """
    best_thresh = 0.5
    best_diff = float('inf')
    thresholds = np.arange(0.0, 1.0, step)
    for thresh in thresholds:
        y_pred_temp = (y_probs >= thresh).astype(int)
        # Ensure we have a 2x2 confusion matrix.
        cm = metrics.confusion_matrix(y_true, y_pred_temp)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        diff = abs(tpr - tnr)
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh
    return best_thresh

# -------------------------------
# Data loading functions for normal vs. ffkcn
# -------------------------------

def get_file_paths_and_labels(dir_list):
    """
    Traverse the given dataset directories and return file paths and labels for images 
    in the "normal" and "ffkcn" subfolders.
    Label 0 for "normal", and label 1 for "ffkcn".
    """
    file_paths = []
    labels = []
    for base_dir in dir_list:
        for class_name in ["normal", "ffkcn"]:
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for ext in ['png', 'jpg', 'jpeg']:
                pattern = os.path.join(class_dir, f'*.{ext}')
                files = glob.glob(pattern)
                file_paths.extend(files)
                label = 0 if class_name == "normal" else 1
                labels.extend([label] * len(files))
    return file_paths, labels

def process_image(file_path, label, augment=False):
    """
    Read an image file, decode it, convert to float32, and resize to IMAGE_SIZE.
    Optionally perform basic augmentation (only used for training data).
    """
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label

def create_dataset(file_paths, labels, batch_size, augment=False, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths))
    ds = ds.map(lambda x, y: process_image(x, y, augment), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------------------
# Model building (binary for normal vs. ffkcn)
# ----------------------------

def build_binary_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)):
    """
    Build the binary classification model using the provided architecture.
    """
    inputs = keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(6, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(12, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(18, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(18, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(6, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    return model

# ----------------------------
# Evaluation & plotting functions
# ----------------------------

def evaluate_model(model, dataset):
    """
    Run inference on the dataset and return true labels and prediction probabilities.
    (Thresholding is postponed until after the optimal threshold is found.)
    """
    y_true, y_probs = [], []
    for images, labels in dataset:
        preds = model.predict(images)
        y_probs.append(preds)
        y_true.extend(labels.numpy())
    y_probs = np.concatenate(y_probs, axis=0).flatten()
    return np.array(y_true), y_probs

def save_metrics(metrics_dict, filepath):
    with open(filepath, 'w') as f:
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value}\n")

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', save_path='cm.pdf'):
    if normalize:
        cm_to_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_to_plot = cm
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_to_plot, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm_to_plot.max() / 2.
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            plt.text(j, i, format(cm_to_plot[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_to_plot[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path='roc.pdf'):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_probs)
    auc_score = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(save_path, format='pdf')
    plt.close()

def plot_precision_recall_curve(y_true, y_probs, save_path='pr.pdf'):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_probs)
    ap = metrics.average_precision_score(y_true, y_probs)
    plt.figure()
    plt.plot(recall, precision, label=f"Precision-Recall (AP = {ap:.2f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.savefig(save_path, format='pdf')
    plt.close()

# ----------------------------
# External Validation Scenario
# ----------------------------

def run_external_validation_scenario(scenario_name, train_dirs, val_dirs, batch_size=BATCH_SIZE, epochs=EPOCHS):
    """
    Run an external validation scenario for normal vs. ffkcn.
    Uses preset weights from EXTERNAL_WEIGHTS for the given scenario.
    After training, the code finds the optimal threshold (minimizing |TPR - TNR|) and uses it for evaluation.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"{scenario_name}_normal_vs_ffkcn_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Load file paths and labels.
    train_files, train_labels = get_file_paths_and_labels(train_dirs)
    val_files, val_labels = get_file_paths_and_labels(val_dirs)

    train_ds = create_dataset(train_files, train_labels, batch_size, augment=True, shuffle=True)
    val_ds = create_dataset(val_files, val_labels, batch_size, augment=False, shuffle=False)

    model = build_binary_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, epochs=epochs, class_weight=EXTERNAL_WEIGHTS[scenario_name], verbose=0)

    # Get true labels and prediction probabilities.
    y_true, y_probs = evaluate_model(model, val_ds)
    # Find the optimal threshold.
    optimal_thresh = find_optimal_threshold(y_true, y_probs, step=0.01)
    # Compute final predictions using the found threshold.
    y_pred = (y_probs >= optimal_thresh).astype(int)

    cm = metrics.confusion_matrix(y_true, y_pred)
    metrics_dict = {
        "Optimal Threshold": optimal_thresh,
        "Accuracy": metrics.accuracy_score(y_true, y_pred),
        "Precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "Recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": metrics.f1_score(y_true, y_pred, zero_division=0),
        "R2 Score": metrics.r2_score(y_true, y_pred),
        "Confusion Matrix": cm.tolist()
    }
    save_metrics(metrics_dict, os.path.join(result_dir, "metrics.txt"))
    plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES, normalize=False,
                          title="Confusion Matrix",
                          save_path=os.path.join(result_dir, "confusion_matrix.pdf"))
    plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES, normalize=True,
                          title="Normalized Confusion Matrix",
                          save_path=os.path.join(result_dir, "confusion_matrix_normalized.pdf"))
    plot_roc_curve(y_true, y_probs, save_path=os.path.join(result_dir, "roc_curve.pdf"))
    plot_precision_recall_curve(y_true, y_probs, save_path=os.path.join(result_dir, "precision_recall_curve.pdf"))
    model.save(os.path.join(result_dir, "model.h5"))
    np.savez(os.path.join(result_dir, "evaluation_data.npz"), y_true=y_true, y_pred=y_pred, y_probs=y_probs)

# ----------------------------
# K-Fold Cross Validation
# ----------------------------

def run_kfold_cross_validation(merged_dirs, k=5, batch_size=BATCH_SIZE, epochs=EPOCHS):
    """
    Run k-fold cross validation on merged datasets for normal vs. ffkcn.
    Uses preset weights from KFOLD_WEIGHTS for each fold.
    For each fold, the optimal threshold is determined and used.
    Then, global (average) metrics are computed from all folds using a global optimal threshold.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_result_dir = os.path.join("results", f"kfold_normal_vs_ffkcn_{timestamp}")
    os.makedirs(base_result_dir, exist_ok=True)

    file_paths, labels = get_file_paths_and_labels(merged_dirs)
    file_paths = np.array(file_paths)
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    y_true_all, y_probs_all, fold_thresholds = [], [], []
    fold_idx = 1

    for train_idx, val_idx in skf.split(file_paths, labels):
        fold_dir = os.path.join(base_result_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        train_files, val_files = file_paths[train_idx], file_paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]

        train_ds = create_dataset(train_files, train_labels, batch_size, augment=True, shuffle=True)
        val_ds = create_dataset(val_files, val_labels, batch_size, augment=False, shuffle=False)

        model = build_binary_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, epochs=epochs, class_weight=KFOLD_WEIGHTS, verbose=0)

        y_true_fold, y_probs_fold = evaluate_model(model, val_ds)
        optimal_thresh = find_optimal_threshold(y_true_fold, y_probs_fold, step=0.01)
        fold_thresholds.append(optimal_thresh)
        y_pred_fold = (y_probs_fold >= optimal_thresh).astype(int)

        y_true_all.extend(y_true_fold)
        y_probs_all.extend(y_probs_fold)

        cm = metrics.confusion_matrix(y_true_fold, y_pred_fold)
        metrics_dict = {
            "Optimal Threshold": optimal_thresh,
            "Accuracy": metrics.accuracy_score(y_true_fold, y_pred_fold),
            "Precision": metrics.precision_score(y_true_fold, y_pred_fold, zero_division=0),
            "Recall": metrics.recall_score(y_true_fold, y_pred_fold, zero_division=0),
            "F1 Score": metrics.f1_score(y_true_fold, y_pred_fold, zero_division=0),
            "R2 Score": metrics.r2_score(y_true_fold, y_pred_fold),
            "Confusion Matrix": cm.tolist()
        }
        save_metrics(metrics_dict, os.path.join(fold_dir, "metrics.txt"))
        plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES, normalize=False,
                              title="Confusion Matrix",
                              save_path=os.path.join(fold_dir, "confusion_matrix.pdf"))
        plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES, normalize=True,
                              title="Normalized Confusion Matrix",
                              save_path=os.path.join(fold_dir, "confusion_matrix_normalized.pdf"))
        plot_roc_curve(y_true_fold, y_probs_fold, save_path=os.path.join(fold_dir, "roc_curve.pdf"))
        plot_precision_recall_curve(y_true_fold, y_probs_fold, save_path=os.path.join(fold_dir, "precision_recall_curve.pdf"))
        model.save(os.path.join(fold_dir, "model.h5"))
        np.savez(os.path.join(fold_dir, "evaluation_data.npz"), y_true=y_true_fold, y_pred=y_pred_fold, y_probs=y_probs_fold)
        fold_idx += 1

    # Aggregate results from all folds.
    y_true_all = np.array(y_true_all)
    y_probs_all = np.array(y_probs_all)
    # Find a global optimal threshold on the aggregated data.
    global_optimal_thresh = find_optimal_threshold(y_true_all, y_probs_all, step=0.01)
    y_pred_all = (y_probs_all >= global_optimal_thresh).astype(int)
    cm = metrics.confusion_matrix(y_true_all, y_pred_all)
    avg_metrics = {
        "Global Optimal Threshold": global_optimal_thresh,
        "Average Accuracy": metrics.accuracy_score(y_true_all, y_pred_all),
        "Average Precision": metrics.precision_score(y_true_all, y_pred_all, zero_division=0),
        "Average Recall": metrics.recall_score(y_true_all, y_pred_all, zero_division=0),
        "Average F1 Score": metrics.f1_score(y_true_all, y_pred_all, zero_division=0),
        "Average R2 Score": metrics.r2_score(y_true_all, y_pred_all),
        "Global Confusion Matrix": cm.tolist(),
        "Fold Optimal Thresholds": fold_thresholds
    }
    avg_dir = os.path.join(base_result_dir, "average_results")
    os.makedirs(avg_dir, exist_ok=True)
    save_metrics(avg_metrics, os.path.join(avg_dir, "average_metrics.txt"))
    plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES, normalize=False,
                          title="Average Confusion Matrix",
                          save_path=os.path.join(avg_dir, "confusion_matrix.pdf"))
    plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES, normalize=True,
                          title="Average Normalized Confusion Matrix",
                          save_path=os.path.join(avg_dir, "confusion_matrix_normalized.pdf"))
    plot_roc_curve(y_true_all, y_probs_all, save_path=os.path.join(avg_dir, "roc_curve.pdf"))
    plot_precision_recall_curve(y_true_all, y_probs_all, save_path=os.path.join(avg_dir, "precision_recall_curve.pdf"))

# ----------------------------
# Main routine
# ----------------------------

def main():
    # Define dataset directories (adjust paths as needed)
    dataset_dirs = {
        "H": "data/dataset.H",
        "K": "data/dataset.K",
        "M": "data/dataset.M"
    }
    external_scenarios = {
        "scenario1": {"train": [dataset_dirs["K"], dataset_dirs["M"]], "val": [dataset_dirs["H"]]},
        "scenario2": {"train": [dataset_dirs["H"], dataset_dirs["M"]], "val": [dataset_dirs["K"]]},
        "scenario3": {"train": [dataset_dirs["H"], dataset_dirs["K"]], "val": [dataset_dirs["M"]]}
    }
    
    # Run external validation experiments.
    for scenario, dirs in external_scenarios.items():
        run_external_validation_scenario(scenario, dirs["train"], dirs["val"])
    
    # Run k-fold cross validation using merged data from all datasets.
    merged_dirs = [dataset_dirs["H"], dataset_dirs["K"], dataset_dirs["M"]]
    run_kfold_cross_validation(merged_dirs, k=5)

if __name__ == '__main__':
    main()
