import os
import glob
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# -------------------------------
# Global settings and preset class weights
# -------------------------------

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 5
EPOCHS = 150

# Original classes present in the dataset directories
ALL_CLASSES = ['ffkcn', 'kcn', 'normal']

# Predefined class weights for each binary task.
# For task "normal_ffkcn_vs_kcn": images in "normal" and "ffkcn" (combined) are labeled 0,
# and "kcn" is labeled 1.
CLASS_WEIGHT_NORMAL_FFKCN_VS_KCN = {0: 1.0, 1: 1.1}  # Adjust these values as needed

# For task "normal_vs_ffkcn": only images in "normal" (label 0) and "ffkcn" (label 1) are used.
CLASS_WEIGHT_NORMAL_VS_FFKCN = {0: 1.0, 1: 2.0}  # Adjust these values as needed

# Mapping of task names to real binary class names.
BINARY_CLASS_NAMES = {
    "normal_ffkcn_vs_kcn": ["normal/ffkcn", "kcn"],
    "normal_vs_ffkcn": ["normal", "ffkcn"]
}

# -------------------------------
# Data loading functions for binary classification
# -------------------------------

def get_binary_file_paths_and_labels(dir_list, task):
    """
    For a given list of dataset directories and a binary task, load image file paths and assign binary labels.
    For task "normal_ffkcn_vs_kcn": images in "normal" and "ffkcn" get label 0; images in "kcn" get label 1.
    For task "normal_vs_ffkcn": include only images in "normal" and "ffkcn"; assign "normal" -> 0 and "ffkcn" -> 1.
    """
    file_paths = []
    labels = []
    for base_dir in dir_list:
        for class_name in ALL_CLASSES:
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for ext in ['png', 'jpg', 'jpeg']:
                pattern = os.path.join(class_dir, f'*.{ext}')
                files = glob.glob(pattern)
                if task == "normal_ffkcn_vs_kcn":
                    if class_name in ['normal', 'ffkcn']:
                        file_paths.extend(files)
                        labels.extend([0] * len(files))
                    elif class_name == 'kcn':
                        file_paths.extend(files)
                        labels.extend([1] * len(files))
                elif task == "normal_vs_ffkcn":
                    if class_name == "normal":
                        file_paths.extend(files)
                        labels.extend([0] * len(files))
                    elif class_name == "ffkcn":
                        file_paths.extend(files)
                        labels.extend([1] * len(files))
    return file_paths, labels

def process_image(file_path, label, augment=False):
    """
    Read an image from file, decode, convert to float32, and resize to IMAGE_SIZE.
    Optionally apply basic augmentation (only for training images).
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
    ds = ds.map(lambda x, y: process_image(x, y, augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def compute_class_weights(labels):
    """
    Compute class weights to address class imbalance.
    (This function is not used now since we use preset weights.)
    """
    counter = Counter(labels)
    total = sum(counter.values())
    num_classes = len(np.unique(labels))
    class_weights = {i: total / (num_classes * counter[i]) for i in counter}
    return class_weights

# ----------------------------
# Model building (binary)
# ----------------------------

def build_binary_model(input_shape=(256, 256, 3)):
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
    Run inference on the dataset and return true labels, binary predicted labels, and prediction probabilities.
    """
    y_true = []
    y_pred = []
    y_probs = []
    for images, labels in dataset:
        preds = model.predict(images)
        y_probs.append(preds)
        y_true.extend(labels.numpy())
        y_pred.extend((preds >= 0.5).astype(int).flatten())
    y_probs = np.concatenate(y_probs, axis=0).flatten()
    return np.array(y_true), np.array(y_pred), y_probs

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
# External Validation Scenario (Binary)
# ----------------------------

def run_external_validation_scenario_binary(scenario_name, train_dirs, val_dirs, task, batch_size=BATCH_SIZE, epochs=EPOCHS):
    # Minimal terminal output.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join("results", f"{scenario_name}_{task}_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)

    # Get binary file paths and labels for train and validation sets
    train_files, train_labels = get_binary_file_paths_and_labels(train_dirs, task)
    val_files, val_labels = get_binary_file_paths_and_labels(val_dirs, task)

    train_ds = create_dataset(train_files, train_labels, batch_size, augment=True, shuffle=True)
    val_ds = create_dataset(val_files, val_labels, batch_size, augment=False, shuffle=False)

    # Use preset class weights based on task.
    if task == "normal_ffkcn_vs_kcn":
        class_weights = CLASS_WEIGHT_NORMAL_FFKCN_VS_KCN
    elif task == "normal_vs_ffkcn":
        class_weights = CLASS_WEIGHT_NORMAL_VS_FFKCN

    model = build_binary_model(input_shape=(256, 256, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, epochs=epochs, class_weight=class_weights, verbose=0)

    y_true, y_pred, y_probs = evaluate_model(model, val_ds)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    r2 = metrics.r2_score(y_true, y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred)

    metrics_dict = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "R2 Score": r2,
        "Confusion Matrix": cm.tolist()
    }
    save_metrics(metrics_dict, os.path.join(result_dir, "metrics.txt"))

    plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES[task],
                          normalize=False, title="Confusion Matrix",
                          save_path=os.path.join(result_dir, "confusion_matrix.pdf"))
    plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES[task],
                          normalize=True, title="Normalized Confusion Matrix",
                          save_path=os.path.join(result_dir, "confusion_matrix_normalized.pdf"))
    plot_roc_curve(y_true, y_probs, save_path=os.path.join(result_dir, "roc_curve.pdf"))
    plot_precision_recall_curve(y_true, y_probs, save_path=os.path.join(result_dir, "precision_recall_curve.pdf"))

    model.save(os.path.join(result_dir, "model.h5"))
    np.savez(os.path.join(result_dir, "evaluation_data.npz"), y_true=y_true, y_pred=y_pred, y_probs=y_probs)

# ----------------------------
# K-Fold Cross Validation (Binary)
# ----------------------------

def run_kfold_cross_validation_binary(merged_dirs, task, k=5, batch_size=BATCH_SIZE, epochs=EPOCHS):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_result_dir = os.path.join("results", f"kfold_{task}_{timestamp}")
    os.makedirs(base_result_dir, exist_ok=True)

    # Load all data for merged directories (binary filtering)
    file_paths, labels = get_binary_file_paths_and_labels(merged_dirs, task)
    file_paths = np.array(file_paths)
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    y_true_all = []
    y_probs_all = []
    fold_idx = 1

    for train_index, val_index in skf.split(file_paths, labels):
        fold_result_dir = os.path.join(base_result_dir, f"fold_{fold_idx}")
        os.makedirs(fold_result_dir, exist_ok=True)
        train_files, val_files = file_paths[train_index], file_paths[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        train_ds = create_dataset(train_files, train_labels, batch_size, augment=True, shuffle=True)
        val_ds = create_dataset(val_files, val_labels, batch_size, augment=False, shuffle=False)

        # Use preset class weights based on task.
        if task == "normal_ffkcn_vs_kcn":
            class_weights = CLASS_WEIGHT_NORMAL_FFKCN_VS_KCN
        elif task == "normal_vs_ffkcn":
            class_weights = CLASS_WEIGHT_NORMAL_VS_FFKCN

        model = build_binary_model(input_shape=(256, 256, 3))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, epochs=epochs, class_weight=class_weights, verbose=0)

        y_true, y_pred, y_probs = evaluate_model(model, val_ds)
        y_true_all.extend(y_true)
        y_probs_all.extend(y_probs)

        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, zero_division=0)
        f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
        r2 = metrics.r2_score(y_true, y_pred)
        cm = metrics.confusion_matrix(y_true, y_pred)

        metrics_dict = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "R2 Score": r2,
            "Confusion Matrix": cm.tolist()
        }
        save_metrics(metrics_dict, os.path.join(fold_result_dir, "metrics.txt"))

        plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES[task],
                              normalize=False, title="Confusion Matrix",
                              save_path=os.path.join(fold_result_dir, "confusion_matrix.pdf"))
        plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES[task],
                              normalize=True, title="Normalized Confusion Matrix",
                              save_path=os.path.join(fold_result_dir, "confusion_matrix_normalized.pdf"))
        plot_roc_curve(y_true, y_probs, save_path=os.path.join(fold_result_dir, "roc_curve.pdf"))
        plot_precision_recall_curve(y_true, y_probs, save_path=os.path.join(fold_result_dir, "precision_recall_curve.pdf"))

        model.save(os.path.join(fold_result_dir, "model.h5"))
        np.savez(os.path.join(fold_result_dir, "evaluation_data.npz"), y_true=y_true, y_pred=y_pred, y_probs=y_probs)
        fold_idx += 1

    # Aggregate overall results from all folds.
    y_true_all = np.array(y_true_all)
    y_probs_all = np.array(y_probs_all)
    y_pred_all = (y_probs_all >= 0.5).astype(int)
    accuracy = metrics.accuracy_score(y_true_all, y_pred_all)
    precision = metrics.precision_score(y_true_all, y_pred_all, zero_division=0)
    recall = metrics.recall_score(y_true_all, y_pred_all, zero_division=0)
    f1 = metrics.f1_score(y_true_all, y_pred_all, zero_division=0)
    r2 = metrics.r2_score(y_true_all, y_pred_all)
    cm = metrics.confusion_matrix(y_true_all, y_pred_all)
    avg_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "R2 Score": r2,
        "Confusion Matrix": cm.tolist()
    }
    avg_dir = os.path.join(base_result_dir, "average_results")
    os.makedirs(avg_dir, exist_ok=True)
    save_metrics(avg_metrics, os.path.join(avg_dir, "average_metrics.txt"))

    plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES[task],
                          normalize=False, title="Average Confusion Matrix",
                          save_path=os.path.join(avg_dir, "confusion_matrix.pdf"))
    plot_confusion_matrix(cm, classes=BINARY_CLASS_NAMES[task],
                          normalize=True, title="Average Normalized Confusion Matrix",
                          save_path=os.path.join(avg_dir, "confusion_matrix_normalized.pdf"))
    plot_roc_curve(y_true_all, y_probs_all, save_path=os.path.join(avg_dir, "roc_curve.pdf"))
    plot_precision_recall_curve(y_true_all, y_probs_all, save_path=os.path.join(avg_dir, "precision_recall_curve.pdf"))

# ----------------------------
# Main routine to run experiments
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
    
    # Define the two binary tasks.
    tasks = ["normal_ffkcn_vs_kcn", "normal_vs_ffkcn"]
    
    for task in tasks:
        # External Validation Scenarios for this task.
        for scenario_name, dirs in external_scenarios.items():
            run_external_validation_scenario_binary(scenario_name, dirs["train"], dirs["val"], task, batch_size=BATCH_SIZE, epochs=EPOCHS)
        # K-Fold Cross Validation (merged data from all datasets)
        merged_dirs = [dataset_dirs["H"], dataset_dirs["K"], dataset_dirs["M"]]
        run_kfold_cross_validation_binary(merged_dirs, task, k=5, batch_size=BATCH_SIZE, epochs=EPOCHS)

if __name__ == '__main__':
    main()
