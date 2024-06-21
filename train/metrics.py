import numpy as np
from pythonUtils import log_output
from pythonUtils import color_palette
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn import manifold
import pandas as pd
import re
from itertools import cycle

class EvaluationMetrics:
    def __init__(self, conf_matrix):
        """
        Initialize with a confusion matrix
        :param conf_matrix: confusion matrix
        """
        self.conf_matrix = conf_matrix
        self.num_classes = conf_matrix.shape[0]
        assert self.num_classes == conf_matrix.shape[1] and self.num_classes != 0, 'Invalid confusion matrix shape'
        self._recompute_metrics()

    def __add__(self, other):
        assert self.conf_matrix.shape == other.conf_matrix.shape, "Confusion matrices must be of same shape"
        combined_matrix = self.conf_matrix + other.conf_matrix
        return EvaluationMetrics(combined_matrix)

    def _recompute_metrics(self):
        self._calculate_basic_metrics()
        self._calculate_accuracy()
        self._calculate_precision()
        self._calculate_recall()
        self._calculate_macro_f1()

    def _calculate_basic_metrics(self):
        self.diagonal = np.diag(self.conf_matrix)
        self.total_sum = self.conf_matrix.sum().astype(float)
        self.tp_tn = self.diagonal.sum()
        self.tp_tn_fp_fn = np.array([self.tp_tn + self.conf_matrix[i, :].sum() + self.conf_matrix[:, i].sum() - 2 * self.conf_matrix[i, i] for i in range(self.num_classes)])
        self.tp_fn = np.array([self.conf_matrix[i, :].sum() for i in range(self.num_classes)])
        self.tp_fp = np.array([self.conf_matrix[:, i].sum() for i in range(self.num_classes)])

    def _calculate_accuracy(self):
        self.class_accuracy = [self.tp_tn / self.tp_tn_fp_fn[i] for i in range(self.num_classes)]
        self.mean_accuracy = np.mean(self.class_accuracy)
        self.overall_accuracy = self.tp_tn / self.total_sum

    def _calculate_precision(self):
        self.class_precision = np.true_divide(self.diagonal, self.tp_fp + 1e-6)
        self.mean_precision = np.mean(self.class_precision)

    def _calculate_recall(self):
        self.class_recall = np.true_divide(self.diagonal, self.tp_fn + 1e-6)
        self.mean_recall = np.mean(self.class_recall)

    def _calculate_macro_f1(self):
        self.class_f1_scores = 2 * np.multiply(self.class_precision, self.class_recall) / (self.class_precision + self.class_recall + 1e-6)
        self.macro_f1_score = np.mean(self.class_f1_scores)

    def save_results(self, result_file):
        metrics = {
            "Mean Accuracy": self.mean_accuracy,
            "Class Accuracy": self.class_accuracy,
            "Mean Precision": self.mean_precision,
            "Class Precision": self.class_precision,
            "Mean Recall": self.mean_recall,
            "Class Recall": self.class_recall,
            "Macro F1 Score": self.macro_f1_score,
            "Class F1 Scores": self.class_f1_scores,
            "Overall Accuracy": self.overall_accuracy,
            "Confusion Matrix": self.conf_matrix
        }
        for key, value in metrics.items():
            log_output(f"{key}: {value}", _file=result_file)

def compute_roc_auc(predictions, true_labels, num_samples, num_classes):
    true_onehot = np.zeros_like(predictions)
    true_onehot[np.arange(true_labels.shape[0]), true_labels] = 1

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_onehot[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(true_onehot.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.mean([interp(all_fpr, fpr[i], tpr[i]) for i in range(num_classes)], axis=0)
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure()
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (area = {roc_auc["macro"]:.4f})', color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")

    macro_out = np.column_stack((fpr["macro"], tpr["macro"]))
    micro_out = np.column_stack((fpr["micro"], tpr["micro"]))
    return plt, macro_out, micro_out

def compute_precision_recall(predictions, true_labels, num_samples, num_classes):
    true_onehot = np.zeros_like(predictions)
    true_onehot[np.arange(true_labels.shape[0]), true_labels] = 1

    precision, recall, avg_precision = {}, {}, {}
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(true_onehot[:, i], predictions[:, i])
        avg_precision[i] = average_precision_score(true_onehot[:, i], predictions[:, i])

    precision["micro"], recall["micro"], _ = precision_recall_curve(true_onehot.ravel(), predictions.ravel())
    avg_precision["micro"] = average_precision_score(true_onehot, predictions, average="micro")

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Average precision score, micro-averaged: AP={avg_precision["micro"]:.2f}')

    pr_out = np.column_stack((recall["micro"], precision["micro"]))
    return plt, pr_out

def tsne_visualization(config, feature_dict, save_dir):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    tsne_results = tsne.fit_transform(feature_dict.get("tsne_features"))
    tsne_norm = (tsne_results - tsne_results.min(0)) / (tsne_results.max(0) - tsne_results.min(0) + 1e-9)

    markers = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H', '+', 'x', '|', '_']
    colors = [color_palette.get('bisque'), color_palette.get('lightgreen'), color_palette.get('slategray'), color_palette.get('cyan'), color_palette.get('blue'), color_palette.get('lime'), 'r', color_palette.get('violet'), 'm', color_palette.get('peru'), color_palette.get('olivedrab'), color_palette.get('hotpink'), color_palette.get('olive'), color_palette.get('sandybrown'), color_palette.get('pink'), color_palette.get('purple')]
    class_labels = config.dataset.get("class_names", [str(i) for i in range(config.classes)])

    font_properties = {'family': 'Times New Roman', 'weight': 'bold'}

    tsne_data = pd.DataFrame({'x': tsne_norm[:, 0], 'y': tsne_norm[:, 1], 'label': feature_dict.get("tsne_labels")})
    plt.figure(figsize=(10, 10))
    for idx in range(config.classes):
        x_vals = tsne_data[tsne_data['label'] == idx]['x']
        y_vals = tsne_data[tsne_data['label'] == idx]['y']
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        plt.scatter(x_vals, y_vals, s=100, marker=marker, c=color, edgecolors=color, alpha=0.65, label=class_labels[idx])

    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right', prop=font_properties)
    plt.savefig(os.path.join(save_dir, "tsne_visualization.png"), bbox_inches="tight")
    plt.close()

def extract_key_values(file_path, key):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    values = [float(re.search(f'{key}:\s(.+?)\%\s\|', line).group(1)) for line in lines]
    return sorted(values, reverse=True)

def calculate_statistics(metrics, top_k):
    top_metrics = metrics[:top_k]
    metrics_array = np.array(top_metrics)
    stats = {
        'max': metrics_array.max(),
        'min': metrics_array.min(),
        'mean': round(metrics_array.mean(), 2),
        'right_space': round(metrics_array.max() - metrics_array.mean(), 2),
        'left_space': round(metrics_array.min() - metrics_array.mean(), 2)
    }
    return stats

def generate_statistics(file_path, key, top_k):
    metrics = extract_key_values(file_path, key)
    return calculate_statistics(metrics, top_k)
