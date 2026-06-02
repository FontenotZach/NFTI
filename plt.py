import matplotlib.pyplot as plt
import numpy as np

# Define the sub-criteria and performance metrics for XGBoost and Neural Network models
criteria = ['PRBC+', 'OR+', 'ICU+', 'IR+', 'Death+', 'Intubate+', 'NFTI']

# xgboost_metrics = {
#     'Accuracy': [97.56, 96.94, 94.04, 99.69, 97.90, 99.09, 90.10],
#     'Precision': [99.24, 99.78, 99.50, 99.97, 99.60, 99.98, 97.34],
#     'Recall': [95.84, 94.10, 88.55, 99.42, 96.18, 98.19, 82.48],
#     'F1 Score': [97.51, 96.86, 93.70, 99.69, 97.86, 99.08, 89.29],
#     'ROC AUC': [0.9946, 0.9887, 0.9722, 0.9997, 0.9956, 0.9983, 0.9470]
# }

# nn_metrics = {
#     'Accuracy': [91.88, 90.63, 86.41, 95.30, 91.41, 92.53, 82.70],
#     'Precision': [94.58, 95.60, 92.39, 96.33, 93.09, 95.48, 89.38],
#     'Recall': [88.82, 85.20, 79.40, 94.22, 89.46, 89.29, 74.24],
#     'F1 Score': [91.61, 90.10, 85.40, 95.26, 91.24, 92.28, 81.11],
#     'ROC AUC': [0.9733, 0.9587, 0.9323, 0.9898, 0.9730, 0.9765, 0.8991]
# }

xgboost_metrics = {
    'Accuracy': [98.08, 97.36, 94.64, 99.82, 98.59, 99.59, 91.61],
    'Precision': [99.16, 99.78, 98.75, 99.98, 99.52, 99.96, 97.15],
    'Recall': [96.97, 94.93, 90.44, 99.66, 97.66, 99.23, 85.75],
    'F1 Score': [98.05, 97.29, 94.41, 99.82, 98.58, 99.59, 91.10],
    'ROC AUC': [0.9969, 0.9904, 0.9777, 0.9998, 0.9975, 0.9989, 0.9567]
}

nn_metrics = {
    'Accuracy': [92.41, 89.80, 87.97, 95.94, 91.97, 92.40, 83.24],
    'Precision': [93.28, 94.29, 92.50, 96.26, 92.93, 93.95, 89.42],
    'Recall': [91.39, 84.76, 82.68, 95.61, 90.86, 90.62, 75.43],
    'F1 Score': [92.33, 89.27, 87.31, 95.93, 91.88, 92.26, 81.84],
    'ROC AUC': [0.9770, 0.9559, 0.9438, 0.9930, 0.9772, 0.9788, 0.9008]
}

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Number of bars per group (XGBoost and Neural Network)
bar_width = 0.35
index = np.arange(len(criteria))

# Create subplots for each metric
fig, axes = plt.subplots(2, 2, figsize=(14, 18))
axes = axes.flatten()

# Loop through the metrics and create grouped bar charts
# Loop through the metrics and create grouped bar charts
for i, metric in enumerate(metrics):
    ax = axes[i]
    
    # XGBoost bars
    ax.bar(index, xgboost_metrics[metric], bar_width, label='XGBoost')

    # Neural Network bars
    ax.bar(index + bar_width, nn_metrics[metric], bar_width, label='Neural Network')

    # Set chart details
    ax.set_ylabel(metric + ' (%)')
    ax.set_title(f'{metric} Comparison')
    
    # Set y-axis range
    ax.set_ylim(70, 100)

    # Set tick positions and labels, ensuring labels are centered
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(criteria, rotation=0, ha='center')  # One label per pair, centered

    # Add horizontal grid marks only
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)

    # Move legend to the bottom within the chart area without covering the bars
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, .14), ncol=2, framealpha=1)  # Legend inside chart, above X-axis

# Adjust layout to prevent overlap
plt.tight_layout(pad=5)
plt.show()