import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(y_true, y_pred_probas, filename, save_directory):

    """
    Plot the ROC curve and save it to a file.
    Parameters:
    y_true : array-like, shape (n_samples,)
        True binary labels or binary label indicators.
    y_pred_probas : array-like, shape (n_samples,)
        Probability estimates of the positive class.
    name : str
        Name of the file to save the plot.
    save_directory : str
        Directory where the plot will be saved.
    """

    valid_indices = y_true.notnull() & y_pred_probas.notnull()
    y_true, y_pred_probas = y_true[valid_indices], y_pred_probas[valid_indices]

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probas)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=3, label='AUC = %0.2f' % roc_auc, color='orange')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xticks([i / 5 for i in range(6)])
    plt.yticks([i / 5 for i in range(6)])
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_directory + filename + '.png', dpi=300, bbox_inches='tight')
    plt.show()