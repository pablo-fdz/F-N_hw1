import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall(y_true, y_pred_probas, title, filename, save_directory):

    """
    Plot the precision-recall curve and save it to a file.
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

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probas)
    avprec = average_precision_score(y_true, y_pred_probas)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, lw=3, label='Av.Precision = %0.2f' % avprec, color='purple')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xticks([i / 5 for i in range(6)])
    plt.yticks([i / 5 for i in range(6)])
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.savefig(save_directory + filename + '.png', dpi=300, bbox_inches='tight')
    plt.show()