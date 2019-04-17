def plot_confusion_matrix(cm,
                          target_names,
                          normalize = True):
    """
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize = (8, 6))
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title('Confusion matrix')
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation = 45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment = "center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment = "center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nAccuracy = {:0.4f}'.format(accuracy))
    plt.show()
    
def get_confusion_matrix(predictor, x_test, y_test):
    '''
    Helper method for plot_confusion_matrix
    
    Inputs:
        predictor: an sklearn predictor that has been fit
        x_test: the vector of features that predictor was trained on
        y_test: the labels from the holdout set
        labels: a vector containing the categories
    '''
    from sklearn.metrics import confusion_matrix
    y_pred = predictor.predict(x_test)
    labels = list(set(y_test))
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels)