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
        plt.xticks(tick_marks, target_names, rotation = 90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment = "center",
                     color = "white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment = "center",
                     color = "white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nAccuracy = {:0.3f}'.format(accuracy))
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
    
def show_labeled_figure(x, y, labels, i, show_colorbar = True):
    import matplotlib.pyplot as plt
    
    plt.figure();
    plt.imshow(x[i]);
    if show_colorbar:
        plt.colorbar();
    plt.grid(False);
    plt.xlabel(labels[y[i]]);
    plt.xticks([])
    plt.yticks([])
    plt.show();
    
def show_predicted_image(i, predictions, true_label, img, label_names):
    """
    Adapted from: https://www.tensorflow.org/tutorials/keras/basic_classification
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    predictions, true_label, img = predictions[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(label_names[predicted_label],
                                100*np.max(predictions),
                                label_names[true_label]),
                                color=color)
    
def plot_predictions(i, predictions_array, true_label, label_names):
    """
    Adapted from: https://www.tensorflow.org/tutorials/keras/basic_classification
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
#     plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)

    plt.xticks(range(10), label_names, rotation = 90)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def show_image_with_predictions(i, predictions, true_label, img, label_names):
    """
    Adapted from: https://www.tensorflow.org/tutorials/keras/basic_classification
    """
    import matplotlib.pyplot as plt
    from gml_utils.visualization import plot_predictions, show_predicted_image
    import numpy as np
    
    plt.figure(figsize = (10,5))
    plt.subplot(1, 2, 1)
    show_predicted_image(i, predictions, true_label, img, label_names)
    plt.subplot(1, 2, 2)
    plot_predictions(i, predictions, true_label, label_names)
    plt.show()



def plot_history(history):
    """
    Adapted from: https://realpython.com/python-keras-text-classification/
    """
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label = 'Training acc')
    plt.plot(x, val_acc, 'r', label = 'Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label = 'Training loss')
    plt.plot(x, val_loss, 'r', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
