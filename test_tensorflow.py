import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf

import main_tensorflow as mf


model_path = 'my_test_model'
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


def load_and_test_model(x_test, y_test):
    _graph = tf.Graph()
    with tf.Session(graph=_graph) as sess:
        model = tf.train.import_meta_graph(f'{model_path}.meta')
        model.restore(sess, model_path)
        _accuracy = _graph.get_tensor_by_name('accuracy:0')
        _x = _graph.get_tensor_by_name('X:0')
        _y = _graph.get_tensor_by_name('Y:0')
        _cost = _graph.get_tensor_by_name('cost:0')
        _y_pred = _graph.get_tensor_by_name('prediction:0')
        _keep_prob = _graph.get_tensor_by_name('keep:0')

        # load raw test data to generate metrics
        raw_y_test_data = mf.load_raw_y_test_data()

        hot_pred, testing_accuracy, testing_loss = sess.run([_y_pred, _accuracy, _cost],
                                                            feed_dict={_x: np.reshape(x_test, [len(x_test), 128, 1, 9]),
                                                                       _y: np.reshape(y_test, [len(y_test), 6]),
                                                                       _keep_prob: 1
                                                                       })
        show_metrics(hot_pred, testing_accuracy, testing_loss, raw_y_test_data)


def show_metrics(prediction, testing_accuracy, testing_loss, y_test):
    pred = prediction.argmax(1)
    print(f'Testing accuracy : {100 * testing_accuracy}%')
    print(f'Testing loss: {testing_loss}')
    print(f'Precission: {100 * metrics.precision_score(y_test, pred, average="weighted")}%')
    print(f'Recall: {100 * metrics.recall_score(y_test, pred, average="weighted")}%')
    print(f'f1_score: {100 * metrics.f1_score(y_test, pred, average="weighted")}%')
    print('Confusion matrix:')
    confusion_matrix = metrics.confusion_matrix(y_test, pred)
    print(confusion_matrix)
    normalised_confusion_matric = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
    print("Confusion matrix (normalised to % of total test data): ")
    print(normalised_confusion_matric)

    plt.figure(figsize=(12, 12))
    plt.imshow(
        normalised_confusion_matric,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised ot % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = mf.load_data()
    x_test = (x_test - np.mean(x_test)) / np.std(x_test)
    load_and_test_model(x_test, y_test)
