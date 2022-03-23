import seaborn as sns
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

classes = ('Plane', 'Car', 'Bird', 'Cat',
           'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

def model_prediction(test_images, model):
    pred = []
    for imgs in test_images:
        pred.append(model(imgs))
    return(pred)

# def model_prediction(test_images, model):
#      pred = model.predict(test_images)
#      pred = [np.argmax(i) for i in pred]
#      pred = np.array(pred)
#      return pred

def confusion_matrix_heatmap(labels, predictions):
    '''
    labels: clean labels of the images
    predictions: labels generated from classification models
    '''
    cm = tf.math.confusion_matrix(labels = labels, predictions = predictions).numpy()
    cm_norm = np.around(cm.astype('float')/cm.sum(axis = 1)[:, np.newaxis], decimals = 2)
    cm_df = pd.DataFrame(cm_norm, index = classes, columns = classes)

    figure = plt.figure(figsize = (10, 10))
    sns.heatmap(cm_df, annot = True, cmap = plt.cm.Reds)
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel('Predicted Label')
    return plt.show()

# import panda as pd
# def prediction_convertion(true_labels, predictions):
#     df = pd.DataFrame({"labels": true_labels, "preds": predictions})
#     df.assign(flag = lambda dataframe: dataframe[''])
#     df['flag'] = np.where(df.labels == df.preds, 1, 0)