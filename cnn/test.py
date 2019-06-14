from keras.models import load_model
from keras import backend as K
from sklearn.metrics import roc_auc_score
import tensorflow as tf

def roc_auc_score_modified(y_true, y_pred):
    
    """
    Modified ROC AUC Scoring method used to train Neural Networks
    - To tackle problems where shuffled batches only contains 1 class
    - Return 0.5 when this happens, an underestimate of the actual metric
    """
    
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5

def auroc(y_true, y_pred):
    
    """
    Custom Metric for Keras Neural Network - Area Under ROC Curve
    """
    
    return tf.py_func(roc_auc_score_modified, (y_true, y_pred), tf.double)
    
model = load_model('fold1_best_weights.hdf5',{'auroc':auroc})