from keras.callbacks import Callback
from dragonn.metrics import *
import warnings
warnings.filterwarnings('ignore')

class MetricsCallback(Callback):
        def __init__(self, train_data, validation_data):
            super().__init__()
            self.validation_data = validation_data
            self.train_data = train_data    
        def on_epoch_end(self, epoch, logs={}):            
            X_train = self.train_data[0]
            y_train = self.train_data[1]
            
            X_val = self.validation_data[0]
            y_val = self.validation_data[1]

            y_train_pred=self.model.predict(X_train)
            y_val_pred=self.model.predict(X_val)

            train_classification_result=ClassificationResult(y_train,y_train_pred)
            val_classification_result=ClassificationResult(y_val,y_val_pred)
            print("Training Data:") 
            print(train_classification_result)
            print("Validation Data:") 
            print(val_classification_result)
            
            
                                                                                    
                                                                                    
