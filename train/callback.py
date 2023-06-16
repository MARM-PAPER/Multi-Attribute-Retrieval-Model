from sklearn.metrics import roc_auc_score
import time
from tensorflow.python.keras.callbacks import Callback
class roc_callback(Callback):
    def __init__(self, validation_data):
        super(Callback, self).__init__()
        self.x = validation_data[0]
        self.y = validation_data[1]
        self.step = 0
        self.log_freq = 10
        self.epoch = 1


    def on_train_begin(self, logs={}):
        self.step = 1
        self._start_time = time.time()
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        #y_pred = self.model.predict(self.x)
        #auc = roc_auc_score(self.y, y_pred)
        #print('%s - auc_test: %s' % (epoch,str(round(auc,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        self.step += 1
        return

    def on_batch_end(self, batch, logs={}):
        if self.step % self.log_freq == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            examples_per_sec = self.log_freq / duration
            y_pred = self.model.predict(self.x)
            auc = roc_auc_score(self.y, y_pred)
            print('Time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ', Step #:', self.step,
                  ', Examples per second:', examples_per_sec, ' test auc: ', str(round(auc,4)))
        return
