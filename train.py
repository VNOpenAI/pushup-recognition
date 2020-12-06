import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras 
from preprocessing import DataSequence
import model
import datetime
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
tf.random.set_seed(1)
np.random.seed(1)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#prepare data
train_data_class = DataSequence('./Resource/train.json',batch_size=16,aug=True)
validation_data_class = DataSequence('./Resource/val.json',batch_size=16,aug=False)

#define callback
checkpoint_filepath = './checkpoint1'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=True)    
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#create model and compile
my_model=model.create_model()
my_model.compile(loss='binary_crossentropy',
              #optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9),
              optimizer=keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy',precision_m,recall_m,f1_m])
class_weight={0: 0.2, 1: 0.8}

#train
history=my_model.fit(train_data_class,
                    epochs=45,
                    verbose=1,
                    validation_data=validation_data_class,
                    class_weight=class_weight,
                    callbacks=[model_checkpoint_callback,tensorboard_callback],
                    ) 
#plot trainning process
fig=plt.figure()
fig.add_axes([0.1,0.1,0.85,0.85])
fig.gca().plot(history.history['loss'],labels='train_loss')
fig.gca().plot(history.history['vasl_loss'],labels='val_loss')
fig.suptitle('loss')
fig.gca().legend()
plt.show()



#save model
my_model.save('model.h5')