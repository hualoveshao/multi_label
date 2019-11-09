from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Input, Dense, GRU, Dropout, BatchNormalization, \
                         MaxPooling1D, Conv1D, Flatten, Concatenate,Bidirectional,TimeDistributed,Add,GlobalMaxPooling1D,SpatialDropout1D,GlobalAveragePooling1D,Concatenate,Embedding
from keras.models import Model
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
import keras
import functools
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger,ReduceLROnPlateau,TensorBoard,Callback
from keras.layers import Layer
import pickle
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.utils import to_categorical

#配置GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" # 使用编号为1，2号的GPU
config = tf.ConfigProto()

config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


#初始化部分数据
epochs = 30
batch_size = 8
max_len = 12000
model_name = "cnn"

def load_from_disk(path_to_disk):
    return pickle.load(open(path_to_disk, 'rb'))

#模型获取
def get_model(weights,model_name="cnn",output_length=8,max_len=12000,max_cnt = 301,embed_size=64,mask_zero=False,trainable=True):
    if(model_name=="rnn"):
        return rnn(output_length, weights,max_cnt,embed_size,max_len,mask_zero,trainable)
    elif(model_name=="cnn"):
        return cnn(output_length, weights,max_cnt,embed_size,max_len,mask_zero,trainable)
    else:
        return None

#定义模型
def rnn(output_length,weights,max_cn,embed_size,max_len,mask_zero,trainable):
    """ Create and return a keras model of a CNN """
    #这个也是可以改的 一会把
    HIDDEN_LAYER_SIZE = 128
    inputs = Input(shape=(max_len,), dtype='int32')
    embed = Embedding(max_cnt+1, embed_size, input_length=max_len, mask_zero=mask_zero, weights=[weights], trainable=trainable)(inputs)
    x = SpatialDropout1D(0.2)(inputs)
    gru = Bidirectional(GRU(
        HIDDEN_LAYER_SIZE,
        input_shape=(maxlen, embedding_size),
        kernel_initializer="glorot_uniform",
        recurrent_initializer='normal',
        return_sequences = True,
        activation='relu',  
    ))(x)
    avg_pool = GlobalAveragePooling1D()(batch_normalization)
    max_pool = GlobalMaxPooling1D()(batch_normalization)
    conc = Concatenate()([avg_pool, max_pool])
    outputs = Dense(output_length, activation='softmax')(conc)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=["accuracy"],
    )
    return model
   
weights = load_from_disk("./data/word2vec_weights")
b = np.zeros((1,64))
b_t = list()
b_t.append(b)
weights = np.insert(weights, 0, values=b_t, axis=0)

model = get_model(weights,modol_name,output_length,max_len)
#获取数据
#对数据使用0填充，这个是不存在的命令，词向量全是零
def seq_padding(X, padding=0):
    ML = max_len
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])

#产生数据
file_names,files,labels = load_from_disk("./data/train_id_data")

train_data_text = list()
train_data_label = list()
val_data_text = list()
val_data_label = list()

index = 0
for s_text,s_label in zip(files,labels):
    s_text = np.array(s_text)+1
    if(index%10!=0):
        train_data_text.append(s_text)
        train_data_label.append(s_label)
    else:
        val_data_text.append(s_text)
        val_data_label.append(s_label)
    index+=1

print(len(train_data_text),len(val_data_text))

train_data = list()
val_data = list()

for s_text,s_label in zip(train_data_text,train_data_label):
    s_label = to_categorical(s_label,8)
    train_data.append((s_text,s_label))

for s_text,s_label in zip(val_data_text,val_data_label):
    s_label = to_categorical(s_label,8)
    val_data.append((s_text,s_label))

class data_generator:
    def __init__(self, data, batch_size=8):
        self.data = data
        self.batch_size = batch_size
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X, Y = [], []
            for i in idxs:
                d = self.data[i]
                text = d[0]
                text = text[:max_len]#进行截断
                X.append(text)
                Y.append(d[1])
                
                if len(X) == self.batch_size or i == idxs[-1]:
                    X = seq_padding(X)
                    yield [X], [Y]
                    [X,Y] = [], []



#开始训练
save_dir = "checkpoint"
filepath="model_{epoch:02d}-{val_loss:.2f}.hdf5"

checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='val_loss', verbose=1, save_best_only=True)
csv_logger = CSVLogger(save_dir+'/training.log')

train_D = data_generator(train_data, batch_size = batch_size)
valid_D = data_generator(val_data, batch_size = batch_size)


model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=epochs,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[checkpoint,csv_logger],
        )