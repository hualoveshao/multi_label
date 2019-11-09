#测试的时候还可以进行处理，数据大于2000条数据进行测试数据
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Activation
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, GlobalAveragePooling1D, Bidirectional, GRU, MaxPooling1D, AveragePooling1D,Concatenate,LSTM
from tensorflow.keras.layers import SpatialDropout1D, BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger 

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import os
import pandas as pd

max_len = 50000
#初始化部分数据

def load_from_disk(path_to_disk):
    return pickle.load(open(path_to_disk, 'rb'))

#获取模型
# custom_objects = get_custom_objects()
models = ["fold1model_13-0.32.hdf5","fold3model_15-0.32.hdf5","fold4model_22-0.34.hdf5"]
models_list = list()
for model_name in models:
    models_list.append(tf.keras.models.load_model("model/"+model_name))

#产生数据
file_names,files,labels = load_from_disk("./data/test_id_data100")

x_test = list()
for s_text in tqdm(files):
    s_text = list(np.array(s_text)+1)
    s_text = s_text[:max_len]
    s_text = list(np.concatenate([s_text, [0] * (max_len - len(s_text))]))
    x_test.append(s_text)

test_result = np.zeros(shape = (len(x_test), 8))

x_test_len = len(x_test)
batch_size = 8
epochs = int(x_test_len/batch_size)+1
sub_index = 0
for index in range(epochs):
    start = index*batch_size
    end = (index+1)*batch_size
    end = min(end,x_test_len)
    print(start,end)
    sing_x_test = np.array(x_test[start:end])
    tmp_sub_index = 0
    for model in models_list:
        result = model.predict(sing_x_test)
        tmp_sub_index = sub_index
        for item in result:
            test_result[tmp_sub_index] += item
            tmp_sub_index+=1
    sub_index = tmp_sub_index
    del sing_x_test
    
test_result/=(len(models))

#file_id,prob0,prob1,prob2,prob3,prob4,prob5,prob6,prob7
submission = pd.DataFrame.from_dict({
'file_id': file_names,
'prob0': list(test_result[:,0]),
'prob1': list(test_result[:,1]),
'prob2': list(test_result[:,2]),
'prob3': list(test_result[:,3]),
'prob4': list(test_result[:,4]),
'prob5': list(test_result[:,5]),
'prob6': list(test_result[:,6]),
'prob7': list(test_result[:,7]),}
)
submission.to_csv('5_100cnn_security_submit.csv', index=False)