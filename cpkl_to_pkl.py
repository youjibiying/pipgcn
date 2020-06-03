import _pickle as cPickle
# import cPickle
import os
import pickle

# def load_cpkl(file):
#     f = open(file, 'r')
#     file_1 = f.read()
#     file_2 = bytes(file_1, 'ascii')
#     target_params = cPickle.loads(file_2, encoding='bytes')
#     return target_params
#
def save_obj(obj, name ):
    with open(name + '.cpkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

data_path='../data/PPIs/train.cpkl'

with open(data_path,'rb') as f:
    u=pickle._Unpickler(f)
    u.encoding='latin1'
    t=u.load()

data_list,data=cPickle.load(open(data_path))
data_new=[]
for d in data:
    tmp={}
    for key in d.keys():
        tmp[str(key,encoding='utf-8')]=d[key]
    data_new.append(tmp)
save_obj(data_new,'train')
train_list, train_data = cPickle.load(open(data_path))
import numpy as np

# Save
# dictionary = {'hello':'world'}
# np.save('my_file.npy', dictionary)
#
# # Load
# read_dictionary = np.load('my_file.npy').item()
# print(read_dictionary['hello']) # displays "world"
