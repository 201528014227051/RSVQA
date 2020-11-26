import dill
import numpy as np 
import os
try:
    import cPickle as pkl
except:
    import pickle as pkl 
import h5py
import json 
import random
from sklearn.svm import SVC
from tqdm import tqdm
from keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, GlobalAveragePooling2D, Concatenate, GlobalMaxPooling2D, Conv2D, Flatten, Lambda, Reshape, Conv1D, MaxPooling1D, TimeDistributed,AveragePooling2D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.merge import Add, Multiply
from keras.regularizers import l2
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import plot_model
from keras.preprocessing import image 
from keras.applications.inception_v3 import conv2d_bn
from keras import backend as K  
from keras import optimizers
from keras.models import load_model
#from main_inception_vgg16 import dense_softmax_catten_model
#from tucker_layer import TuckerLyaer
#from contextfusion_layer import ContextFusionLyaer
#from mixfusion_layer import MixFusionLyaer
import tensorflow as tf 
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
K.set_learning_phase(0)
batch_size = 60#360#240#120
learning_rate = 0.0001
epoch_nums = 10
word_feature_dim = 1024#300

USE_GRU = True
USE_IMG_SEMANTIC = True
USE_QUE_SEMANTIC = True 
USE_BIFUSION = True
USE_Inception = False

model_path = './main_inception_vgg16_woinception_lr0.0005/'
#model_path = './softmax_multi_models_categorical_crossentropy_bs360_bert_optimer_cmp_xariver_100/'
#model_path = './softmax_multi_models_categorical_crossentropy_bs120_bert_optimer_cmp_xariver_100_atten/'
#model_path = './softmax_multi_models_categorical_crossentropy_bs120_bert_optimer_cmp_xariver_100_atten_contextfusion/'
#model_path = './main_only_ques_atten_categorical_crossentropy_bs60_bert_optimer_cmp_xariver_100_atten_bilinear/'
#model_path = './main_inception_vgg16_categorical_crossentropy_bs60_bert_optimer_cmp_xariver_100_atten_bilinear/'
#model_path = './main_inception_vgg16_categorical_crossentropy_bs60_bert_optimer_cmp_xariver_100_atten_bilinear_lr0.0005/'
#he_normal

name_pkl = './fast_load_data_python3.pkl'
dill.load_session(name_pkl)

dic_file = './answer_dic.pkl'
with open(dic_file, 'rb') as f:
    answer2id_dic = pkl.load(f)

id2answer_dic = {value:key for key, value in answer2id_dic.items()}

dic_file = './questions_word_dic.pkl'
with open(dic_file, 'rb') as f:
    questions_word_2id_dic = pkl.load(f)

dic_file = './questions_dic.pkl'
with open(dic_file, 'rb') as f:
    questions2id_dic = pkl.load(f)

id2question_dic = {value:key for key, value in questions2id_dic.items()}

dic_file = './imid2path_dic.pkl'
if not os.path.exists(dic_file):
    im_path_list = []
    img_fea_file = h5py.File('./image_feature.h5','r')
    def printname(name):
        im_path_list.append(name)
        #print(name)
    img_fea_file.visit(printname)
    # for key in img_fea_file:
    #     im_path_list.append(key)
    #     print(key)
    img_fea_file.close()
    im_path_list_real = []
    for path in im_path_list:
        print(path)
        try:
            if path.split('.')[1] == 'tif':
                im_path_list_real.append(path)
            elif path.split('.')[1] == 'jpg':
                im_path_list_real.append(path)
            elif path.split('.')[1] == 'png':
                im_path_list_real.append(path)
            print(path)
        except:
            pass
    imid2path_dic = {i:path for i, path in enumerate(im_path_list_real)}
    with open(dic_file, 'wb') as f:
        pkl.dump(imid2path_dic, f)
else:
    imid2path_dic = pkl.load(open(dic_file, 'rb'))

path2imid_dic = {value:key for key, value in imid2path_dic.items()}

print('The number of images:%d'%len(path2imid_dic))

    # ucm_theme_qap = load_text(path_ucm_theme)
    # sydney_theme_qap = load_text(path_sydney_theme)
    # rsicd_theme_qap = load_text(path_rsicd_theme)
# three for theme
print('load information of datasets done\n')


answer_key = [key for key, value in answer2id_dic.items()]
dic_anskey_list = {}
for key in answer_key:
    dic_anskey_list[key] = []

path_ucm = '/home/user2/qubo_captions/data/UCM/imgs/'

# construct bow for the questions in dataset
# id2question_dic  questions_word_2id_dic
print('construct glove for the questions in dataset\n')
# get the longset question
num_ques = len(id2question_dic)
longset_question = 0
for i in range(num_ques):
    cur_question = id2question_dic[i]
    word_list = cur_question.strip().split(' ')
    if len(word_list) > longset_question:
        longset_question = len(word_list)
print('The longest question length:%d'%longset_question)

line_id2features = {}
GLOVE_DIR = './output_questions.json'
embeddings_index = {}
for line in open(GLOVE_DIR, 'r'):
    tmp = json.loads(line)
    line_id2features[tmp['linex_index']] = tmp['features']

id2que_glove = {}
num_words = longset_question*word_feature_dim # for a uniform interface, be the feature dimension of the question
num_ques = len(id2question_dic)
for i in range(num_ques):
    cur_que = id2question_dic[i]
    print(cur_que)
    cur_question = line_id2features[i]
    glove = np.zeros((1, longset_question*word_feature_dim))
    ques_len_per = len(cur_question)-1# remove the sep token 
    for id_q in range(1,ques_len_per):
        word_tmp = line_id2features[i][id_q]
        fea_tmp = word_tmp['layers'][0]['values']# last layer's feature 
        fea_arr = np.asarray(fea_tmp)
# the glove is only a token to represent the word feature, it could be actually is a bert representation!
        #print(glove[0, (id_q-1)*word_feature_dim:((id_q-1)*word_feature_dim+word_feature_dim)].shape)
        glove[0, (id_q-1)*word_feature_dim:((id_q-1)*word_feature_dim+word_feature_dim)] = fea_arr
    id2que_glove[i] = glove

print('load data from txt..\n')
train_list = []
txt_train = open('./train_list.txt','r')
while True:
    line = txt_train.readline()
    if not line: break
    id_list = line.strip().split(':')
    train_list.append(id_list)
txt_train.close()

val_list = []
txt_val = open('./val_list.txt','r')
while True:
    line = txt_val.readline()
    if not line: break
    id_list = line.strip().split(':')
    val_list.append(id_list)
txt_val.close()

test_list = []
txt_test = open('./test_list.txt','r')
while True:
    line = txt_test.readline()
    if not line: break
    id_list = line.strip().split(':')
    test_list.append(id_list)
txt_test.close()
print('construct model.')
print('construct model.')
channel_axis = 3
def inception(x):
    branch1x1 = conv2d_bn(x, 320, 1, 1)

    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
    branch3x3 = layers.concatenate(
        [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9')

    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = layers.concatenate(
        [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

    branch_pool = AveragePooling2D(
        (3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed')
    return x


def dense_softmax_catten_model(fea_dim, class_num, USE_GRU = True, USE_IMG_SEMANTIC = True, USE_QUE_SEMANTIC = True, USE_BIFUSION = True, USE_Inception = True, regularizer=1e-8):
    base_model = VGG16(weights='imagenet')
    #input_img = base_model.input
    number_token = int(fea_dim/1024)
    input_fea = Input(shape=(number_token, 1024), name='fea_input')#batch_shape=(batch_size
    input_fea2 = TimeDistributed(Dense(units=512, kernel_regularizer=l2(regularizer), activation ='linear'), name = 'text_embedding')(input_fea)
    if USE_GRU == True:
        input_fea2 = GRU(units=512, recurrent_regularizer=l2(regularizer),kernel_regularizer=l2(regularizer), 
                    bias_regularizer=l2(regularizer), return_sequences=True, name='text_gru')(input_fea2)
        GetLastState = Lambda(lambda x:x[:,-1,:])
        input_fea2_single = GetLastState(input_fea2)
    else:
        input_fea2_single = Reshape((number_token*512,), name='reshape_for_no_gru')(input_fea2)
        input_fea2_single = Dense(units=512, kernel_regularizer=l2(regularizer), activation ='relu', name='dense_for_no_gru')(input_fea2_single)

    fc_output = base_model.get_layer('predictions').output
    block5_conv3 = base_model.get_layer('block5_conv3').output #14*14*512
    if USE_Inception == True:
        block5_conv3 = inception(block5_conv3)
        block5_conv3 = conv2d_bn(block5_conv3, 512, 3,3,name='dim_change')
    block5_conv3 = Reshape((196,512))(block5_conv3)

    #flat_out = Flatten()(block5_conv3)#512
    fc_output = Dense(units = 512, kernel_regularizer=l2(regularizer), activation ='linear', name='dim_reduc')(fc_output)
    #fc_output = Reshape((1, 512))(fc_output)
#kernel_initializer = 'he_normal',
    #concat_img = Concatenate(axis=1, name='concat_img')([fc_output, block5_conv3])

    def quesattn2imgs(x):
        q, im = x[0], x[1]
        print(K.int_shape(q))
        print(K.int_shape(im))
        w = K.batch_dot(q, im, [1,2])
        w = K.softmax(w)
        o = K.batch_dot(w, im, [1,1])
        return o 
    def quesattn2imgs_out_shape(input_shape):
        return input_shape[0]
    attn_layer = Lambda(quesattn2imgs, output_shape=quesattn2imgs_out_shape)
    attn_layer2 = Lambda(quesattn2imgs, output_shape=quesattn2imgs_out_shape)
    img_context = attn_layer([input_fea2_single, block5_conv3])
    que_contenxt = attn_layer2([fc_output, input_fea2])
    if USE_IMG_SEMANTIC == True:
        concat_img = Concatenate(axis=1, name='concat_img')([fc_output, img_context])
    else:
        concat_img = img_context
    if USE_QUE_SEMANTIC == True:
        concat_que = Concatenate(axis=1, name='concat_que')([input_fea2_single, que_contenxt])
    else:
        concat_que = que_contenxt
    def bilinear_layer(x):
        img, que = x[0], x[1]
        dimq = K.int_shape(que)[1]
        dimi = K.int_shape(img)[1]
        img_pre = tf.expand_dims(img, 1)
        img_pre = tf.tile(img_pre, tf.stack([1,dimq,1]))
        que_pre = tf.expand_dims(que, 2)
        que_pre = tf.tile(que_pre, tf.stack([1,1,dimi]))
        o = tf.multiply(img_pre, que_pre)
        o = tf.reshape(o, [-1, dimq*dimi])
        return o
    def bilinear_layer_shape(input_shape):
        x=input_shape[0][1]
        y=input_shape[1][1]
        return tuple([input_shape[0][0], x*y])
    BilinearLyaer = Lambda(bilinear_layer, output_shape=bilinear_layer_shape)
    if USE_BIFUSION == True:
        concat2 = BilinearLyaer([concat_img, concat_que])
    else:
        concat2 = Concatenate(axis=1, name='concat_img_and_que')([concat_img, concat_que])
    ouput = Dense(units=class_num, kernel_regularizer=l2(regularizer), activation='softmax', name='dense2')(concat2)
    model = Model(inputs=[base_model.input, input_fea], outputs=ouput)
    return model 
#construct model
fea_dim= num_words
class_num=len(id2answer_dic)
print('number of words:%d'%num_words)
print('number of class:%d'%class_num)
dense_model = dense_softmax_catten_model(fea_dim=fea_dim, class_num=class_num, USE_GRU = USE_GRU, USE_IMG_SEMANTIC = USE_IMG_SEMANTIC, USE_QUE_SEMANTIC = USE_QUE_SEMANTIC, USE_BIFUSION = USE_BIFUSION, USE_Inception = USE_Inception)
#'categorical_crossentropy'
print('get the image content from img paths...')
img_leng = 224*224*3
path_train_h5 = './img_rgb_optimer.pkl'
if not os.path.exists(path_train_h5):
    img_id2rgb = {}
    for v_id in range(len(imid2path_dic)):
        img = image.load_img('/'+imid2path_dic[v_id], target_size=(224,224))
        img = image.img_to_array(img)
        img_con = img.reshape(img_leng, 1)
        img_id2rgb[v_id] = img_con[:,0]
    #train_content_h5 = h5py.File(path_train_h5, 'w')
    #train_content_h5['data'] = img_id2rgb
    train_content_h5 = open(path_train_h5, 'wb+')
    pkl.dump(img_id2rgb, train_content_h5)
    train_content_h5.close()
else:
    with open(path_train_h5, 'rb') as f:
        img_id2rgb = pkl.load(f)
    #train_content_h5 = h5py.File(path_train_h5, 'r')
    #img_id2rgb = train_content_h5['data'][:]
    #train_content_h5.close()
print('get true data using id ...\n')
# not true , the id is more suitable
def from_list_to_array(train_list, img_leng, id2answer_dic): #id2que_glove  to get the question representation
    train_vqa_data = np.zeros((len(train_list), 2))
    train_vqa_y = np.zeros((len(train_list), len(id2answer_dic)))
    i = 0
    for vqa in tqdm(train_list):
        v_id = int(vqa[0])
        q_id = int(vqa[1])
        a_id = int(vqa[2])
        train_vqa_data[i, 0] = v_id
        train_vqa_data[i, 1] = q_id
        train_vqa_y[i, a_id] = 1 
        i = i+1 
    return train_vqa_data, train_vqa_y

#train_vqa_data, train_vqa_y = from_list_to_array(train_list, img_leng, id2answer_dic)
#val_vqa_data, val_vqa_y = from_list_to_array(val_list, img_leng, id2answer_dic)
test_vqa_data, test_vqa_y = from_list_to_array(test_list, img_leng, id2answer_dic)

type_list = []
yn_id_list = []
num_id_list = []
other_id_list = []
i = 0
for vqa in tqdm(test_list):
    v_id = int(vqa[0])
    q_id = int(vqa[1])
    question = id2question_dic[q_id]
    content = question.strip().split(' ')
    first_w = content[0]
    if first_w == 'does':
        type_list.append('yn')
        yn_id_list.append(i)
    elif first_w == 'how':
        type_list.append('number')
        num_id_list.append(i)
    else:
        type_list.append('other')
        other_id_list.append(i)
    i = i+1
assert(len(yn_id_list)+len(num_id_list)+len(other_id_list)==len(test_list))

# random
print('shuffle...')
#rand_id = [i for i in range(len(train_vqa_data))]
#random.shuffle(rand_id)
#train_vqa_data = train_vqa_data[rand_id]
#train_vqa_y = train_vqa_y[rand_id]

#rand_id = [i for i in range(len(val_vqa_data))]
#random.shuffle(rand_id)
#val_vqa_data = val_vqa_data[rand_id]
#val_vqa_y = val_vqa_y[rand_id]

def get_rgb_or_que(data, img_id2rgb, begin, end):
    id_array = data[begin:end]
    list_tmp = []
    for i in range(len(id_array)):
        list_tmp.append(img_id2rgb[id_array[i]])
    return np.squeeze(np.asarray(list_tmp))

def data_generator(data, y, batch_size, id2que_glove, img_id2rgb, num_words):
    steps = int(len(data)/batch_size)
    num, dimy = y.shape
    img_len = 224*224*3 
    dim2 = num_words 
    batch_img = np.zeros((batch_size, 224, 224, 3))
    batch_data = np.zeros((batch_size, dim2))
    batch_y = np.zeros((batch_size, dimy))
    while True:
        for i in range(steps):
            batch_img_content = get_rgb_or_que(data[:,0], img_id2rgb, i*batch_size, (i+1)*batch_size)
            batch_img = batch_img_content.reshape(batch_size, 224, 224, 3)
            batch_img = preprocess_input(batch_img)
            batch_data = get_rgb_or_que(data[:,1], id2que_glove, i*batch_size, (i+1)*batch_size)
            batch_y = y[i*batch_size:(i+1)*batch_size]
            yield [{'input_1':batch_img,'fea_input':batch_data}, {'dense2':batch_y}] 
            batch_data = np.zeros((batch_size, dim2))
            batch_y = np.zeros((batch_size, dimy))

training_history_filename = './training_history.log'
csv_logger = CSVLogger(training_history_filename, append=False)


print('predicting...')
def prepare_list_for_test(test_vqa_data, id2que_glove, img_id2rgb, num_words):
    num, dim = test_vqa_data.shape 
    dim2 = 224*224*3
    img_content = np.zeros((num, 224, 224, 3))
    bow_content = np.zeros((num, num_words))
    img_con = get_rgb_or_que(test_vqa_data[:,0], img_id2rgb, 0, num)#test_vqa_data[:, 0:dim2]
    img_content = img_con.reshape(num, 224,224,3)
    bow_content = get_rgb_or_que(test_vqa_data[:,1], id2que_glove, 0, num)
    #bow_content = bow_content.reshape(num, 1024, int(num_words/1024))
    #bow_content = np.transpose(bow_content,(0,2,1))
    bow_content = bow_content.reshape(num, int(num_words/1024), 1024)
    return [img_content, bow_content]
# test for ten epoch
print(model_path)
model_all = os.listdir(model_path)
hdf5_list = model_all
#hdf5_list = [i for i in model_all if i[15]=='0']
for file in hdf5_list:
    print(file)
    hdf5_file = os.path.join(model_path, file)
    #dense_model = load_model(hdf5_file, custom_objects={'focal_loss_fixed':focal_loss_fixed})
    #dense_model = load_model(hdf5_file)
    dense_model.load_weights(hdf5_file)
    res = dense_model.predict(prepare_list_for_test(test_vqa_data,id2que_glove, img_id2rgb, num_words))
    right = 0
    for i in range(len(test_vqa_y)):
        if np.argmax(test_vqa_y[i]) == np.argmax(res[i]):
            right = right + 1
    acc = float(right)/len(test_vqa_y)
    print('acc overall: %f'%acc)

    yn_data = test_vqa_data[yn_id_list]
    yn_y = test_vqa_y[yn_id_list]
    res = dense_model.predict(prepare_list_for_test(yn_data, id2que_glove, img_id2rgb, num_words))
    right = 0
    for i in range(len(yn_y)):
        if np.argmax(yn_y[i]) == np.argmax(res[i]):
            right = right + 1
    acc = float(right)/len(yn_y)
    print('acc of yes/no: %f'%acc)

    num_data = test_vqa_data[num_id_list]
    num_y = test_vqa_y[num_id_list]
    res = dense_model.predict(prepare_list_for_test(num_data, id2que_glove, img_id2rgb, num_words))
    right = 0
    for i in range(len(num_y)):
        if np.argmax(num_y[i]) == np.argmax(res[i]):
            right = right + 1
    acc = float(right)/len(num_y)
    print('acc of number: %f'%acc)

    other_data = test_vqa_data[other_id_list]
    other_y = test_vqa_y[other_id_list]
    res = dense_model.predict(prepare_list_for_test(other_data, id2que_glove, img_id2rgb, num_words))
    right = 0
    for i in range(len(other_y)):
        if np.argmax(other_y[i]) == np.argmax(res[i]):
            right = right + 1
    acc = float(right)/len(other_y)
    print('acc of others: %f'%acc)
