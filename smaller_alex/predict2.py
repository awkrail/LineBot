# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
'''
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer import Variable
from chainer import serializers
'''
import numpy as np
import cv2
#import _pickle as cpickle
import os
#import random
from PIL import Image
import json

this_script_dir = os.path.dirname(os.path.abspath(__file__)) + '/'


'''
def load_decrease_pickle():
    dec_path = 'decrease_pickle'
    dict = {'0': {}, '1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {}, '9': {}}

    for directory in os.listdir(dec_path):
        if directory.find('.DS_Store') > -1:
            continue
        for image in os.listdir(dec_path + '/' + directory):
            if image.find('.DS_Store') > -1:
                continue
            np_pickle = np.load(dec_path + '/' + directory + '/' + image).reshape(3, 5625)
            r, g, b = np_pickle[0], np_pickle[1], np_pickle[2]
            rImg = np.asarray(np.float32(r) / 255.0).reshape(75, 75)
            gImg = np.asarray(np.float32(g) / 255.0).reshape(75, 75)
            bImg = np.asarray(np.float32(b) / 255.0).reshape(75, 75)
            all_ary = np.asarray([rImg, gImg, bImg])

            new_name = image.replace('.npy', '')
            dict[directory][new_name] = all_ary

    return dict
'''

def cutout_face(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(this_script_dir + 'haarcascade_frontalface_alt.xml')
    face = faceCascade.detectMultiScale(gray, 1.1, 3)
    if len(face) > 0:
        for rect in face:
            x = rect[0]
            y = rect[1]
            width = rect[2]
            height = rect[3]
            dst = img[y:y+height, x:x+width]
            fixed_dst = cv2.resize(dst, (75, 75))
            path = this_script_dir + 'sent_image/fixed_sent.jpg'
            cv2.imwrite(path, fixed_dst)
            return True
    else:
        return False


def img2numpy():
    '''
    この関数,もっと高速に手順を減らすことができそう
    :return: imgからnumpyへ変換して返す
    '''
    img = np.asarray(Image.open(this_script_dir + 'sent_image/fixed_sent.jpg').convert('RGB'), dtype=np.int8)
    r_img = []
    g_img = []
    b_img = []

    for i in range(75):
        for j in range(75):
            r_img.append(img[i][j][0])
            g_img.append(img[i][j][1])
            b_img.append(img[i][j][2])

    all_ary = r_img + g_img + b_img
    all_np = np.array(all_ary, dtype=np.float32).reshape(3, 5625)
    r, g, b = all_np[0], all_np[1], all_np[2]
    rImg = np.asarray(np.float32(r) / 255.0).reshape(75, 75)
    gImg = np.asarray(np.float32(g) / 255.0).reshape(75, 75)
    bImg = np.asarray(np.float32(b) / 255.0).reshape(75, 75)

    rgb = np.asarray([rImg, gImg, bImg]).reshape(1, 3, 75, 75)

    #ｇ

    return rgb

def get_similiar(vector, number):
    with open(this_script_dir + 'output_json/output_%d.json' % number, 'r') as f:
        dict = json.load(f)
        dict = json.loads(dict)

    min = 10000
    similiar_key = None

    for key, np_img in dict.items():
        rlt = np.linalg.norm(vector-np_img)
        if rlt < min:
            min = rlt
            similiar_key = key

    return similiar_key

def get_url(similiar_key):
    json_path = this_script_dir + 'result_json/'
    url = None

    if similiar_key[0] == 'r':
        with open(json_path + 'rakuten.json', 'r') as f:
            dicts = json.load(f)
            for dict in dicts:
                # print(dict['id'])
                if dict['file_id'] == similiar_key:
                    url = dict['url']
                    break
    else:
        with open(json_path + 'hotpepper.json', 'r') as f:
            dicts = json.load(f)
            for dict in dicts:
                if dict['id'] == int(similiar_key):
                    # TODO: 単純な加算は危険. 超えたらどうするの？
                    url = dict['url']
                    break

    return url


class AlexNet(chainer.Chain):

    input_size = 227

    def __init__(self):
        super(AlexNet, self).__init__(
            conv1 = L.Convolution2D(None, 96, 6, stride=3),
            conv2 = L.Convolution2D(None, 128, 3, pad=2),
            # conv3 = L.Convolution2D(None, 256, 3, pad=1),
            conv4 = L.Convolution2D(None, 128, 3, pad=1),
            fc6 = L.Linear(None, 6400),
            fc7 = L.Linear(None, 6400),
            fc8 = L.Linear(None, 1000),
            fc9 = L.Linear(None, 10)
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv1(x))), 3, stride=1)
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 3, stride=1)
        # h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = F.dropout(F.relu(self.fc8(h)))
        h = self.fc9(h)

        return h

model = AlexNet()
chainer.serializers.load_npz(this_script_dir + 'smaller_alex_not_Classifier.zip', model)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

def predict(image_path):
    if cutout_face(image_path):
        img = img2numpy()
        '''
        model = AlexNet()
        chainer.serializers.load_npz(this_script_dir + 'smaller_alex_not_Classifier.zip', model)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        '''

        # predict
        y = model(img)
        similiar_key = get_similiar(y.data, np.argmax(y.data))
        #print(similiar_key)
        url = get_url(similiar_key)
        return url

    else:
        return '顔画像が検知できませんでした'


if __name__ == '__main__':
    print predict(this_script_dir + 'sent_image/sent_image.jpg')

'''
# imageデータをnumpy(4次元テンソル)に変換して返す
img = img2numpy()
# モデルのロード
model = AlexNet()
chainer.serializers.load_npz('smaller_alex_not_Classifier', model)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

#predict
y = model(img)
s_f = F.softmax(y).data
print(np.argmax(s_f))
similiar_key = get_similiar(s_f, np.argmax(s_f))
print(similiar_key)
url = get_url(similiar_key)
print(url)
'''
