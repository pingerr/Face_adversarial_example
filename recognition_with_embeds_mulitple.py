#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import copy
import os
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.misc import imread, imresize

from nets.model_irse import IR_50, IR_152
from nets.insightface import Backbone, MobileFaceNet

model_root = './ckpt/'
model_path_map = {'IR_50': model_root + 'backbone_ir50_ms1m_epoch120.pth',
                  'IR_152': model_root + 'Backbone_IR_152_Epoch_112_Batch.pth',
                  'IR_SE_50': model_root + 'model_ir_se50.pth'}


def preprocess_for_model(image):
    image = (image - 127.5) / 128.0
    image = image.swapaxes(2, 3).swapaxes(1, 2)
    image = torch.from_numpy(image)
    return image


def load_images_with_names(input_dir, batch_size=1):
    images = []
    filenames = []
    personnames = []
    idx = 0

    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2personname = {dev.iloc[i]['ImageName']: dev.iloc[i]['PersonName'] for i in range(len(dev))}
    for filename in filename2personname.keys():
        image = imread(os.path.join(input_dir, filename)).astype(np.float32)
        images.append(image)
        filenames.append(filename)
        personnames.append(filename2personname[filename])
        idx += 1

        if idx == batch_size:
            images = np.array(images)
            yield images, filenames, personnames
            images = []
            filenames = []
            personnames = []
            idx = 0
    if idx > 0:
        images = np.array(images)
        yield images, filenames, personnames


def recognition_one(image_embed_1, image_embed_2, image_embed_3):

    pkl_path1 = './embeds_pkl/all_by_ir50.pkl'
    pkl_path2 = './embeds_pkl/all_by_ir152.pkl'
    pkl_path3 = './embeds_pkl/all_by_irse50.pkl'

    with open(pkl_path1, 'rb+') as f:
        label2classname, all_embeds_1 = pkl.load(f)
    f.close()
    with open(pkl_path2, 'rb+') as f:
        _, all_embeds_2 = pkl.load(f)
    f.close()
    with open(pkl_path3, 'rb+') as f:
        _, all_embeds_3 = pkl.load(f)
    f.close()
    idx2label = all_embeds_1[:, 0]
    all_embeds_1 = all_embeds_1[:, 1:]
    all_embeds_2 = all_embeds_2[:, 1:]
    all_embeds_3 = all_embeds_3[:, 1:]

    image_embed_1 = image_embed_1 / np.linalg.norm(image_embed_1, axis=1, keepdims=True)
    image_embed_2 = image_embed_2 / np.linalg.norm(image_embed_2, axis=1, keepdims=True)
    image_embed_3 = image_embed_3 / np.linalg.norm(image_embed_3, axis=1, keepdims=True)

    all_embeds1 = all_embeds_1 / np.linalg.norm(all_embeds_1, axis=1, keepdims=True)
    all_embeds2 = all_embeds_2 / np.linalg.norm(all_embeds_2, axis=1, keepdims=True)
    all_embeds3 = all_embeds_3 / np.linalg.norm(all_embeds_3, axis=1, keepdims=True)

    cos_distances1 = image_embed_1.dot(all_embeds1.T)  # (batch, 512).(512, 1w+) = (batch, 1w+)
    cos_distances2 = image_embed_2.dot(all_embeds2.T)
    cos_distances3 = image_embed_3.dot(all_embeds3.T)

    # print('cos_dist shape:', cos_distances.shape)
    cos_distances = (cos_distances1 * 1.2 + cos_distances2 * 0.3 + cos_distances3 * 0.7) / 2.2

    idx = np.argmax(cos_distances[0])
    label = idx2label[idx]
    # print(label)

    cos_dist = np.max(cos_distances[0])
    # print('cos_dist:', cos_dist)
    name = label2classname[label]
    # print('name:', name)

    cos_distances_ = copy.deepcopy(cos_distances)
    cos_distances_[0][idx] = np.min(cos_distances_[0])
    second_idx = np.argmax(cos_distances_[0])
    second_cos_dist = np.max(cos_distances_[0])

    second_label = idx2label[second_idx]
    second_name = label2classname[second_label]

    # second_embed = all_embeds[second_idx]

    # second_embed = second_embed / np.linalg.norm(second_embed, axis=0, keepdims=True)

    return label, name, cos_dist, second_label, second_name, second_cos_dist


if __name__ == '__main__':

    print('building...')

    pkl_path1 = './embeds_pkl/all_by_ir50.pkl'
    pkl_path2 = './embeds_pkl/all_by_ir152.pkl'
    pkl_path3 = './embeds_pkl/all_by_irse50.pkl'

    device = torch.device('cuda')

    # model 1
    model_ir50 = IR_50([112, 112])
    model_ir50.load_state_dict(torch.load(model_path_map['IR_50'], map_location='cuda'))
    model_ir50.eval().to(device).zero_grad()

    # model 2
    model_IR_152 = IR_152([112, 112])
    model_IR_152.load_state_dict(torch.load(model_path_map['IR_152'], map_location='cuda'))
    model_IR_152.eval().to(device).zero_grad()
    #
    # # model 3
    IR_SE_50 = Backbone(50, mode='ir_se')
    IR_SE_50.load_state_dict(torch.load(model_path_map['IR_SE_50'], map_location='cuda'))
    IR_SE_50.eval().to(device).zero_grad()



    num = 0
    similar_num = 0
    for images, _, personnames in load_images_with_names('./images/raw_images'):

        tensor = preprocess_for_model(images).detach().to(device)
        embedding1 = model_ir50(tensor)
        embedding2 = model_IR_152(tensor)
        embedding3 = IR_SE_50(tensor)

        num += 1
        print('===============================> 第 {} 张:'.format(num))

        embedding1 = embedding1.data.cpu().detach().numpy()
        embedding2 = embedding2.data.cpu().detach().numpy()
        embedding3 = embedding3.data.cpu().detach().numpy()

        idx, name, cos_dist, second_idx, second_name, second_cos_dist =\
            recognition_one(embedding1, embedding2, embedding3)

        if personnames[0] == name:
            similar_num += 1

        print('[predict label]', int(idx), '[predict name]:', name, '[raw_name]:', personnames[0], '[cos_dist]:', cos_dist)
        print()
        print('second label:', int(second_idx))
        print('second name:', second_name)
        print('second cos dist:', second_cos_dist)
        print()
    print('total accuracy rate:', similar_num / 712.0)
