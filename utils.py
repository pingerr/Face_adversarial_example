import numpy as np
import pandas as pd
import os
import copy
import pickle as pkl
import torch.nn.functional as F
import scipy.stats as st
import torch
import random
import argparse
from PIL import Image
from scipy.misc import imread


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./images/raw_images')
parser.add_argument('--output_dir', default='./images/advimages/model_(1*2, 2, 3)_13_25_cosloss_gs2_3'
                                            '_mi0.9')
parser.add_argument('--max_epsilon', help='Maximum size of adversarial perturbation.', default=11.0)
parser.add_argument('--image_size', default=112)
parser.add_argument('--image_resize', default=130)
parser.add_argument('--batch_size', default=8)
parser.add_argument('--prob', help='probability of using diverse inputs.', default=0.5)
parser.add_argument('--sig', help='gradient smoothing', default=3)
parser.add_argument('--kernlen', help='gradient smoothing kernel len', default=3)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--iterations', default=25)
parser.add_argument('--cos_margin', help='', default=0.8)

args = parser.parse_args()


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def input_diversity(image, prob=args.prob, low=args.image_size, high=args.image_resize):
    if random.random()<prob:
        return image
    rnd = random.randint(low, high)
    rescaled = F.upsample(image, size=[rnd, rnd], mode='bilinear')
    h_rem = high - rnd
    w_rem = high - rnd
    pad_top = random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
    padded = F.upsample(padded, size=[low, low], mode='bilinear')
    return padded


def load_images_with_names(input_dir, batch_size):
    images = []
    filenames = []
    personnames = []
    idx = 0

    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2personname = {dev.iloc[i]['ImageName']: dev.iloc[i]['PersonName'] for i in range(len(dev))}
    for filename in filename2personname.keys():
        image = (imread(os.path.join(input_dir, filename))).astype(np.float32)
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


def process(img):
    img = img.swapaxes(2, 3).swapaxes(1, 2)
    img = (img - 127.5) / 128.0
    img = np.array(img, dtype=np.float32)
    img = torch.from_numpy(img)

    return img


def save_images(r_images, images, filenames, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    total_distance_ = 0

    for i, filename in enumerate(filenames):

        image = (images[i] * 128.0 + 127.5)
        image = np.around(image).clip(0, 255).astype(np.uint8)
        Image.fromarray(image).save(os.path.join(output_dir, filename))
        distance = calc_dist(r_images[i], image)
        print("distance:", distance)
        total_distance_ += distance
    print('current mean distance: ', total_distance_ / len(filenames))

    return total_distance_


def calc_dist(r_img, adv_img):
    adv_image_arr = np.array(r_img).astype(np.int32)
    raw_image_arr = np.array(adv_img).astype(np.int32)
    diff = adv_image_arr.reshape((-1, 3)) - raw_image_arr.reshape((-1, 3))
    distance = np.mean(np.sqrt(np.sum((diff ** 2), axis=1)))
    return distance


class Classifier(object):
    def __init__(self):
        with open('./embeds_pkl/all_by_ir50.pkl', 'rb+') as f:
            self.label2classname, aver_embeds_ = pkl.load(f)
        f.close()
        self.aver_embeds_ = aver_embeds_[:, 1:]
        self.idx2label = aver_embeds_[:, 0]

    def classifier(self, image_embed_):
        image_embed = image_embed_ / np.linalg.norm(image_embed_, axis=1, keepdims=True)  # (bt, 512)

        aver_embeds = self.aver_embeds_ / np.linalg.norm(self.aver_embeds_, axis=1, keepdims=True)  # (1w+, 512)

        cos_distances = image_embed.dot(aver_embeds.T)  # (89, 512).(512, 1w+) = (bt, 1w+)

        idx = np.argmax(cos_distances, axis=1)  # (89,)

        label = self.idx2label[idx]

        return label


class Mask(object):
    def __init__(self):
        pkl_path1 = './embeds_pkl/all_by_ir50.pkl'
        pkl_path2 = './embeds_pkl/all_by_ir152.pkl'
        pkl_path3 = './embeds_pkl/all_by_irse50.pkl'

        with open(pkl_path1, 'rb+') as f:
            self.label2classname, all_embeds_1 = pkl.load(f)
        f.close()
        with open(pkl_path2, 'rb+') as f:
            _, all_embeds_2 = pkl.load(f)
        f.close()
        with open(pkl_path3, 'rb+') as f:
            _, all_embeds_3 = pkl.load(f)
        f.close()
        self.idx2label = all_embeds_1[:, 0]  # 1w+,
        self.all_embeds_1 = all_embeds_1[:, 1:].astype(np.float32)  # 1w+, 512
        self.all_embeds_2 = all_embeds_2[:, 1:].astype(np.float32)
        self.all_embeds_3 = all_embeds_3[:, 1:].astype(np.float32)

    def get_perturb_mask(self, image_embed_1, image_embed_2, image_embed_3, true_idx, margin):

        image_embed_1 = image_embed_1 / np.linalg.norm(image_embed_1, axis=1, keepdims=True)
        image_embed_2 = image_embed_2 / np.linalg.norm(image_embed_2, axis=1, keepdims=True)
        image_embed_3 = image_embed_3 / np.linalg.norm(image_embed_3, axis=1, keepdims=True)

        all_embeds1 = self.all_embeds_1 / np.linalg.norm(self.all_embeds_1, axis=1, keepdims=True)
        all_embeds2 = self.all_embeds_2 / np.linalg.norm(self.all_embeds_2, axis=1, keepdims=True)
        all_embeds3 = self.all_embeds_3 / np.linalg.norm(self.all_embeds_3, axis=1, keepdims=True)

        cos_distances1 = image_embed_1.dot(all_embeds1.T)  # (batch, 512).(512, 1w+) = (batch, 1w+)
        cos_distances2 = image_embed_2.dot(all_embeds2.T)
        cos_distances3 = image_embed_3.dot(all_embeds3.T)

        cos_distances = (cos_distances1 + cos_distances2 * 0.35 + cos_distances3 * 0.65) / 2.0

        max_idx = np.argmax(cos_distances, axis=1)  # (89,)

        max_cos_dist = np.max(cos_distances, axis=1)  # (89,)

        max_true_dists = []
        for i in range(0, true_idx.shape[0]):
            bool_cond = np.equal(self.idx2label, true_idx[i])
            index = np.where(bool_cond)

            max_true_dist = np.max(cos_distances[i][index])
            max_true_dists.append(max_true_dist)

        distance_mask = np.greater(max_cos_dist, np.add(max_true_dists, margin))

        zero_mask = np.zeros(true_idx.shape[0])
        one_mask = np.ones(true_idx.shape[0])
        peturb_mask = np.where(np.equal(distance_mask, True), zero_mask, one_mask)

        return peturb_mask


def found_bias_v1(image_embed_1, image_embed_2, image_embed_3, batch_size):

        pkl_path1 = './embeds_pkl/712_by_ir50.pkl'
        pkl_path2 = './embeds_pkl/712_by_ir152.pkl'
        pkl_path3 = './embeds_pkl/712_by_irse50.pkl'

        with open(pkl_path1, 'rb+') as f:
            _, all_embeds_1 = pkl.load(f)
        f.close()
        with open(pkl_path2, 'rb+') as f:
            _, all_embeds_2 = pkl.load(f)
        f.close()
        with open(pkl_path3, 'rb+') as f:
            _, all_embeds_3 = pkl.load(f)
        f.close()
        idx2label = all_embeds_1[:, 0]
        all_embeds_1 = all_embeds_1[:, 1:].astype(np.float32)
        all_embeds_2 = all_embeds_2[:, 1:].astype(np.float32)
        all_embeds_3 = all_embeds_3[:, 1:].astype(np.float32)

        image_embed_1 = image_embed_1 / np.linalg.norm(image_embed_1, axis=1, keepdims=True)
        image_embed_2 = image_embed_2 / np.linalg.norm(image_embed_2, axis=1, keepdims=True)
        image_embed_3 = image_embed_3 / np.linalg.norm(image_embed_3, axis=1, keepdims=True)

        all_embeds1 = all_embeds_1 / np.linalg.norm(all_embeds_1, axis=1, keepdims=True)
        all_embeds2 = all_embeds_2 / np.linalg.norm(all_embeds_2, axis=1, keepdims=True)
        all_embeds3 = all_embeds_3 / np.linalg.norm(all_embeds_3, axis=1, keepdims=True)

        cos_distances1 = image_embed_1.dot(all_embeds1.T)  # (batch, 512).(512, 712) = (batch, 712)
        cos_distances2 = image_embed_2.dot(all_embeds2.T)
        cos_distances3 = image_embed_3.dot(all_embeds3.T)

        cos_distances = (cos_distances1 + cos_distances2 * 0.35 + cos_distances3 * 0.65) / 2.0

        idx = np.argmax(cos_distances, axis=1)  # (batch,)

        label = idx2label[idx]

        second_idxes = []
        for i in range(0, batch_size):
            cos_distances_ = copy.deepcopy(cos_distances)
            cos_distances_[i][idx[i]] = np.min(cos_distances_[i])
            second_idx = np.argmax(cos_distances_[i])

            second_idxes.append(second_idx)

        second_embed1 = all_embeds_1[second_idxes]
        second_embed2 = all_embeds_2[second_idxes]
        second_embed3 = all_embeds_3[second_idxes]

        return second_embed1, second_embed2, second_embed3


def found_bias_v2(image_embed_1, image_embed_2, image_embed_3, batch_size):
    pkl_path1 = './embeds_pkl/all712_by_ir50.pkl'
    pkl_path2 = './embeds_pkl/all712_by_ir152.pkl'
    pkl_path3 = './embeds_pkl/all712_by_irse50.pkl'

    with open(pkl_path1, 'rb+') as f:
        _, all_embeds_1 = pkl.load(f)
    f.close()
    with open(pkl_path2, 'rb+') as f:
        _, all_embeds_2 = pkl.load(f)
    f.close()
    with open(pkl_path3, 'rb+') as f:
        _, all_embeds_3 = pkl.load(f)
    f.close()
    idx2label = all_embeds_1[:, 0]
    all_embeds_1 = all_embeds_1[:, 1:].astype(np.float32)
    all_embeds_2 = all_embeds_2[:, 1:].astype(np.float32)
    all_embeds_3 = all_embeds_3[:, 1:].astype(np.float32)

    image_embed_1 = image_embed_1 / np.linalg.norm(image_embed_1, axis=1, keepdims=True)
    image_embed_2 = image_embed_2 / np.linalg.norm(image_embed_2, axis=1, keepdims=True)
    image_embed_3 = image_embed_3 / np.linalg.norm(image_embed_3, axis=1, keepdims=True)

    all_embeds1 = all_embeds_1 / np.linalg.norm(all_embeds_1, axis=1, keepdims=True)
    all_embeds2 = all_embeds_2 / np.linalg.norm(all_embeds_2, axis=1, keepdims=True)
    all_embeds3 = all_embeds_3 / np.linalg.norm(all_embeds_3, axis=1, keepdims=True)

    cos_distances1 = image_embed_1.dot(all_embeds1.T)  # (batch, 512).(512, 5K+) = (batch, 5k+)
    cos_distances2 = image_embed_2.dot(all_embeds2.T)
    cos_distances3 = image_embed_3.dot(all_embeds3.T)

    # print('cos_dist shape:', cos_distances.shape)
    cos_distances = (cos_distances1 + cos_distances2 * 0.35 + cos_distances3 * 0.65) / 2.0

    idx = np.argmax(cos_distances, axis=1)  # (batch,)

    labels = idx2label[idx]

    second_idxes = []
    for i in range(0, batch_size):
        bool_cond = np.equal(idx2label, labels[i])
        true_index = np.where(bool_cond)
        cos_distances_ = copy.deepcopy(cos_distances)
        cos_distances_[i][true_index] = np.min(cos_distances_[i])
        second_idx = np.argmax(cos_distances_[i])
        second_idxes.append(second_idx)
    # print(second_idxes)
    second_embed1 = all_embeds_1[second_idxes]
    second_embed2 = all_embeds_2[second_idxes]
    second_embed3 = all_embeds_3[second_idxes]

    return second_embed1, second_embed2, second_embed3


def found_bias_v3(image_embed_1, image_embed_2, image_embed_3, batch_size, true_labels):
    pkl_path1 = './embeds_pkl/all_by_ir50.pkl'
    pkl_path2 = './embeds_pkl/all_by_ir152.pkl'
    pkl_path3 = './embeds_pkl/all_by_irse50.pkl'

    with open(pkl_path1, 'rb+') as f:
        _, all_embeds_1 = pkl.load(f)
    f.close()
    with open(pkl_path2, 'rb+') as f:
        _, all_embeds_2 = pkl.load(f)
    f.close()
    with open(pkl_path3, 'rb+') as f:
        _, all_embeds_3 = pkl.load(f)
    f.close()
    idx2label = all_embeds_1[:, 0]
    all_embeds_1 = all_embeds_1[:, 1:].astype(np.float32)
    all_embeds_2 = all_embeds_2[:, 1:].astype(np.float32)
    all_embeds_3 = all_embeds_3[:, 1:].astype(np.float32)

    image_embed_1 = image_embed_1 / np.linalg.norm(image_embed_1, axis=1, keepdims=True)
    image_embed_2 = image_embed_2 / np.linalg.norm(image_embed_2, axis=1, keepdims=True)
    image_embed_3 = image_embed_3 / np.linalg.norm(image_embed_3, axis=1, keepdims=True)

    all_embeds1 = all_embeds_1 / np.linalg.norm(all_embeds_1, axis=1, keepdims=True)
    all_embeds2 = all_embeds_2 / np.linalg.norm(all_embeds_2, axis=1, keepdims=True)
    all_embeds3 = all_embeds_3 / np.linalg.norm(all_embeds_3, axis=1, keepdims=True)

    cos_distances1 = image_embed_1.dot(all_embeds1.T)  # (batch, 512).(512, 5K+) = (batch, 5k+)
    cos_distances2 = image_embed_2.dot(all_embeds2.T)
    cos_distances3 = image_embed_3.dot(all_embeds3.T)

    cos_distances = (cos_distances1 + cos_distances2 * 0.35 + cos_distances3 * 0.65) / 2.0

    idx = np.argmax(cos_distances, axis=1)  # (batch,)

    labels = idx2label[idx]

    second_idxes = []
    for i in range(0, batch_size):
        bool_cond = np.equal(idx2label, true_labels[i])
        true_index = np.where(bool_cond)
        cos_distances_ = copy.deepcopy(cos_distances)
        cos_distances_[i][true_index] = np.min(cos_distances_[i])
        second_idx = np.argmax(cos_distances_[i])
        second_idxes.append(second_idx)

    second_embed1 = all_embeds_1[second_idxes]
    second_embed2 = all_embeds_2[second_idxes]
    second_embed3 = all_embeds_3[second_idxes]

    return second_embed1, second_embed2, second_embed3
