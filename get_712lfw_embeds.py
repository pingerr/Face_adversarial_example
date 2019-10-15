
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from scipy import misc

from nets.model_irse import IR_50, IR_152
from nets.insightface import Backbone, MobileFaceNet


model_root = './ckpt/'
model_path_map = {'IR_50': model_root + 'backbone_ir50_ms1m_epoch120.pth',
                  'IR_152': model_root + 'Backbone_IR_152_Epoch_112_Batch.pth',
                  'IR_SE_50': model_root + 'model_ir_se50.pth'}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--read_path', type=str, default='./images/lfw_align_112',
                        help='path to image file or directory to images')

    return parser.parse_args()


def load_personnames(input_dir='./images/raw_images'):
    personnames = []
    idx = 0

    dev = pd.read_csv(os.path.join(input_dir, 'dev.csv'))
    filename2personname = {dev.iloc[i]['ImageName']: dev.iloc[i]['PersonName'] for i in range(len(dev))}
    for filename in filename2personname.keys():
        personnames.append(filename2personname[filename])
        idx += 1

    return personnames


def get_dataset_and_num(root_path):
    print('reading rootpath:', root_path)
    classnames = load_personnames()

    num_class = len(classnames)
    dataset = []
    total_num = 0
    for idx in range(num_class):
        classname = classnames[idx]
        classdir = os.path.join(root_path, classname)
        imagepaths, num = get_image_paths(classdir)
        total_num += num
        dataset.append((idx, classname, imagepaths))
    return dataset, total_num


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        images.sort()
        image_paths = [os.path.join(facedir, img) for img in images]
        # nums = len(image_paths)
    return image_paths, len(image_paths)


def load_image(path):
    print('reading image %s' % path)

    img = misc.imread(path).astype(np.float32)

    img = (img-127.5) / 128.0

    return img


def get_embds(model, images, device):

    if images.shape[0] > 20:
        for_stack = []
        if images.shape[0] % 20 == 0:
            for i in range(images.shape[0] // 20):
                tensor = torch.from_numpy(images[i * 20: (i + 1) * 20])
                print('tensor inner shape:', tensor.shape)
                variable = tensor.detach().to(device)
                embeddings_ = model(variable).data.cpu().detach().numpy()
                for_stack.append(embeddings_)
        else:
            for i in range(images.shape[0]//20 + 1):

                if i == images.shape[0]//20:
                    tensor = torch.from_numpy(images[i*20:])
                    print('last inner shape:', tensor.shape)
                    variable = tensor.detach().to(device)
                    embeddings_ = model(variable).data.cpu().detach().numpy()
                    for_stack.append(embeddings_)
                else:
                    tensor = torch.from_numpy(images[i*20: (i+1)*20])
                    print('tensor inner shape:', tensor.shape)
                    variable = tensor.detach().to(device)
                    embeddings_ = model(variable).data.cpu().detach().numpy()
                    for_stack.append(embeddings_)
        embeddings = np.concatenate(for_stack, axis=0)

    else:
        tensor = torch.from_numpy(images)
        print('tensor shape:', tensor.shape)
        variable = tensor.detach().to(device)

        embeddings = model(variable).data.cpu().detach().numpy()

    embds = embeddings

    print('embedding a class!')
    print('images num:', len(embds))

    return embds


def main():
    args = get_args()

    print('loading...')
    device = torch.device('cuda')

    # model 1
    # model_ir50_epoch120 = IR_50([112, 112])
    # model_ir50_epoch120.load_state_dict(torch.load(model_path_map['IR_50'], map_location='cuda'))
    # model_ir50_epoch120.eval().to(device).zero_grad()

    # model 2
    # model_IR_152_Epoch_112 = IR_152([112, 112])
    # model_IR_152_Epoch_112.load_state_dict(torch.load(model_path_map['IR_152'], map_location='cuda'))
    # model_IR_152_Epoch_112.eval().to(device).zero_grad()
    #
    # # model 3
    IR_SE_50 = Backbone(50, mode='ir_se')
    IR_SE_50.load_state_dict(torch.load(model_path_map['IR_SE_50'], map_location='cuda'))
    IR_SE_50.eval().to(device).zero_grad()

    model = IR_SE_50

    dataset, n = get_dataset_and_num(args.read_path)

    print('total image nums:', n)

    all_embds = np.zeros((n, 513))  # 512 + idx
    idx2classname = {}
    num = 0
    for idx, classname, imagepaths in dataset:

        images_batch = []
        idx2classname[idx] = classname
        print(idx, classname)
        for imagepath in imagepaths:
            images_batch.append(load_image(imagepath))

        images_batch = np.array(images_batch).swapaxes(2, 3).swapaxes(1, 2)

        print('images in shape :', images_batch.shape)
        a_class_embds = get_embds(model, images_batch, device)
        print('this class embeds shape', a_class_embds.shape)

        num_next = num + a_class_embds.shape[0]
        all_embds[num:num_next, 0] = idx
        all_embds[num:num_next, 1:] = a_class_embds
        num = num_next

    print('all done!')
    print('saving...')
    f1 = idx2classname
    f2 = all_embds

    with open('./embeds_pkl/all712_by_irse50.pkl',
              'wb') as file:
        pickle.dump((f1, f2), file)
    file.close()


    print('done!')


if __name__ == '__main__':
    main()
