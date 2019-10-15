import numpy as np
import torch
import time
import argparse
import pickle as pkl
import torch.nn.functional as F

from nets.model_irse import IR_50, IR_152
from nets.insightface import Backbone, MobileFaceNet

from utils import gkern, load_images_with_names, process, save_images, found_bias_v2, Classifier, Mask


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./images/raw_images')
parser.add_argument('--output_dir', default='./images/advimages/'
                                            'model_(1*1.7, 2*0.35, 3*0.65)_13_26_cosloss(1:1)_margin0.2_gs3_3')
parser.add_argument('--max_epsilon', help='Maximum size of adversarial perturbation.', default=13.0)
parser.add_argument('--image_size', default=112)
parser.add_argument('--batch_size', default=8)
parser.add_argument('--sig', help='gradient smoothing', default=3)
parser.add_argument('--kernlen', help='gradient smoothing kernel len', default=3)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--iterations', default=26)
parser.add_argument('--cos_margin', help='', default=0.22)

args = parser.parse_args()


def main():

    m = Mask()
    c = Classifier()

    device = torch.device('cuda')

    # IR_50
    model_ir50 = IR_50([112, 112])
    model_ir50.load_state_dict(torch.load('./ckpt/backbone_ir50_ms1m_epoch120.pth', map_location='cuda'))
    model_ir50.eval().to(device).zero_grad()

    # IR_152
    model_ir152 = IR_152([112, 112])
    model_ir152.load_state_dict(torch.load('./ckpt/Backbone_IR_152_Epoch_112_Batch.pth', map_location='cuda'))
    model_ir152.eval().to(device).zero_grad()

    # IR_SE_50
    model_irse50 = Backbone(50, mode='ir_se')
    model_irse50.load_state_dict(torch.load('./ckpt/model_ir_se50.pth', map_location='cuda'))
    model_irse50.eval().to(device).zero_grad()

    eps = (args.max_epsilon / 255.0)
    alpha = eps / args.iterations

    momentum = args.momentum

    kernel = gkern(args.kernlen, args.sig).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    stack_kernel = torch.Tensor(stack_kernel).to(device)

    counter = 0
    total_distance = 0
    num = 1
    for raw_images, filenames, _ in load_images_with_names(args.input_dir, args.batch_size):

        if num * args.batch_size > 712:
            batch_size = 712 - (num - 1) * args.batch_size
        else:
            batch_size = args.batch_size
        num += 1

        in_tensor = process(raw_images)
        raw_variable = in_tensor.detach().to(device)

        # raw embedding
        raw_ir50 = model_ir50(raw_variable)
        raw_ir152 = model_ir152(raw_variable)
        raw_irse50 = model_irse50(raw_variable)

        true_labels = c.classifier(raw_ir50.data.cpu().detach().numpy())

        bias_ir50, bias_ir152, bias_irse50 = found_bias_v2(raw_ir50.data.cpu().detach().numpy(),
                                                        raw_ir152.data.cpu().detach().numpy(),
                                                        raw_irse50.data.cpu().detach().numpy(),
                                                        batch_size)

        perturbation = torch.Tensor(batch_size, 3, 112, 112).uniform_(-0.01, 0.01).to(device)
        in_variable = raw_variable + perturbation
        in_variable.data.clamp_(-1.0, 1.0)
        in_variable.requires_grad = True

        last_grad = 0.0
        momentum_sum = 0.0

        for step in range(args.iterations):

            new_ir50 = model_ir50(in_variable)
            new_ir152 = model_ir152(in_variable)
            new_irse50 = model_irse50(in_variable)

            loss1 = - torch.mean(torch.cosine_similarity(x1=raw_ir50,
                                                        x2=new_ir50, dim=1) * 1.7 +
                                torch.cosine_similarity(x1=raw_ir152,
                                                        x2=new_ir152, dim=1) * 0.35 +
                                torch.cosine_similarity(x1=raw_irse50,
                                                        x2=new_irse50, dim=1) * 0.65) / 2.7

            loss2 = torch.mean(torch.cosine_similarity(x1=torch.from_numpy(bias_ir50).detach().to(device),
                                                       x2=new_ir50, dim=1) * 1.7 +
                               torch.cosine_similarity(x1=torch.from_numpy(bias_ir152).detach().to(device),
                                                       x2=new_ir152, dim=1) * 0.35 +
                               torch.cosine_similarity(x1=torch.from_numpy(bias_irse50).detach().to(device),
                                                       x2=new_irse50, dim=1) * 0.65) / 2.7
            loss = loss1 + loss2

            print('loss :', loss)

            loss.backward(retain_graph=True)

            data_grad = in_variable.grad.data

            data_grad = F.conv2d(data_grad, stack_kernel, padding=(args.kernlen - 1) // 2, groups=3)

            for i in range(data_grad.shape[0]):
                data_grad[i] = data_grad[i] / torch.mean(data_grad[i].norm(2, 0) / 1.713)

            if iter == 0:
                noise = data_grad
            else:
                noise = last_grad * momentum + data_grad * 0.9

            last_grad = noise.detach()
            norm = noise.norm(dim=1).unsqueeze(1)
            index = norm.mean()
            momentum_sum = momentum_sum * momentum + 1.0
            d_img = noise * norm * alpha / (momentum_sum*index)
            d_img = d_img/d_img.norm(dim=1).mean()*alpha

            perturb_mask = m.get_perturb_mask(new_ir50.data.detach().cpu().numpy(),
                                              new_ir152.data.detach().cpu().numpy(),
                                              new_irse50.data.detach().cpu().numpy(),
                                              true_labels, args.cos_margin)

            in_variable.data = in_variable.data + \
                               d_img * torch.from_numpy(perturb_mask.reshape([batch_size, 1, 1, 1])).to(device).float()

            raw_variable.data = torch.clamp(in_variable.data, -1.0, 1.0)
            in_variable.grad.data.zero_()

        advs = raw_variable.data.cpu().detach().numpy()
        advs = advs.swapaxes(1, 2).swapaxes(2, 3)

        total_distance_ = save_images(raw_images, advs, filenames, args.output_dir)
        total_distance += total_distance_
        counter += batch_size
        print('attack images num : [%d / 712]' % counter)
    print('mean_dist:', total_distance / 712.0)


if __name__ == '__main__':
    time1 = time.time()
    main()
    t = time.time() - time1
    print('code run time {:.0f}min {:.0f}second'.format(t // 60, t % 60))
