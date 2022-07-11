"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
import wandb
from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torch.nn.functional as F
from utils.fps import farthest_point_sample_tensor
import numpy as np
import random

### set logging
logging.getLogger().setLevel(logging.INFO)

# Argument Parser
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.deepv3.DeepR50V3PlusD',
                    help='Network architecture. We have DeepR50V3PlusD (backbone: ResNet50) \
                    and DeepR101NV3PlusD (backbone: ResNet101).')
parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--combine_all', action='store_true', default=False,
                    help='combine both train, val, and test sets of the source data')
parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                    help='uniformly sample images across the multiple source domains')
parser.add_argument('--val_dataset', nargs='*', type=str, default=['bdd100k'],
                    help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
parser.add_argument('--cv', type=int, default=0,
                    help='cross-validation split id to use. Default # of splits set to 3 in config')
parser.add_argument('--class_uniform_pct', type=float, default=0,
                    help='What fraction of images is uniformly sampled')
parser.add_argument('--class_uniform_tile', type=int, default=1024,
                    help='tile size for class uniform sampling, this one should not exceed 760 for synthia')
parser.add_argument('--coarse_boost_classes', type=str, default=None,
                    help='use coarse annotations to boost fine data with specific classes')

parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                    help='class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class (use nll class weighting using batch stats')

parser.add_argument('--jointwtborder', action='store_true', default=False,
                    help='Enable boundary label relaxation')
parser.add_argument('--strict_bdr_cls', type=str, default='',
                    help='Enable boundary label relaxation for specific classes')
parser.add_argument('--rlx_off_iter', type=int, default=-1,
                    help='Turn off border relaxation after specific epoch count')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Warm Restarts new learning rate ratio compared to original lr')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Warm Restart new poly exp')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='Use Nvidia Apex AMP')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int,
                    help='0 means no aug, 1 means hard negative mining iter 1,' +
                    '2 means hard negative mining iter 2')

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=40000)
parser.add_argument('--max_cu_epoch', type=int, default=100000,
                    help='Class Uniform Max Epochs')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--crop_nopad', action='store_true', default=False)
parser.add_argument('--rrotate', type=int,
                    default=0, help='degree of random roate')
parser.add_argument('--color_aug', type=float,
                    default=0.0, help='level of color augmentation')
parser.add_argument('--gblur', action='store_true', default=False,
                    help='Use Guassian Blur Augmentation')
parser.add_argument('--bblur', action='store_true', default=False,
                    help='Use Bilateral Blur Augmentation')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=12,
                    help='Batch size for Validation per gpu')
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)

parser.add_argument('--city_mode', type=str, default='train',
                    help='experiment directory date name')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Use Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Dump Augmentated Images for sanity check')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Minimum testing to verify nothing failed, ' +
                    'Runs code for 1 epoch of train and val')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                    help='Weight Scaling for the losses')
parser.add_argument('--maxSkip', type=int, default=0,
                    help='Skip x number of  frames of video augmented dataset')
parser.add_argument('--scf', action='store_true', default=False,
                    help='scale correction factor')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--image_in', action='store_true', default=False,
                    help='Input Image Instance Norm')

parser.add_argument('--eval_epoch', type=int, default=1,
                    help='eval interval')

## style consistency
parser.add_argument('--sc_weight', type=float, default=10.0,
                    help='weight for consistency loss, e.g. js loss')

## retrospection consistency
parser.add_argument('--teacher', action='store_true', default=True,
                    help=' teacher: imgnet model')
parser.add_argument('--rc_layers', nargs='*', type=str, default=['layer4'],
                    help='a list of layers for retrospection loss : layer 0,1,2,3,4')
parser.add_argument('--rc_weights', nargs='*', type=float, default=[1.0],
                    help='weight for each layer feature of retrospection layer')


## style hallucination
parser.add_argument('--style_dim', type=int, default=64,
                    help='style compose dimension')
parser.add_argument('--base_style_num', type=int, default=64,
                    help='num of base style for style space, it should be same with the style dim, and it can also be larger for over modeling')
parser.add_argument('--concentration_coeff', type=float, default=0.0156,
                    help='coefficient for concentration')

## proto use seed
parser.add_argument('--proto_select_epoch', type=int, default=3,
                    help='epoch to select proto')
parser.add_argument('--online_proto', action='store_true', default=True,
                    help=' use online prototype')
parser.add_argument('--dynamic_proto', action='store_true', default=True,
                    help='replace proto every several epochs')
parser.add_argument('--set_proto_seed', action='store_true', default=True,
                    help='set seed for prototype dataloader')
parser.add_argument('--proto_trials', type=int, default=1)

## wandb for logs
parser.add_argument('--wandb_name', type=str, default='',
                    help='use wandb and wandb name')

args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1

# Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    # args.apex = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
# Initialize distributed communication
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)

torch.distributed.init_process_group(backend='nccl',
                                     init_method=args.dist_url,
                                     world_size=args.world_size,
                                     rank=args.local_rank)


def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    assert_and_infer_cfg(args)
    writer = prep_experiment(args, parser)
    if args.wandb_name:
        if args.local_rank == 0:
            wandb.init(project='SHADE', name=args.wandb_name, config=args)

    train_loader, val_loaders, train_obj, extra_val_loaders = datasets.setup_loaders(args)

    ## train proto loader
    if args.online_proto:
        _class_uniform_pct = args.class_uniform_pct
        args.class_uniform_pct = 0
        train_proto_loader, _,_,_ = datasets.setup_loaders(args)
        args.class_uniform_pct = _class_uniform_pct

    criterion, criterion_val = loss.get_loss(args)
    criterion_aux = loss.get_loss_aux(args)
    net = network.get_net(args)
    optim, scheduler = optimizer.get_optimizer(args, net)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    epoch = 0
    i = 0
    best_mean_iu = 0
    best_epoch = 0

    if args.snapshot:
        epoch, mean_iu = optimizer.load_weights(net, optim, scheduler,
                            args.snapshot, args.restore_optimizer)
        if args.restore_optimizer is True:
            iter_per_epoch = len(train_loader)
            i = iter_per_epoch * epoch
        else:
            epoch = 0

    if args.local_rank == 0:
        msg_args = ''
        args_dict = vars(args)
        for k, v in args_dict.items():
            msg_args = msg_args + str(k) + ' : ' + str(v) + ', '
        logging.info(msg_args)

    if args.teacher: ## imagenet model
        teacher_model = network.get_net(args)
        teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
        teacher_model = network.warp_network_in_dataparallel(teacher_model, args.local_rank)

    else:
        teacher_model = None

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True)

        if args.online_proto:
            if (args.dynamic_proto and not epoch % (args.proto_select_epoch)) \
                or (not args.dynamic_proto and epoch == args.proto_select_epoch):
                if i < args.max_iter * 0.95:
                    validate_for_prototype(train_proto_loader, net, epoch)

        print("#### iteration", i)
        torch.cuda.empty_cache()

        i = train(train_loader, net, optim, epoch, writer, scheduler, args.max_iter, teacher_model, criterion, criterion_aux)
        train_loader.sampler.set_epoch(epoch + 1)

        if (epoch+1) % args.eval_epoch == 0 or i >= args.max_iter:
            # torch.cuda.empty_cache()
            if args.local_rank == 0:
                print("Saving pth file...")
                evaluate_eval(args, net, optim, scheduler, None, None, [],
                            writer, epoch, "None", None, i, save_pth=True)

            for dataset, val_loader in extra_val_loaders.items():
                print("Extra validating... This won't save pth file")
                mean_iu = validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)

                if args.local_rank == 0:
                    if mean_iu > best_mean_iu:
                        best_mean_iu = mean_iu
                        best_epoch = epoch
                    
                    msg = 'Best Epoch:{}, Best mIoU:{:.5f}'.format(best_epoch, best_mean_iu)
                    if args.wandb_name:
                        wandb.log({
                            'epoch': best_epoch,
                            'cur_miou': mean_iu,
                            'best_miou': best_mean_iu
                        })
                    logging.info(msg)
                break

        if args.class_uniform_pct:
            if epoch >= args.max_cu_epoch:
                train_obj.build_epoch(cut=True)
                train_loader.sampler.set_num_samples()
            else:
                train_obj.build_epoch()

        epoch += 1

    for dataset, val_loader in extra_val_loaders.items():
        print("Extra validating... This won't save pth file")
        miou = validate(val_loader, dataset, net, criterion_val, optim, scheduler, epoch, writer, i, save_pth=False)
        if args.local_rank == 0:
            if args.wandb_name:
                wandb.log({
                    dataset: miou
                })

def train(train_loader, net, optim, curr_epoch, writer, scheduler, max_iter, teacher_model, criterion, criterion_aux):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()
    if teacher_model is not None:
        teacher_model.eval()
    
    # ipdb.set_trace()

    train_total_loss = AverageMeter()
    time_meter = AverageMeter()
    sc_loss_meter = AverageMeter()
    rc_loss_meter = AverageMeter()

    curr_iter = curr_epoch * len(train_loader)

    for i, data in enumerate(train_loader):
        if curr_iter >= max_iter:
            break

        inputs, gts, _, aux_gts = data
        # ipdb.set_trace()
        # Multi source and AGG case
        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1)
            gts = gts.transpose(0, 1).squeeze(2)
            aux_gts = aux_gts.transpose(0, 1).squeeze(2)

            inputs = [input.squeeze(0) for input in torch.chunk(inputs, num_domains, 0)]
            gts = [gt.squeeze(0) for gt in torch.chunk(gts, num_domains, 0)]
            aux_gts = [aux_gt.squeeze(0) for aux_gt in torch.chunk(aux_gts, num_domains, 0)]
        else:
            B, C, H, W = inputs.shape
            num_domains = 1
            inputs = [inputs]
            gts = [gts]
            aux_gts = [aux_gts]

        batch_pixel_size = C * H * W
            
        for di, ingredients in enumerate(zip(inputs, gts, aux_gts)):
            input, gt, aux_gt = ingredients

            start_ts = time.time()

            img_gt = None
            input, gt, aux_gt = input.cuda(), gt.cuda(), aux_gt.cuda()
            
            gt = torch.cat((gt, gt), dim=0)
            aux_gt = torch.cat((aux_gt, aux_gt), dim=0)
            if teacher_model is not None:
                with torch.no_grad():
                    imgnet_out = teacher_model(input, out_prob=True, return_style_features=args.rc_layers)


            optim.zero_grad()

            outputs = net(input, style_hallucination=True, out_prob=True, return_style_features=args.rc_layers)

            main_out = outputs['main_out']
            aux_out = outputs['aux_out']
            
            main_loss = criterion(main_out, gt)

            if aux_gt.dim() == 1:
                aux_gt = gt
            aux_gt = aux_gt.unsqueeze(1).float()
            aux_gt = F.interpolate(aux_gt, size=aux_out.shape[2:], mode='nearest')
            aux_gt = aux_gt.squeeze(1).long()
            aux_loss = criterion_aux(aux_out, aux_gt)

            total_loss = main_loss + (0.4 * aux_loss)
                
            if args.sc_weight:
                outputs_sm = F.softmax(main_out, dim=1) ##  2B,C,H,W, first B is x, last B is x_new
                im_prob = outputs_sm[:B] 
                aug_prob = outputs_sm[B:] 

                aug_prob = aug_prob.permute(0,2,3,1).reshape(-1, datasets.num_classes)
                im_prob = im_prob.permute(0,2,3,1).reshape(-1, datasets.num_classes)
                
                p_mixture = torch.clamp((aug_prob + im_prob) / 2., 1e-7, 1).log()
                consistency_loss = args.sc_weight * (
                            F.kl_div(p_mixture, aug_prob, reduction='batchmean') +
                            F.kl_div(p_mixture, im_prob, reduction='batchmean') 
                            ) / 2.
                
                total_loss = total_loss + consistency_loss

                sc_loss_meter.update(consistency_loss.item(), batch_pixel_size)


            if isinstance(args.rc_layers, list):
                f_style = outputs['features']
                f_imgnet = imgnet_out['features']
                feat_loss = 0.
                for layer, l_w in zip(args.rc_layers, args.rc_weights):
                    _f_imgnet = torch.cat((f_imgnet[layer], f_imgnet[layer]), dim=0).detach()
                    _floss = loss.calc_feat_dist(gt, _f_imgnet, f_style[layer], datasets.num_classes)
                    feat_loss = feat_loss + l_w * _floss
                    
                total_loss = total_loss + feat_loss
                
                rc_loss_meter.update(feat_loss.item(), C)

            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / args.world_size
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)

            total_loss.backward()
            optim.step()

            time_meter.update(time.time() - start_ts)

            del total_loss, log_total_loss
                
            if args.local_rank == 0:
                if i % 30 == 29:

                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [sc loss {:0.6f}], [rc loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        curr_epoch, i + 1, len(train_loader), curr_iter, train_total_loss.avg, sc_loss_meter.avg, rc_loss_meter.avg,
                        optim.param_groups[-1]['lr'], time_meter.avg) #  / args.train_batch_size

                    logging.info(msg)
                    if args.wandb_name:
                        wandb.log({
                            'loss':train_total_loss.avg,
                            'SC loss':sc_loss_meter.avg,
                            'RC loss':rc_loss_meter.avg
                        })
                    # Log tensorboard metrics for each iteration of the training phase
                    writer.add_scalar('loss/train_loss', (train_total_loss.avg),
                                    curr_iter)
                    train_total_loss.reset()
                    time_meter.reset()

        curr_iter += 1
        scheduler.step()

        if i > 20 and args.test_mode:
            return curr_iter

    return curr_iter

def validate(val_loader, dataset, net, criterion, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    for val_idx, data in enumerate(val_loader):
        inputs, gt_image, img_names, _ = data

        if len(inputs.shape) == 5:
            B, D, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W)
            gt_image = gt_image.view(-1, 1, H, W)

        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
        assert inputs.size()[2:] == gt_image.size()[1:]

        batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            output = net(inputs)
        del inputs

        assert output.size()[2:] == gt_image.size()[1:]
        assert output.size()[1] == datasets.num_classes

        val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

        del gt_cuda

        # Collect data from different GPU to a single GPU since
        # encoding.parallel.criterionparallel function calculates distributed loss
        # functions
        predictions = output.data.max(1)[1].cpu()

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        if val_idx > 10 and args.test_mode:
            break

        # Image Dumps
        if val_idx < 10:
            dump_images.append([gt_image, predictions, img_names])

        iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                             datasets.num_classes)
        del output, val_idx, data

    iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
    torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
    iou_acc = iou_acc_tensor.cpu().numpy()

    if args.local_rank == 0:
        mean_iu = evaluate_eval(args, net, optim, scheduler, val_loss, iou_acc, dump_images,
                    writer, curr_epoch, dataset, None, curr_iter, save_pth=save_pth)
    else:
        mean_iu = 0

    return mean_iu


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def validate_for_prototype(train_loader, net, epoch):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    criterion: loss fn
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    style_list = torch.empty(size=(0,2,args.style_dim)).cuda() # 0,2,C
    for trial in range(args.proto_trials):
        if args.set_proto_seed: 
            train_loader.sampler.set_epoch(epoch + trial)
        for val_idx, data in enumerate(train_loader):
            img, _, _, _ = data 
            img = img.cuda()

            with torch.no_grad():
                features = net(img, return_style_features=True) # B,C,H,W

            img_mean = features.mean(dim=[2,3]) # B,C
            img_var = features.var(dim=[2,3]) # B,C
            img_sig = (img_var+1e-7).sqrt()
            img_statis = torch.stack((img_mean, img_sig), dim=1) # B,2,C
            style_list = torch.cat((style_list, img_statis), dim=0)

            del img

            # Logging
            if val_idx % 20 == 0:
                if args.local_rank == 0:
                    logging.info("trial {:d} \t validating for prototype: {:d} / {:d} ".format(trial, val_idx, len(train_loader)))
            if args.test_mode and val_idx > 30:
                break
            del data

    style_list = concat_all_gather(style_list) # N,2,C

    print('final style list size : ',style_list.size())
    style_list = style_list.reshape(style_list.size(0), -1).detach()

    proto_styles, centroids = farthest_point_sample_tensor(style_list, args.base_style_num) # C,2C


    proto_styles = proto_styles.reshape(args.base_style_num, 2, args.style_dim)
    proto_mean, proto_std = proto_styles[:,0], proto_styles[:,1]
    print('proto style shape ===> ',proto_styles.shape)

    if args.local_rank == 0:
        print('style info first after calculation~~~')
        print(proto_mean)


    net.module.SHM.proto_mean.copy_(proto_mean)
    net.module.SHM.proto_std.copy_(proto_std)

    del style_list, proto_styles, proto_mean, proto_std




if __name__ == '__main__':
    main()
