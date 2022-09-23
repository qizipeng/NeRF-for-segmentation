import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os
# from collections import OrderedDict
# from ddp_model import NerfNet
import time
from data_loader_split import load_data_split
from utils import mse2psnr, colorize_np, to8b
import imageio
from train import config_parser, setup_logger, setup, cleanup, render_single_image, create_nerf
import logging
import torch.nn.functional as F


logger = logging.getLogger(__package__)
###dataset8
carla8 = [
[70 ,70 ,70],
[150 , 60 , 45],
[180 ,130 , 70],
[232 , 35 ,244],
[ 35 ,142 ,107],
[100 ,170, 145],
[160 ,190, 110],
[153 ,153 ,153],
[80 ,90 ,55],
[ 50 ,120, 170],
[128 , 64 ,128],
[ 50 ,234, 157],
[142  , 0 ,  0],
[  0, 220, 220],
[ 30 ,170, 250],
[156, 102, 102],
[ 40  ,40 ,100],
[81 , 0 ,81],
[140 ,150, 230],
[100 ,100, 150]
]
###dataset9
carla9 = [
    [70, 70, 70],
    [232, 35, 244],
    [160, 190, 110],
    [35, 142, 107],
    [80, 90, 55],
    [150, 60, 45],
    [40, 40, 100],
    [153, 153, 153],
    [100, 100, 150],
    [50, 120, 170],
    [100, 170, 145],
    [156, 102, 102],
    [81, 0, 81],
    [142, 0, 0],
    [0, 220, 220],
    [128, 64, 128],
    [50, 234, 157],
    [30, 170, 250],
    [140, 150, 230],
    [180, 130, 70]
]
carla10 = [
    [70, 70, 70],
       [232 , 35, 244],
       [160 ,190, 110],
       [ 35 ,142, 107],
       [80 ,90, 55],
       [150 , 60,  45],
       [ 40 , 40, 100],
       [153 ,153, 153],
       [100 ,100, 150],
       [ 50 ,120, 170],
       [100 ,170, 145],
       [156 ,102, 102],
       [81  ,0 ,81],
       [142 ,  0  , 0],
       [  0 ,220 ,220],
       [128 , 64 ,128],
       [ 50 ,234 ,157],
       [ 30 ,170 ,250],
       [140 ,150 ,230],
       [180 ,130,  70]
]

##carla10
carla10=[
   [70 ,70 ,70],
   [232 , 35, 244],
   [160 ,190, 110],
   [ 35 ,142, 107],
   [80 ,90, 55],
   [150 , 60,  45],
   [ 40 , 40, 100],
   [153 ,153, 153],
   [100 ,100, 150],
   [ 50 ,120, 170],
   [100 ,170, 145],
   [156 ,102, 102],
   [81  ,0 ,81],
   [142 ,  0  , 0],
   [  0 ,220 ,220],
   [128 , 64 ,128],
   [ 50 ,234 ,157],
   [ 30 ,170 ,250],
   [140 ,150 ,230],
   [180 ,130,  70]
]


linken=[
    [70 ,70 ,70],
     [128 , 64 ,128],
    [ 35 ,142 ,107],
 ]
zhijiage=[
    [128 , 64 ,128],
     [160 ,190, 110],
    [142  , 0 ,  0],
        [70 ,70 ,70],
    [ 50 ,120, 170]
 ]
carla13 = [
   [150 , 60 , 45],
   [70 ,70 ,70],
[232  ,35 ,244],
[160 ,190 ,110],
[153 ,153 ,153],
[ 35 ,142, 107],
[80 ,90 ,55],
[100 ,170 ,145],
[128 , 64 ,128],
[ 50 ,234 ,157],
[ 30 ,170 ,250],
[  0 ,220 ,220],
[142 ,  0 ,  0],
[ 50 ,120 ,170],
[100 ,100 ,150],
[ 40 , 40, 100],
[81  ,0 ,81],
[156 ,102 ,102],
[140 ,150 ,230],
[180 ,130 , 70] ]


carla14 = [
   [70 ,70 ,70],
   [80 ,90 ,55],
   [232 , 35 ,244],
   [160 ,190 ,110],
   [ 35 ,142 ,107],
   [153 ,153 ,153],
   [100 ,170 ,145],
   [128 , 64 ,128],
   [ 50 ,234 ,157],
   [142 ,  0 ,  0],
   [ 50 ,120 ,170],
   [  0 ,220, 220],
   [81  ,0 ,81],
   [ 30 ,170 ,250],
   [180 ,130 , 70],
   [156 ,102 ,102],
   [140 ,150 ,230],
   [ 40 , 40, 100]
]

carla = carla14
def ddp_test_nerf(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 1024*12
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    start, models = create_nerf(rank, args)

    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]
    # start testing
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        ###### load data and create ray samplers; each process should do this
        ray_samplers = load_data_split(args.datadir, args.scene, split, try_load_min_depth=args.load_min_depth)
        for idx in range(len(ray_samplers)):
            ### each process should do this; but only main process merges the results
            fname = '{:06d}.png'.format(idx)
            if ray_samplers[idx].img_path is not None:
                fname = os.path.basename(ray_samplers[idx].img_path)

            if os.path.isfile(os.path.join(out_dir, fname)):
                logger.info('Skipping {}'.format(fname))
                continue

            time0 = time.time()
            ret = render_single_image(rank, args.world_size, models, ray_samplers[idx], args.chunk_size)
            dt = time.time() - time0
            if rank == 0:    # only main process should do this
                logger.info('Rendered {} in {} seconds'.format(fname, dt))

                # im = to8b(im)

                # print(ret[-1]["rgb"][0][0])
                # result = torch.zeros((512,512,20))
                # # for x in range(0, 512, 32):
                # #     for y in range(0, 512, 32):
                # #         for i in range(0, 32):
                # #             for j in range(0, 32):
                # #                 result[x+i, y+j,:] = ret[-1]["rgb"][512 * (x + i) + y + j][:]
                # num = 0
                # for x in range(0, 512, 32):
                #     for y in range(0, 512, 32):
                #         result[x:x+32,y:y+32,:] = ret[-1]["rgb"][num].reshape(32, 32, -1)
                #         num = num+1
                # result = ret[-1]["rgb"].reshpae(32,32,-1)
                if args.IsSegmantic== True:
                    im = F.softmax(ret[-1]["rgb"], dim=-1)
                    im = torch.argmax(im, dim=-1)
                    print(im.shape)
                    im = im.numpy()
                    im_seg = np.zeros((512, 512, 3))
                    for i in range(512):
                        for j in range(512):
                            im_seg[i,j,:] = carla[im[i,j]][:]

                    imageio.imwrite(os.path.join(out_dir, fname), im.astype(np.uint8))
                    imageio.imwrite(os.path.join(out_dir, "color"+fname), im_seg.astype(np.uint8))
                else:
                    im = ret[-1]['rgb'].numpy()
                    # compute psnr if ground-truth is available
                    if ray_samplers[idx].img_path is not None:
                        gt_im = ray_samplers[idx].get_img()
                        psnr = mse2psnr(np.mean((gt_im - im) * (gt_im - im)))
                        logger.info('{}: psnr={}'.format(fname, psnr))
                    im = to8b(im)  ##regress with rgb  /255 *255
                    imageio.imwrite(os.path.join(out_dir, fname), im)

                    # im = ret[-1]['fg_depth'].numpy()
                    # # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                    # im = to8b(im)
                    # imageio.imwrite(os.path.join(out_dir, 'fg_depth_' + fname), im)
                    #
                    # im = ret[-1]['bg_depth'].numpy()
                    # # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                    # im = to8b(im)
                    # imageio.imwrite(os.path.join(out_dir, 'bg_depth_' + fname), im)


                    depthf = ret[-1]['fg_depth'].numpy()
                    # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                    #depthf = to8b(depth1)
                    # imageio.imwrite(os.path.join(out_dir, 'fg_depth_' + fname), im)

                    depthb = ret[-1]['bg_depth'].numpy()
                    # im = colorize_np(im, cmap_name='jet', append_cbar=True)
                    #depthb = to8b(depth2)
                    depth = depthf+depthb
                    depth = to8b(depth)
                    imageio.imwrite(os.path.join(out_dir, 'depth_' + fname), depth)



            torch.cuda.empty_cache()

    # clean up for multi-processing
    cleanup()


def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    torch.multiprocessing.spawn(ddp_test_nerf,
                                args=(args,),
                                nprocs=args.world_size,
                                join=True)


if __name__ == '__main__':
    setup_logger()
    test()

