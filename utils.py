from PIL import Image, ImageDraw
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import imageio
import shutil
##dataset8
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
carla9=[
   [70 ,70 ,70],
   [ 35, 142, 107],
   [153, 153, 153],
   [80 ,90, 55],
   [81 , 0 ,81],
   [232 , 35, 244],
   [128 , 64, 128],
   [ 50, 234 ,157],
   [  0 ,220 ,220],
   [ 50, 120, 170],
   [ 30 ,170, 250],
   [140 ,150, 230],
   [160 ,190, 110],
   [180 ,130,  70],
   [142 ,  0 ,  0],
   [100, 170, 145],
   [156 ,102, 102],
   [ 40,  40, 100]
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
##carla11
carla11 = [
[150,  60,  45],
    [70 ,70 ,70],
    [232 , 35 ,244],
    [140 ,150 ,230],
    [180 ,130 , 70],
    [153 ,153 ,153],
    [142 ,  0,   0],
    [81  ,0 ,81],
    [100 ,170 ,145],
    [ 50 ,120, 170],
    [80 ,90 ,55],
    [ 40 , 40 ,100],
    [156 ,102 ,102],
    [160 ,190 ,110],
    [  0 ,220 ,220],
    [ 30 ,170 ,250],
    [128 , 64 ,128],
    [ 50 ,234 ,157],
    [ 35 ,142, 107]
]
#carla12
carla12= [

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
     [100 ,170, 145],
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





carla = carla13

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
import cv2


def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None):
    fig = Figure(figsize=(1.2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = ['{:3.2f}'.format(x) for x in tick_loc]
    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im

def colorize_np(x, cmap_name='jet', mask=None, append_cbar=False):
    if mask is not None:
        # vmin, vmax = np.percentile(x[mask], (1, 99))
        vmin = np.min(x[mask])
        vmax = np.max(x[mask])
        vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        x = np.clip(x, vmin, vmax)
        # print(vmin, vmax)
    else:
        vmin = x.min()
        vmax = x.max() + TINY_NUMBER

    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.zeros_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name)

    if append_cbar:
        x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new, cbar
# tensor
def colorize(x, cmap_name='jet', append_cbar=False, mask=None):
    x = x.numpy()
    if mask is not None:
        mask = mask.numpy().astype(dtype=np.bool)
    x, cbar = colorize_np(x, cmap_name, mask)

    if append_cbar:
        x = np.concatenate((x, np.zeros_like(x[:, :5, :]), cbar), axis=1)

    x = torch.from_numpy(x)
    return x

# misc utils
def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)

img_HWC2CHW = lambda x: x.permute(2, 0, 1)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)

def normalize(x):
    min = x.min()
    max = x.max()

    return (x - min) / ((max - min) + TINY_NUMBER)

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
# gray2rgb = lambda x: np.tile(x[:,:,np.newaxis], (1, 1, 3))
mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)
def depth2pts_outside(ray_o, ray_d, depth):
    '''
    ray_o, ray_d: [..., 3]
    depth: [...]; inverse of distance to sphere origin
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(ray_d * ray_o, dim=-1) / torch.sum(ray_d * ray_d, dim=-1)
    p_mid = ray_o + d1.unsqueeze(-1) * ray_d
    p_mid_norm = torch.norm(p_mid, dim=-1)
    ray_d_cos = 1. / torch.norm(ray_d, dim=-1)
    d2 = torch.sqrt(1. - p_mid_norm * p_mid_norm) * ray_d_cos
    p_sphere = ray_o + (d1 + d2).unsqueeze(-1) * ray_d

    rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
    rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
    phi = torch.asin(p_mid_norm)
    theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
    rot_angle = (phi - theta).unsqueeze(-1)     # [..., 1]

    # now rotate p_sphere
    # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                   torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                   rot_axis * torch.sum(rot_axis*p_sphere, dim=-1, keepdim=True) * (1.-torch.cos(rot_angle))
    p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
    pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

    # now calculate conventional depth
    depth_real = 1. / (depth + TINY_NUMBER) * torch.cos(theta) * ray_d_cos + d1
    return pts, depth_real




def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)

def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes, ignore_label):
    if (true_labels == ignore_label).all():
        return [0] * 4

    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    valid_pix_ids = true_labels != ignore_label
    predicted_labels = predicted_labels[valid_pix_ids]
    true_labels = true_labels[valid_pix_ids]

    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(number_classes)))

    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(np.float).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1))  # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask
    print(np.sum(conf_mat))
    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
    ious = np.zeros(number_classes)
    for class_id in range(number_classes):
        ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
    miou = nanmean(ious)
    miou_valid_class = np.mean(ious[exsiting_class_mask])
    return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious



def generategif(path):
    images = []
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))
    for file in files:
        img = Image.open(os.path.join(path,file))
        # draw = ImageDraw.Draw(img)
        images.append(img)
    images[0].save('ss.gif',
                   save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)
def masktorgb(path):
    files = os.listdir(os.path.join(path,"mask"))
    for file in files:
        print(file)
        img = Image.open(os.path.join(path,"mask",file))
        img = np.array(img)
        if img.ndim==3:
            img = img[:,:,0]
        im_seg = np.zeros((512, 512, 3))
        for i in range(512):
            for j in range(512):
                im_seg[i, j, :] = carla[img[i, j]][:]
        imageio.imwrite(os.path.join(path, "segcolor" , file), im_seg.astype(np.uint8))

def sort(x):
    if x[:-4].split("_")[-1] :
        return 0
    else :
        return int(x[:-4].split("_")[-1].replace("0", ""))


def renamefiles_(path1):
    files1 = os.listdir(os.path.join(path1))
    for file in files1:
        names = file[:-4].split("_")[-1]
        name = 100*int(names[0]) + 10*int(names[1])+ 1*int(names[2])
        shutil.copy(os.path.join(path1, file),os.path.join("/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset14/semantic_nerf/result-", str(name)+".png"))


def renamefiles(path1, path2):
    files1 = os.listdir(os.path.join(path1))
    files1.sort(key=lambda x: int(x[:-4]))
    files2 = os.listdir(os.path.join(path2))
    files2.sort(key=lambda x: int(x[:-4]))
    print(files2)
    for i in range(len(files1)):
        shutil.copy(os.path.join(path2,files2[i]), os.path.join("/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset14/semantic_nerf/result--",files1[i]))

def renamefiles__(path1):
    files1 = os.listdir(os.path.join(path1))
    print(files1)
    for i in range(len(files1)):
        os.rename(os.path.join(path1,files1[i]), os.path.join(path1,files1[i][5::]))


def transmask3channel(path):
    files = os.listdir(path)
    for file in files:
        image = np.zeros((512, 512, 3))
        img = Image.open(os.path.join(path,file))
        img = np.array(img)
        image[:,:,0] = img
        image[:, :, 1] = img
        image[:, :, 2] = img
        imageio.imwrite("/home/qzp/nerf/nerf++/121212/nerfplusplus-master/logs/carla84/result/"+file,image.astype(np.uint8))

def caculatemetrics():
    # iousgts = os.listdir("./data/test_result/segmantic_nerf/gt")
    # predicts = os.listdir("./data/test_result/segmantic_nerf/result")
    # gt_masks = []
    # predict_masks = []
    #
    #
    # for predict in predicts:
    #     # print(predict)
    #     # print("semantic_class__" + str(int(predict.split('.')[0].split("_")[-1])) + ".png")
    #     img = Image.open(os.path.join("./data/test_result/segmantic_nerf/gt","semantic_class__" + str(int(predict.split('.')[0].split("_")[-1])) + ".png"))
    #     img = np.array(img)
    #     img = img[:,:,0]
    #     gt_masks.append(img)
    #     img2 = Image.open(os.path.join("./data/test_result/segmantic_nerf/result/", predict))
    #     img2 = np.array(img2)
    #     predict_masks.append(img2)


    # gts = os.listdir("./data/test_result/unet_result/gt")
    # predicts = os.listdir("./data/test_result/unet_result/result")
    # gt_masks = []
    # predict_masks = []
    #
    #
    # for predict in predicts:
    #     # print(predict)
    #     # print("semantic_class__" + str(int(predict.split('.')[0].split("_")[-1])) + ".png")
    #     img = Image.open(os.path.join("./data/test_result/unet_result/gt",predict))#"semantic_class__" + str(int(predict.split('.')[0].split("_")[-1])) + ".png"))
    #     img = np.array(img)
    #     img = img[:,:,0]
    #     gt_masks.append(img)
    #     img2 = Image.open(os.path.join("./data/test_result/unet_result/result/", predict))
    #     img2 = np.array(img2)
    #
    #
    #     predict_masks.append(img2)


    # gts = os.listdir("./data/test_result/my_nerf/gt")
    # predicts = os.listdir("./data/test_result/my_nerf/result")
    # gt_masks = []
    # predict_masks = []
    #
    #
    # for predict in predicts:
    #     # print(predict)
    #     # print("semantic_class__" + str(int(predict.split('.')[0].split("_")[-1])) + ".png")
    #     img = Image.open(os.path.join("./data/test_result/my_nerf/gt",predict))#"semantic_class__" + str(int(predict.split('.')[0].split("_")[-1])) + ".png"))
    #     img = np.array(img)
    #     img = img[:,:,0]
    #     gt_masks.append(img)
    #     img2 = Image.open(os.path.join("./data/test_result/my_nerf/result/", predict))
    #     img2 = np.array(img2)
    #
    #
    #     predict_masks.append(img2)

    gtpath = "/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset9/my_nerf/gt"
    predictpath = "/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset9/my_nerf/result12"
    # gtpath = "/home/qzp/nerf/nerf++/Pytorch-UNet-master/data/ratioreal2/gt"
    # predictpath = "/home/qzp/nerf/nerf++/Pytorch-UNet-master/data/ratioreal2/30%/test/results2"


    # gtpath = "/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset8/mynerf_chaowangluo/gt"
    # predictpath = "/home/semantic_nerfqzp/nerf/nerf++/mynerf/data/test_result/dataset8/mynerf_chaowangluo/carla84"
    gts = os.listdir(gtpath)
    predicts = os.listdir(predictpath)
    gt_masks = []
    predict_masks = []

    for predict in predicts:
        # print(predict)
        # print("semantic_class__" + str(int(predict.split('.')[0].split("_")[-1])) + ".png")
        img = Image.open(os.path.join(gtpath,predict))#"semantic_class__" + str(int(predict.split('.')[0].split("_")[-1])) + ".png"))
        img = np.array(img)
        img = img[:,:,0]
        gt_masks.append(img)

        img2 = Image.open(os.path.join(predictpath, predict))
        img2 = np.array(img2)
        # img2[img2==13]=-1
        # img2[img2 == 14] = -1
        predict_masks.append(img2)


    gt_masks = np.stack(gt_masks, 0)
    predict_masks = np.stack(predict_masks, 0)
    miou_test, miou_test_validclass, total_accuracy_test, class_average_accuracy_test, ious_test = \
        calculate_segmentation_metrics(true_labels=gt_masks, predicted_labels=predict_masks,
                                       number_classes=18, ignore_label=-1)

    print("miou_test",miou_test)
    print("miou_test_calidclass", miou_test_validclass)
    print("total_accuracy_test",total_accuracy_test)
    print("class_average_accuracy_test",class_average_accuracy_test)
    print("ious_test",ious_test)


if __name__ == "__main__":
    pass
    # generategif("/home/qzp/nerf/nerf++/nerfplusplus-master/logs/carla69/render_test_-00001")

    # masktorgb("/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset13/semantic_nerf")
    # masktorgb("/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset12/unet")
    # masktorgb("/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset12/deeplabv3")

    # masktorgb("/home/qzp/nerf/nerf++/mynerf/data/test_result/zhijiage/unet")

    # renamefiles_("/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset14/semantic_nerf/result")
    # renamefiles("/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset14/semantic_nerf/gt","/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset14/semantic_nerf/result-")
    # renamefiles__("/home/qzp/nerf/nerf++/video/9/my_nerf/segcolor")
    caculatemetrics()
    # transmask3channel("/home/qzp/nerf/nerf++/121212/nerfplusplus-master/logs/carla84/mask")

    # path = "/home/qzp/nerf/nerf++/mynerf/data/test_result/dataset8/mynerf_chaowangluo/carla84/3001836.png"
    # img = Image.open(os.path.join(path))
    # img = np.array(img)
    # result = np.zeros((512,512,3),dtype= np.uint8)
    # for i in range(512):
    #     for j in range(512):
    #         if img[i,j]==13 or img[i,j]==14:
    #             result[i,j,0]=255
    #             result[i, j, 1] = 255
    #             result[i, j, 2] = 255
    #
    # result= Image.fromarray(result)
    # result.show()






