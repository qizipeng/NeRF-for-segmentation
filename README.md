# NeRF-for-segmentation
A novel NeRF++ based method for remote sensing scene segmetation!

## Abstract:

Remote sensing image segmentation, as a challenging but fundamental task, has drawn increasing attention in
the remote sensing field. Recent advances in deep learning have
greatly boosted research on this task. However, the existing deep
learning-based segmentation methods heavily rely on a large
amount of pixel-wise labeled training data, and the labeling
process is time-consuming and labor-intensive. In this paper, we
focus on the scenario that leverages the 3D structure of multi-view
images and a limited number of annotations to generate accurate
novel view segmentation. Under this scenario, we propose a
novel method for remote sensing image segmentation based on
implicit 3D scene representation, which generates arbitrary-view
segmentation output from limited segmentation annotations. The
proposed method employs a two-stage training strategy. In the
first stage, we optimize the implicit neural representations of a 3D
scene and encode their multi-view images into a neural radiance
field. In the second stage, we transform the scene color attribute
into semantic labels and propose a ray-convolution network to
aggregate local 3D consistency cues across different locations.
We also design a color-radiance network to help our method
generalize to unseen views. Experiments on both synthetic and
real-world data suggest that our method significantly outperforms
deep convolutional networks (CNN)-based methods and other
view synthesis-based methods. We also show that the proposed
method can be applied as a novel data augmentation approach
that benefits CNN-based segmentation method.

## The problem we want to solve:

![Image text](https://github.com/qizipeng/NeRF-for-segmentation/blob/master/Figs/problem.png)

## The method we proposed:
![Image text](https://github.com/qizipeng/NeRF-for-segmentation/blob/master/Figs/method.png)

In our paper, a two-stage implicit neural field optimization
method for remote sensing scene segmentation is proposed.
An overall of our method is shown in the above Fig. 1) In the
implicit representation stage, we follow the NeRF pipeline
and feed the 3D coordinates and the view angle of the 3D
locations to two multilayer perceptrons, i.e., a location-MLP
and a view-MLP. In this way, the coordinate-dependent density
attribute and view-dependent color attribute can be generated
for each 3D location. 2) In the semantic segmentation stage,
we effectively used the spatial information extracted from the
above stage to generate segmentation results. Specifically, we
freeze the weights of the Location-MLP so that the spatial
information implied by the density attribute can be reused.
A prediction head named Seg-MLP is then introduced to
generate semantic features based on density features. Considering 
the insufficiency of the annotation of limited views, we
also propose a ray-convolution network, where pixel features
are extracted and fused with semantic features. This enables
the model to properly employ spatial information and pixel
information to produce a more accurate segmentation output.

## The datsets we constructed:
![Image text](https://github.com/qizipeng/NeRF-for-segmentation/blob/master/Figs/rgb.png) ![Image text](https://github.com/qizipeng/NeRF-for-segmentation/blob/master/Figs/seg.png)

To quantitatively evaluate our method, we build a multi-
view aerial remote sensing dataset named Carla-MVS based on
the well-known Carla simulation platform. We also conduct qualitative experiments on real-world
remote sensing images from Google Earth.