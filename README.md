# NeRF-for-segmentation
A novel NeRF++ based method for remote sensing scene segmetation!

Abstract:

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

The problem we want to solve:

！[image](./Figs/problem.png)
