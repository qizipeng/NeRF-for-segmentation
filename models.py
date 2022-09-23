import torch
import torch.nn as nn
from collections import OrderedDict
from utils import depth2pts_outside, HUGE_NUMBER, TINY_NUMBER
import matplotlib.pyplot as plot
import numpy as np

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# model = Model()
#
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model, device_ids=[0,1,2])### need to change
#
# model.to(device)

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                       log_sampling=True, include_input=True,
                       periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out

class MLPNet(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_viewdirs=3,
                 skips=[4], use_viewdirs=False, IsSegmantic = True, pointclass = True, Finetune = True, nclass = 18):
        '''
        :param D: network depth
        :param W: network width
        :param input_ch: input channels for encodings of (x, y, z)
        :param input_ch_viewdirs: input channels for encodings of view directions
        :param skips: skip connection in network
        :param use_viewdirs: if True, will use the view directions as input
        '''
        super().__init__()

        self.input_ch = input_ch
        self.input_ch_viewdirs = input_ch_viewdirs
        self.use_viewdirs = use_viewdirs
        self.skips = skips
        self.IsSegmantic = IsSegmantic
        self.pointclass = pointclass
        self.Finetune = Finetune
        self.base_layers = []
        self.nclass = nclass
        dim = self.input_ch
        for i in range(D):
            self.base_layers.append(
                nn.Sequential(nn.Linear(dim, W), nn.ReLU())
            )
            dim = W
            if i in self.skips and i != (D-1):      # skip connection after i^th layer
                dim += input_ch
        self.base_layers = nn.ModuleList(self.base_layers)
        # self.base_layers.apply(weights_init)        # xavier init
        if self.IsSegmantic == True:
            self.freeze(self.base_layers) ### freeze

        sigma_layers = [nn.Linear(dim, 1), ]       # sigma must be positive
        self.sigma_layers = nn.Sequential(*sigma_layers)
        # self.sigma_layers.apply(weights_init)      # xavier init
        if self.IsSegmantic == True:
            self.freeze(self.sigma_layers)  ### freeze

        ###ray 方向卷积
        if self.IsSegmantic ==True:
            self.conv1_1 = nn.Conv2d(in_channels=dim, out_channels=dim,kernel_size=[1,3],padding=[0,1])
            # self.conv1_2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=[1, 3], padding=[0, 1])
            self.BN1 = nn.BatchNorm2d(dim)
        # rgb color
        rgb_layers = []
        base_remap_layers = [nn.Linear(dim, 256), ]
        self.base_remap_layers = nn.Sequential(*base_remap_layers)
        # self.base_remap_layers.apply(weights_init)
        if self.IsSegmantic == False:
            if self.use_viewdirs:
                dim = 256 + self.input_ch_viewdirs
            else:
                dim = 256
        else:
            dim = 256
        for i in range(1):
            rgb_layers.append(nn.Linear(dim, W // 2))
            rgb_layers.append(nn.ReLU())
            dim = W // 2
        if self.IsSegmantic == False:
            rgb_layers.append(nn.Linear(dim, 3)) ###分20类
            rgb_layers.append(nn.Sigmoid())     # rgb values are normalized to [0, 1]
            self.rgb_layers = nn.Sequential(*rgb_layers)
        elif self.IsSegmantic == True:
            self.conv2_1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=[1, 3], padding=[0, 1])
            self.conv2_2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=[1, 5], padding=[0, 2])
            # self.conv2_2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=[1, 3], padding=[0, 1])
            # self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=[1, 5], padding=[0, 2])
            # self.conv4 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=[1, 7], padding=[0, 3])
            self.rgb_layers_ = nn.Sequential(*rgb_layers)
            self.new_Linear = nn.Linear(dim, self.nclass)
            # self.conv5 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=[1, 3], padding=[0, 1])
            # self.conv5 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=[1, 3], padding=[0, 1])
            # rgb_layers.append(self.new_Linear) ###改名字

            # self.rgb_layers.apply(weights_init)
            self.BN2 = nn.BatchNorm2d(dim)
            self.Relu = nn.ReLU()

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
    def forward(self, input, m = 0):
        '''
        :param input: [..., input_ch+input_ch_viewdirs]
        :return [..., 4]
        '''
        input_pts = input[..., :self.input_ch]
        base = self.base_layers[0](input_pts)
        # print(base)
        for i in range(len(self.base_layers)-1):
            if i in self.skips:
                base = torch.cat((input_pts, base), dim=-1)
            base = self.base_layers[i+1](base)

        sigma = self.sigma_layers(base)
        sigma = torch.abs(sigma)
        # if self.IsSegmantic == True:
        #     base = torch.unsqueeze(base,0)
        #     base = base.permute(0,3,1,2)
        #     base = self.conv1_1(base)
        #     base = self.BN1(base)
        #     base = self.Relu(base)
        #     # base = self.conv1_2(base)
        #     base = base.permute(0, 2, 3, 1)
        #     base = torch.squeeze(base,0)

        base_remap = self.base_remap_layers(base)
        if self.IsSegmantic == False:
            if self.use_viewdirs:
                input_viewdirs = input[..., -self.input_ch_viewdirs:]
                rgb = self.rgb_layers(torch.cat((base_remap, input_viewdirs), dim=-1))
            else:
                rgb = self.rgb_layers(base_remap)
            ret = OrderedDict([('rgb', rgb),
                               ('sigma', sigma.squeeze(-1))])
            return ret
        if self.IsSegmantic == True:
            if self.Finetune == True:
                rgb = self.rgb_layers_(base_remap)
                rgb = self.new_Linear(rgb)
                ret = OrderedDict([('rgb', rgb),
                                   ('sigma', sigma.squeeze(-1))])
                return ret
            rgb = self.rgb_layers_(base_remap)
            rgb = torch.unsqueeze(rgb, 0)
            rgb = rgb.permute(0, 3, 1, 2)
            if m == 0:
                rgb = self.conv2_1(rgb)
            else:
                rgb = self.conv2_2(rgb)
            rgb = self.Relu(rgb)
            rgb = rgb.permute(0, 2, 3, 1)
            rgb = torch.squeeze(rgb, 0)
            # rgb = self.conv2_2(rgb)
            # rgb = self.BN2(rgb)
            # rgb = self.Relu(rgb)
            # rgb = self.conv3(rgb)
            # rgb = self.conv4(rgb)
            # rgb = rgb.permute(0, 2, 3, 1)
            # rgb = torch.squeeze(rgb, 0)

            if self.pointclass == False:
                rgb = self.new_Linear(rgb)

            # rgb = torch.unsqueeze(rgb, 0)
            # rgb = rgb.permute(0, 3, 1, 2)
            # rgb = self.conv5(rgb)
            # rgb = rgb.permute(0, 2, 3, 1)
            # rgb = torch.squeeze(rgb, 0)
            # if m==1:
            #     s = sigma.squeeze(-1)
            #     s = s.cpu().numpy()
            #     for i in range(100):
            #         s_ = s[i,::]
            #     #     s1 = s[i,0:-2]
            #     #     s2 = s[i,1:-1]
            #     #     s_ = s2-s1
            #     #     s_ = s_.cpu().numpy()
            #     #     s_1 = np.where(s_>0)
            #     #     mean = np.mean(s_1)
            #     #     s__ = s_[s_>mean]
            #     #     print(len(s__))
            #
            #         # va , index= torch.topk(s,10,1)
            #         # va = s.gather(1,index)
            #         # print(va.shape)
            #         plot.cla()
            #         plot.plot(range(0, len(s_)), s_)
            #         plot.savefig("./sigma/sigma_{}_{}.png".format(m,i))

            ret = OrderedDict([('rgb', rgb),
                                ('sigma', sigma.squeeze(-1))])
            return ret

class Nerf(nn.Module):
    def __int__(self, args):
        pass

class cnn_network(nn.Module):
    def __init__(self):
        super(cnn_network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.BN1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.BN3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.BN4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.BN2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.BN3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.BN4(x)
        x = self.relu4(x)

        return x

class Nerfplus(nn.Module):
    def __init__(self,args):
        super(Nerfplus, self).__init__()
        self.fg_embedder_position = Embedder(input_dim=3,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.fg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.fg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.fg_embedder_position.out_dim,
                             input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs, IsSegmantic= args.IsSegmantic, pointclass=args.pointclass,Finetune=args.Finetune,nclass=args.nclass)
        # background; bg_pt is (x, y, z, 1/r)
        self.bg_embedder_position = Embedder(input_dim=4,
                                             max_freq_log2=args.max_freq_log2 - 1,
                                             N_freqs=args.max_freq_log2)
        self.bg_embedder_viewdir = Embedder(input_dim=3,
                                            max_freq_log2=args.max_freq_log2_viewdirs - 1,
                                            N_freqs=args.max_freq_log2_viewdirs)
        self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
                             input_ch=self.bg_embedder_position.out_dim,
                             input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
                             use_viewdirs=args.use_viewdirs, IsSegmantic= args.IsSegmantic, pointclass=args.pointclass,Finetune=args.Finetune,nclass=args.nclass)
        self.IsSegmantic = args.IsSegmantic
        self.Finetune = args.Finetune
        self.pointclass = args.pointclass
        if self.IsSegmantic == True:
            self.cnn_net = cnn_network()
            self.fg_fc_ = nn.Linear(128+args.color_channel,args.nclass)
            self.bg_fc_ = nn.Linear(128+args.color_channel, args.nclass)
            self.relu = nn.ReLU()
            self.color_class1 = nn.Linear(3,16)
            self.color_class2_ = nn.Linear(16, args.color_channel) ###64
            self.color_class3_ = nn.Linear(args.color_channel, args.nclass)

    def forward(self, ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, color = None, m = 0):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals, bg_z_vals: [..., N_samples]
        :return
        '''
        # print(ray_o.shape, ray_d.shape, fg_z_max.shape, fg_z_vals.shape, bg_z_vals.shape)
        color_class = None
        if self.IsSegmantic and self.pointclass and self.Finetune==False:
            color_class = self.color_class1(color)
            color_class = self.relu(color_class)
            color_class = self.color_class2_(color_class)
            color_class_ = self.relu(color_class)
            color_class = self.color_class3_(color_class_)
        ray_d_norm = torch.norm(ray_d, dim=-1, keepdim=True)  # [..., 1]
        viewdirs = ray_d / ray_d_norm      # [..., 3]
        dots_sh = list(ray_d.shape[:-1])

        ######### render foreground
        N_samples = fg_z_vals.shape[-1]
        fg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        fg_pts = fg_ray_o + fg_z_vals.unsqueeze(-1) * fg_ray_d
        input = torch.cat((self.fg_embedder_position(fg_pts),
                           self.fg_embedder_viewdir(fg_viewdirs)), dim=-1)
        if self.IsSegmantic:
            fg_raw = self.fg_net(input, m)
        else:
            fg_raw = self.fg_net(input)
        # alpha blending
        fg_dists = fg_z_vals[..., 1:] - fg_z_vals[..., :-1]
        # account for view directions
        fg_dists = ray_d_norm * torch.cat((fg_dists, fg_z_max.unsqueeze(-1) - fg_z_vals[..., -1:]), dim=-1)  # [..., N_samples]
        fg_alpha = 1. - torch.exp(-fg_raw['sigma'] * fg_dists)  # [..., N_samples]
        T = torch.cumprod(1. - fg_alpha + TINY_NUMBER, dim=-1)   # [..., N_samples]
        bg_lambda = T[..., -1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]
        fg_weights = fg_alpha * T     # [..., N_samples]
        fg_rgb_map = torch.sum(fg_weights.unsqueeze(-1) * fg_raw['rgb'], dim=-2)  # [..., 3]
        if self.IsSegmantic and self.pointclass:
            fg_rgb_map = self.fg_fc_(torch.cat((fg_rgb_map, color_class_), dim=-1))

        fg_depth_map = torch.sum(fg_weights * fg_z_vals, dim=-1)     # [...,]

        # render background
        N_samples = bg_z_vals.shape[-1]
        bg_ray_o = ray_o.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_ray_d = ray_d.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_viewdirs = viewdirs.unsqueeze(-2).expand(dots_sh + [N_samples, 3])
        bg_pts, _ = depth2pts_outside(bg_ray_o, bg_ray_d, bg_z_vals)  # [..., N_samples, 4]
        input = torch.cat((self.bg_embedder_position(bg_pts),
                           self.bg_embedder_viewdir(bg_viewdirs)), dim=-1)
        # near_depth: physical far; far_depth: physical near
        input = torch.flip(input, dims=[-2,])
        bg_z_vals = torch.flip(bg_z_vals, dims=[-1,])           # 1--->0
        bg_dists = bg_z_vals[..., :-1] - bg_z_vals[..., 1:]
        bg_dists = torch.cat((bg_dists, HUGE_NUMBER * torch.ones_like(bg_dists[..., 0:1])), dim=-1)  # [..., N_samples]
        if self.IsSegmantic:
            bg_raw = self.bg_net(input,m)
        else:
            bg_raw = self.bg_net(input)


        bg_alpha = 1. - torch.exp(-bg_raw['sigma'] * bg_dists)  # [..., N_samples]
        # Eq. (3): T
        # maths show weights, and summation of weights along a ray, are always inside [0, 1]
        T = torch.cumprod(1. - bg_alpha + TINY_NUMBER, dim=-1)[..., :-1]  # [..., N_samples-1]
        T = torch.cat((torch.ones_like(T[..., 0:1]), T), dim=-1)  # [..., N_samples]
        bg_weights = bg_alpha * T  # [..., N_samples]
        bg_rgb_map = torch.sum(bg_weights.unsqueeze(-1) * bg_raw['rgb'], dim=-2)  # [..., 3]

        if self.IsSegmantic and self.pointclass:
            bg_rgb_map = self.bg_fc_(torch.cat((bg_rgb_map, color_class_), dim=-1))

        bg_depth_map = torch.sum(bg_weights * bg_z_vals, dim=-1)  # [...,]

        # composite foreground and background
        bg_rgb_map = bg_lambda.unsqueeze(-1) * bg_rgb_map
        bg_depth_map = bg_lambda * bg_depth_map
        rgb_map = fg_rgb_map + bg_rgb_map



        ret = OrderedDict([('rgb', rgb_map),            # loss
                           ("color_class", color_class),
                           ('fg_weights', fg_weights),  # importance sampling
                           ('bg_weights', bg_weights),  # importance sampling
                           ('fg_rgb', fg_rgb_map),      # below are for logging
                           ('fg_depth', fg_depth_map),
                           ('bg_rgb', bg_rgb_map),
                           ('bg_depth', bg_depth_map),
                           ('bg_lambda', bg_lambda)])
        return ret

class Model(nn.Module):
    def __init__(self, args ,model = "Nerf++"):
        super(Model, self).__init__()
        self.args = args
        if model =="Nerf":
            self.net = Nerf(args=self.args)
        if model == "Nerf++":
            self.net = Nerfplus(args=self.args)
    def forward(self,ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, color = None, m = 0):
        ret = self.net(ray_o, ray_d, fg_z_max, fg_z_vals, bg_z_vals, color, m)
        return ret


