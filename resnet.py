import torch.nn as nn

norm_cfg = {
    'BN': ('bn', nn.BatchNorm2d),
    'SyncBN': ('bn', nn.SyncBatchNorm),
    'GN': ('gn', nn.GroupNorm),
}

def build_norm_layer(cfg, num_features, postfix=''):
    """Build normalization layer.

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, num_groups=min(cfg_['num_groups'], num_features//4))

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 norm_cfg=dict(type='BN'),
                 activation='relu'):
        super(BasicBlock, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'mish':
            self.activation = nn.Mish(inplace=True)
        else:
            raise NotImplementedError("Invalid activation function")

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 norm_cfg=dict(type='BN'),
                 activation='relu'):
        """Bottleneck block for ResNet."""
        super(Bottleneck, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'mish':
            self.activation = nn.Mish(inplace=True)
        else:
            raise NotImplementedError("Invalid activation function")

        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.activation(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)

        out = self.activation(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   norm_cfg=dict(type='BN'),
                   activation='relu',
                   **kwargs):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    block_kwargs = dict(
            norm_cfg=norm_cfg,
            )
    block_kwargs.update(kwargs)
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            activation=activation,
            **block_kwargs))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                activation=activation,
                **block_kwargs))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.

    Example:
        >>> from openselfsup.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 base_n_channels=64,
                 per_stage_wide=[1, 1, 1, 1],
                 per_stage_block=None,
                 small_image=False,
                 activation='mish',
                 ):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.base_n_channels = base_n_channels
        self.inplanes = self.base_n_channels
        self.per_stage_wide = per_stage_wide

        self.small_image = small_image
        self.activation = activation

        if per_stage_block is None:
            self.block = [self.block] * len(self.stage_blocks)
        else:
            # A list of strings, either 'BasicBlock' or 'Bottleneck'
            assert len(per_stage_block) == len(self.stage_blocks)
            self.block = []
            for block_name in per_stage_block:
                if block_name == 'BasicBlock':
                    curr_block = BasicBlock
                elif block_name == 'Bottleneck':
                    curr_block = Bottleneck
                else:
                    raise NotImplementedError
                self.block.append(curr_block)

        if self.small_image: # When using 32x32 images in CIFAR
            self._make_stem_layer(in_channels, kernel_size=3, stride=1, padding=1)
        else:
            self._make_stem_layer(in_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = self.base_n_channels * 2**i
            planes *= per_stage_wide[i]
            res_layer = make_res_layer(
                self.block[i],
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                norm_cfg=norm_cfg,
                activation=activation)
            self.inplanes = planes * self.block[i].expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._freeze_stages()

        self.feat_dim = self.block[-1].expansion * self.base_n_channels * 2**(
            len(self.stage_blocks) - 1) * per_stage_wide[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.norm2.weight, 0)
                

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, kernel_size=7, stride=2, padding=3):
        self.conv1 = nn.Conv2d(
            in_channels,
            self.base_n_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, self.base_n_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)

        if self.activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif self.activation == 'mish':
            self.activation = nn.Mish(inplace=True)
        else:
            raise NotImplementedError("Invalid activation function")

        if not self.small_image:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x, pre_avg_pool=False):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)  # r50: 64x128x128
        if not self.small_image:
            x = self.maxpool(x)  # r50: 64x56x56
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
        if pre_avg_pool:
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
