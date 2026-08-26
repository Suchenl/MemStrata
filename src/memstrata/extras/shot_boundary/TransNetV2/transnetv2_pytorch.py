# TransNetV2 implementation
import torch
import torch.nn as nn
import torch.nn.functional as functional


class TransNetV2(nn.Module):

    def __init__(self,
                 F=16, L=3, S=2, D=1024,
                 use_many_hot_targets=True,
                 use_frame_similarity=True,
                 use_color_histograms=True,
                 use_mean_pooling=False,
                 dropout_rate=0.5,
                 use_convex_comb_reg=False,
                 use_resnet_features=False,
                 use_resnet_like_top=False,
                 frame_similarity_on_last_layer=False):
        super(TransNetV2, self).__init__()

        if use_resnet_features or use_resnet_like_top or use_convex_comb_reg or frame_similarity_on_last_layer:
            raise NotImplementedError(
                "Some options not implemented in Pytorch version of Transnet!")

        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.)] +
            [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4,
                            n_blocks=S, filters=F * 2 ** i) for i in range(1, L)]
        )

        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]), lookup_window=101, output_dim=128, similarity_dim=128, use_bias=True
        ) if use_frame_similarity else None
        self.color_hist_layer = ColorHistograms(
            lookup_window=101, output_dim=128
        ) if use_color_histograms else None

        self.dropout = nn.Dropout(
            dropout_rate) if dropout_rate is not None else None

        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6
        if use_frame_similarity:
            output_dim += 128
        if use_color_histograms:
            output_dim += 128

        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        self.eval()

    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8, \
            "incorrect input type and/or shape"
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.)

        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = x.reshape(x.shape[0], x.shape[1], -1)

        if self.frame_sim_layer is not None:
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        if self.color_hist_layer is not None:
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        x = self.fc1(x)
        x = functional.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)

        one_hot = self.cls_layer1(x)

        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot


class StackedDDCNNV2(nn.Module):

    def __init__(self,
                 in_filters,
                 n_blocks,
                 filters,
                 shortcut=True,
                 pool_type="avg",
                 stochastic_depth_drop_prob=0.0):
        super(StackedDDCNNV2, self).__init__()

        assert pool_type == "max" or pool_type == "avg"

        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList([
            DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters, octave_conv=False,
                          dilation_rate=2 ** (i - 1)) for i in range(1, n_blocks + 1)
        ])

        self.pool = nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2]) if pool_type == "max" else \
            nn.AvgPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2])

    def forward(self, x):
        for block in self.DDCNN:
            x = block(x)
        return self.pool(x)


class DilatedDCNNV2(nn.Module):

    def __init__(self, in_filters, filters, octave_conv=False, dilation_rate=1):
        super(DilatedDCNNV2, self).__init__()

        self.conv1 = nn.Conv3d(in_filters, filters, kernel_size=[3, 3, 3], padding=[1, 1, 1], bias=False)
        self.bn1 = nn.BatchNorm3d(filters)

        self.conv2 = nn.Conv3d(filters, filters, kernel_size=[3, 3, 3], padding=[dilation_rate, 1, 1],
                               dilation=[dilation_rate, 1, 1], bias=False)
        self.bn2 = nn.BatchNorm3d(filters)

        self.conv3 = nn.Conv3d(filters, filters, kernel_size=[3, 3, 3], padding=[1, dilation_rate, dilation_rate],
                               dilation=[1, dilation_rate, dilation_rate], bias=False)
        self.bn3 = nn.BatchNorm3d(filters)

        self.conv4 = nn.Conv3d(filters, filters, kernel_size=[3, 3, 3], padding=[1, 1, 1], bias=False)
        self.bn4 = nn.BatchNorm3d(filters)

    def forward(self, x):
        x = functional.relu(self.bn1(self.conv1(x)))
        x = functional.relu(self.bn2(self.conv2(x)))
        x = functional.relu(self.bn3(self.conv3(x)))
        x = functional.relu(self.bn4(self.conv4(x)))
        return x


class FrameSimilarity(nn.Module):

    def __init__(self, in_filters, lookup_window=101, output_dim=128, similarity_dim=128, use_bias=True):
        super(FrameSimilarity, self).__init__()

        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)
        self.lookup_window = lookup_window

    def forward(self, block_features):
        features = []
        for feature in block_features:
            feature = torch.mean(feature, dim=[3, 4])
            feature = feature.permute(0, 2, 1)
            features.append(feature)
        x = torch.cat(features, 2)
        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_steps, _ = x.shape
        similarities = torch.zeros(batch_size, time_steps, self.lookup_window, device=x.device)

        for i in range(self.lookup_window):
            offset = i - self.lookup_window // 2
            if offset < 0:
                similarities[:, -offset:, i] = torch.sum(x[:, -offset:] * x[:, :offset], dim=2)
            elif offset > 0:
                similarities[:, :-offset, i] = torch.sum(x[:, :-offset] * x[:, offset:], dim=2)
            else:
                similarities[:, :, i] = 1.0

        x = self.fc(similarities)
        x = functional.relu(x)
        return x


class ColorHistograms(nn.Module):

    def __init__(self, lookup_window=101, output_dim=128):
        super(ColorHistograms, self).__init__()

        self.fc = nn.Linear(lookup_window * 3, output_dim)
        self.lookup_window = lookup_window

    def forward(self, inputs):
        # inputs shape [B, T, H, W, 3]
        batch_size, time_steps, height, width, _ = inputs.shape
        inputs = inputs.float()

        # Compute histograms
        hists = []
        for channel in range(3):
            channel_inputs = inputs[:, :, :, :, channel]
            bins = torch.linspace(0, 255, 33, device=inputs.device)
            hist = torch.zeros(batch_size, time_steps, 32, device=inputs.device)
            for i in range(32):
                hist[:, :, i] = torch.sum((channel_inputs >= bins[i]) & (channel_inputs < bins[i + 1]), dim=[2, 3])
            hist = functional.normalize(hist, p=1, dim=2)
            hists.append(hist)

        x = torch.cat(hists, 2)
        # Compute intersections
        similarities = torch.zeros(batch_size, time_steps, self.lookup_window * 3, device=inputs.device)
        for i in range(self.lookup_window):
            offset = i - self.lookup_window // 2
            for channel in range(3):
                hist = hists[channel]
                idx = i * 3 + channel
                if offset < 0:
                    similarities[:, -offset:, idx] = torch.sum(torch.min(hist[:, -offset:], hist[:, :offset]), dim=2)
                elif offset > 0:
                    similarities[:, :-offset, idx] = torch.sum(torch.min(hist[:, :-offset], hist[:, offset:]), dim=2)
                else:
                    similarities[:, :, idx] = 1.0

        x = self.fc(similarities)
        x = functional.relu(x)
        return x
