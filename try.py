import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import torchaudio
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import transformers
import wandb
import json
import torch.nn as nn
import math
import librosa
import torch.nn as nn
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import os

from dataset.dcase24 import get_training_set, get_test_set, get_eval_set
from helpers.init import worker_init_fn
from models.baseline import get_model
from helpers.utils import mixstyle
from helpers import nessi
from thop import profile, clever_format




class ChannelAttention(nn.Module):
    """Channel Attention as proposed in the paper 'Convolutional Block Attention Module'"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
 
class SpatialAttention(nn.Module):
    """Spatial Attention as proposed in the paper 'Convolutional Block Attention Module'"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #print("Spatial X : {}".format(x.shape))
        x = self.conv1(x)
        return self.sigmoid(x)
 
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
   
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
 
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
 
    """
 
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input_tensor):
        """
 
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
 
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
 
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor
 
 
class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """
 
    def __init__(self, num_channels):
        """
 
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, input_tensor, weights=None):
        """
 
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()
 
        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)
 
        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor
 
 
class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """
 
    def __init__(self, num_channels, reduction_ratio=4):
        """
 
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)
 
    def forward(self, input_tensor):
        """
 
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.add(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor
 
class simam_module(torch.nn.Module):
    """
    Re-implementation of the simple attention module (SimAM)
    """
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()
 
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
 
    @staticmethod
    def get_module_name():
        return "simam"
 
    def forward(self, x):
 
        b, c, h, w = x.size()
 
        n = w * h - 1
 
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
 
        return x * self.activaton(y)


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    This module applies both channel and spatial attention to the input feature map.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        """
        Initializes the CBAM module.

        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for channel attention. Default is 16.
            kernel_size (int): Kernel size for spatial attention. Default is 7.
        """
        super(CBAMBlock, self).__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the CBAM module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying channel and spatial attention.
        """
        # Apply Channel Attention
        ca = self.channel_attention(x)
        x = x * ca

        # Apply Spatial Attention
        sa = self.spatial_attention(torch.cat([torch.mean(x, dim=1, keepdim=True),
                                              torch.max(x, dim=1, keepdim=True)[0]], dim=1))
        x = x * sa

        return x
    


class ConvBlock(nn.Module):
    """
    A Convolutional Block that performs a convolution followed by batch normalization 
    and a ReLU activation.
    """
    def __init__(self, in_channels, out_channels, 
                 kernel_size=(3, 3), stride=(1, 1), 
                 padding=(1, 1), add_bias=False):
        """
        Initializes the ConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Size of the convolutional kernel. Default is (3, 3).
            stride (tuple): Stride of the convolution. Default is (1, 1).
            padding (tuple): Zero-padding added to both sides of the input. Default is (1, 1).
            add_bias (bool): If True, adds a learnable bias to the output. Default is False.
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              bias=add_bias)
        self.bn = nn.BatchNorm2d(out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the convolutional and linear layers.
        """
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initializes weights based on the layer type.

        Args:
            m (nn.Module): The module to initialize.
        """
        if isinstance(m, nn.Linear):
            # Xavier Uniform initialization for Linear layers
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            # Kaiming Uniform initialization for Conv2d layers
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm weights and biases
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass through the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution, batch normalization, and ReLU activation.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu_(x)
        return x
    

class CBAMCNN(nn.Module):
    """
    A 3-Layer Convolutional Neural Network.
    """

    def __init__(self, num_classes=10, verbose=False):
        """
        Initializes the CBAMCNN model. Don't need to change default arguments unless I got the num_classes
        wrong.

        Args:
            num_classes (int): Number of output classes. Default is 10.
            verbose (bool): If True, prints debug statements during forward pass. Default is False.
        """

        super(CBAMCNN, self).__init__()
        self.verbose = verbose  # Toggle for debug statements
        
        # Here I am defining the model layers in sequential order (i.e. the order I will pass my input through)

        self.conv1 = ConvBlock(in_channels=1, out_channels=16)
        self.attention1 = SpatialSELayer(num_channels=16) # Replace w/ other Attention Modules if needed
        self.maxpool1 = nn.MaxPool2d((4,4))

        self.conv2 = ConvBlock(in_channels=16, out_channels=24,
                               kernel_size=(5,5), padding="same")
        self.attention2 = SpatialSELayer(num_channels=24) # Replace w/ other Attention Modules if needed
        self.maxpool2 = nn.MaxPool2d((2,4))
        self.dropout1 = nn.Dropout(p=0.2)
        
        self.conv3 = ConvBlock(in_channels=24, out_channels=32,
                               kernel_size=(7,7), padding="same")
        self.attention3 = SpatialSELayer(num_channels=32) # Replace w/ other Attention Modules if needed
        self.maxpool3 = nn.MaxPool2d((2,4))
        
        # Fully Connected Layers
        self.fcdropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=512,
                             out_features=num_classes)


    def forward(self, x):
        """
        Defines the forward pass of the CBAMCNN model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 1, height, width).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """

        # First Convolutional Block
        x = self.conv1(x)
        if self.verbose: 
            print("After conv1 : {}".format(x.shape))
        x = self.attention1(x)
        if self.verbose:
            print("After Attention Module 1 : {}".format(x.shape))
        x = self.maxpool1(x)
        if self.verbose: 
            print("After maxpool1 : {}".format(x.shape))

        # Second Convolutional Block
        x = self.conv2(x)
        if self.verbose:
            print("After conv2 : {}".format(x.shape))
        x = self.attention2(x)
        if self.verbose:
            print("After Attention Module 2 : {}".format(x.shape))
        x = self.maxpool2(x)
        if self.verbose:
            print("After maxpool2 : {}".format(x.shape))
        x = self.dropout1(x)

        # Third Convolutional Block
        x = self.conv3(x)
        if self.verbose: 
            print("After conv3 : {}".format(x.shape))
        x = self.attention3(x)
        if self.verbose:
            print("After Attention Module 3 : {}".format(x.shape))
        x = self.maxpool3(x)
        if self.verbose: 
            print("After maxpool3 : {}".format(x.shape))

        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        if self.verbose: 
            print("After x flatten : {}".format(x.shape))

        # Fully Connected Layers
        x = self.fcdropout(x)
        x = self.fc1(x)
        if self.verbose:
            print("Final X : {}".format(x.shape))
        
        return x


class PLModule(pl.LightningModule):
    def __init__(self, config, model_config):
        super().__init__()
        self.config = config
        self.model_config = model_config

        # module for resampling waveforms on the fly
        resample = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.sample_rate
        )
        get_model_fn = model_config['model_fn']
        self.model = get_model_fn(**model_config["net"])
        
        # module to preprocess waveforms into log mel spectrograms
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.window_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max
        )
        
        mel_teacher = torchaudio.transforms.MelSpectrogram(
            sample_rate=32000,
            n_fft=1024,
            win_length=800,
            hop_length=320,
            n_mels=128,
            f_min=0,
            f_max=None
        )

        freqm = torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True)
        timem = torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)

        self.mel = torch.nn.Sequential(
            resample,
            mel
        )

        self.mel_teacher = torch.nn.Sequential(
            resample,
            mel_teacher
        )

        self.mel_augment = torch.nn.Sequential(
            freqm,
            timem
        )
        self.model = CBAMCNN()  # Replace with own model #TODO 

        if self.config.use_teacher:
            # Here we assume that the teacher model has the same architecture as used during teacher training.
            # For instance, if you're using a PaSST teacher, import the teacher model definition:
            from passt import get_model as get_passt_teacher
            from cp_resnet import get_model as get_cp_resnet_teacher
            
             # For the CP‑ResNet teacher, remove the "input_fdim" parameter:
            cp_resnet_net_config = self.model_config["net"].copy()
            cp_resnet_net_config.pop("input_fdim", None)  # Remove if it exists
            cp_resnet_net_config.pop("s_patchout_t", None)  # Remove this key for CP‑ResNet
            cp_resnet_net_config.pop("s_patchout_f", None)  # And remove this too, if not needed

            
            # Instantiate the teacher model with appropriate parameters:
            self.teacher_model_1 = get_passt_teacher(**self.model_config["net"])
            self.teacher_model_2 = get_cp_resnet_teacher(**cp_resnet_net_config)

            # Load the pre-trained teacher weights:
            self.teacher_model_1.load_state_dict(torch.load(self.config.teacher_checkpoint_1, map_location='cpu'))
            self.teacher_model_2.load_state_dict(torch.load(self.config.teacher_checkpoint_2, map_location='cpu'))

            # Set the teacher to evaluation mode and freeze its parameters:
            self.teacher_model_1.eval()
            for param in self.teacher_model_1.parameters():
                param.requires_grad = False
            self.teacher_model_2.eval()
            for param in self.teacher_model_2.parameters():
                param.requires_grad = False
                
            if str(self.config.precision) in ["16", "16.0"]:
                self.teacher_model_1.half()
                self.teacher_model_2.half()    
            print("Both Teachers Initiated!")
        else:
            self.teacher_model_1 = None
            self.teacher_model_2 = None
            print("No Teacher at all!")
            
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        # pl 2 containers:
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x) # Convert raw waveform into spectrogram

        if self.training: # IF training, we want to apply augmentations
            x = self.mel_augment(x) # Apply augmentations to the mel spec
            #x = self.freqmix(x)
        x = (x + 1e-5).log()
        #print("X mel : {}".format(x.shape))
    
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, files, labels, devices, cities = train_batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(self.device)
        bs = labels.size(0)  # Cache batch size
        
        x_teacher = self.mel_teacher(x) # Convert raw audio into MelSpec for Teacher Model
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if self.config.mixstyle_p > 0:
            # frequency mixstyle
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)

        # Student model forward pass
        student_logits = self.model(x)
        loss_ce = F.cross_entropy(student_logits, labels)
        
        # If a teacher model is provided, compute the distillation loss.
        if self.teacher_model_1 is not None and self.teacher_model_2 is not None:
            # Use no_grad to ensure teacher is not updated.
            with torch.no_grad():
                teacher_logits_1 = self.teacher_model_1(x_teacher)
                teacher_logits_2 = self.teacher_model_2(x_teacher)
            # Combine the teacher ouputs (average them)
            if isinstance(teacher_logits_1, tuple):
                teacher_logits_1 = teacher_logits_1[0]
            if isinstance(teacher_logits_2, tuple):
                teacher_logits_2 = teacher_logits_2[0]
            teacher_logits= (teacher_logits_1 + teacher_logits_2)/2.0
            T = self.config.temperature  # temperature for softening
            # Compute softened probabilities
            soft_student = F.log_softmax(student_logits / T, dim=1)
            print("Student: {}, Teacher: {}".format(student_logits.shape, teacher_logits.shape))
            soft_teacher = F.softmax(teacher_logits / T, dim=1)
            loss_kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
            # Combine the losses: distillation loss and standard cross-entropy loss
            loss = self.config.distillation_alpha * loss_kd + (1 - self.config.distillation_alpha) * loss_ce
        else:
            loss = loss_ce

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss.detach().cpu())
        self.log("epoch", self.current_epoch)
        self.log("hard_loss", loss_ce, on_step=True, on_epoch=True,batch_size=bs)
        if self.teacher_model_1 is not None and self.teacher_model_2 is not None:
            self.log("soft_loss", loss_kd, on_step=True, on_epoch=True,batch_size=bs)

        return loss
    
    def on_train_epoch_end(self):
        pass

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch

        y_hat = self.forward(x)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(self.device)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'val' for logging
        self.log_dict({"val/" + k: logs[k] for k in logs})
        self.validation_step_outputs.clear()
        
    def on_test_epoch_start(self):
        if not next(self.model.parameters()).dtype == torch.half:
            self.model.half()
    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, cities = test_batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(self.device)

        # maximum memory allowance for parameters: 128 KB
        # baseline has 61148 parameters -> we can afford 16-bit precision
        # since 61148 * 16 bit ~ 122 kB
 
        # assure fp16
        
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'test' for logging
        self.log_dict({"test/" + k: logs[k] for k in logs})
        self.test_step_outputs.clear()

    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch

        # assure fp16
        self.model.half()

        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)

        return files, y_hat
    
os.environ["WANDB_MODE"] = "online"
    
def train(config):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'24 Task 1.",
        tags=["DCASE24"],
        config=vars(config),  # this logs all hyperparameters for us
        name=config.experiment_name
    )
    
    roll_samples = config.orig_sample_rate * config.roll_sec
    train_dl = DataLoader(
        dataset=get_training_set(config.subset, roll=roll_samples),
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,  # Optimized for faster host-to-GPU transfer
        persistent_workers=True if config.num_workers > 0 else False  # Reuse workers across epochs
    )

    # train dataloader
    assert config.subset in {100, 50, 25, 10, 5}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."
    roll_samples = config.orig_sample_rate * config.roll_sec
    train_dl = DataLoader(dataset=get_training_set(config.subset, roll=roll_samples),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          shuffle=True)

    test_dl = DataLoader(dataset=get_test_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size,        
                         pin_memory=True,
                         persistent_workers=True if config.num_workers > 0 else False
    )
    
    # create pytorch lightening module
    pl_module = PLModule(config, model_config)

    # get model complexity from nessi and log results to wandb
    sample = next(iter(test_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    # log MACs and number of parameters for our model
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='cpu',
                         devices=1,
                         precision=config.precision,
                         callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)])
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, test_dl)

    # final test step
    # here: use the validation split
    trainer.test(ckpt_path='last', dataloaders=test_dl)

    wandb.finish()


def evaluate(config):
    import os
    from sklearn import preprocessing
    import pandas as pd
    import torch.nn.functional as F
    from dataset.dcase24 import dataset_config

    assert config.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select" \
                                      f"the desired checkpoint."

    # create folder to store predictions
    os.makedirs("predictions", exist_ok=True)
    out_dir = os.path.join("predictions", config.ckpt_id)
    os.makedirs(out_dir, exist_ok=True)

    # load lightning module from checkpoint
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    trainer = pl.Trainer(logger=False,
                         accelerator='cpu',
                         devices=1,
                         precision=config.precision)

    # evaluate lightning module on development-test split
    test_dl = DataLoader(dataset=get_test_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size, 
                         pin_memory=True,
                         persistent_workers=True if config.num_workers > 0 else False
    )

    # get model complexity from nessi
    sample = next(iter(test_dl))[0][0].unsqueeze(0).to(pl_module.device)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)

    print(f"Model Complexity: MACs: {macs}, Params: {params}")
    assert macs <= nessi.MAX_MACS, "The model exceeds the MACs limit and must not be submitted to the challenge!"
    assert params <= nessi.MAX_PARAMS_MEMORY, \
        "The model exceeds the parameter limit and must not be submitted to the challenge!"

    allowed_precision = int(nessi.MAX_PARAMS_MEMORY / params * 8)
    print(f"ATTENTION: According to the number of model parameters and the memory limits that apply in the challenge,"
          f" you are allowed to use at max the following precision for model parameters: {allowed_precision} bit.")

    # obtain and store details on model for reporting in the technical report
    info = {}
    info['MACs'] = macs
    info['Params'] = params
    res = trainer.test(pl_module, test_dl)
    info['test'] = res

    # generate predictions on evaluation set
    eval_dl = DataLoader(dataset=get_eval_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    predictions = trainer.predict(pl_module, dataloaders=eval_dl)
    # all filenames
    all_files = [item[len("audio/"):] for files, _ in predictions for item in files]
    # all predictions
    all_predictions = torch.cat([torch.as_tensor(p) for _, p in predictions], 0)
    all_predictions = F.softmax(all_predictions, dim=1)

    # write eval set predictions to csv file
    df = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    le = preprocessing.LabelEncoder()
    le.fit_transform(df[['scene_label']].values.reshape(-1))
    class_names = le.classes_
    df = {'filename': all_files}
    scene_labels = [class_names[i] for i in torch.argmax(all_predictions, dim=1)]
    df['scene_label'] = scene_labels
    for i, label in enumerate(class_names):
        df[label] = all_predictions[:, i]
    df = pd.DataFrame(df)

    # save eval set predictions, model state_dict and info to output folder
    df.to_csv(os.path.join(out_dir, 'output.csv'), sep='\t', index=False)
    torch.save(pl_module.model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
    with open(os.path.join(out_dir, "info.json"), "w") as json_file:
        json.dump(info, json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 24 argument parser')

    # general
    parser.add_argument('--project_name', type=str, default="DCASE24_Task1")
    parser.add_argument('--experiment_name', type=str, default="Baseline")
    parser.add_argument('--num_workers', type=int, default=0)  # number of workers for dataloaders
    parser.add_argument('--precision', type=str, default="32")

    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default=None)  # for loading trained model, corresponds to wandb id

    # dataset
    # subset in {100, 50, 25, 10, 5}
    parser.add_argument('--orig_sample_rate', type=int, default=44100)
    parser.add_argument('--subset', type=int, default=100)

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network (3 main dimensions to scale the baseline)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--channels_multiplier', type=float, default=1.8)
    parser.add_argument('--expansion_rate', type=float, default=2.1)

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mixstyle_p', type=float, default=0.4)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--roll_sec', type=int, default=0.1)  # roll waveform over time

    # peak learning rate (in cosinge schedule)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--warmup_steps', type=int, default=2000)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_length', type=int, default=3072)  # in samples (corresponds to 96 ms)
    parser.add_argument('--hop_length', type=int, default=500)  # in samples (corresponds to ~16 ms)
    parser.add_argument('--n_fft', type=int, default=4096)  # length (points) of fft, e.g. 4096 point FFT
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--f_min', type=int, default=0)  # mel bins are created for freqs. between 'f_min' and 'f_max'
    parser.add_argument('--f_max', type=int, default=None)

    # Knowledge distillation arguments:
    parser.add_argument('--use_teacher', action='store_true', help='Enable teacher model for knowledge distillation')
    parser.add_argument('--model_name', type=str, default='passt_dirfms_1', help='Path to the teacher model checkpoint')
    parser.add_argument('--temperature', type=float, default=2) # Temperature for Knowledge Distillation
    parser.add_argument('--distillation_alpha', type=float, default=0.02) # Loss weight for Knowledge Distillation
    parser.add_argument('--teacher_checkpoint_1', type=str, default=r"./resources/passt_dirfms_1.pt", 
                    help='Path to the first teacher model checkpoint')
    parser.add_argument('--teacher_checkpoint_2', type=str, default=r"./resources/cpr_128k_dirfms_1.pt", 
                    help='Path to the second teacher model checkpoint')

    # Add other necessary arguments
    args = parser.parse_args()
    from passt import get_model as get_passt
    from cp_resnet import get_model as get_cp_resnet
    
    # Define the model_config based on the model_name
    if args.model_name in ["cpr_128k_dirfms_1",
                           "cpr_128k_dirfms_2",
                           "cpr_128k_dirfms_3",
                           "cpr_128k_fms_1",
                           "cpr_128k_fms_2",
                           "cpr_128k_fms_3"]:
        model_config = {
            "mel": {
                "sr": 32000,
                "n_mels": 256,
                "win_length": 3072,
                "hopsize": 750,
                "n_fft": 4096,
                "fmax": None,
                "fmax_aug_range": 1000,
                "fmin": 0,
                "fmin_aug_range": 1
            },
            "net": {
                # "rho": 8,
                # "base_channels": 32,
                # "maxpool_stage1": [1],
                # "maxpool_kernel": (2, 1),
                # "maxpool_stride": (2, 1)
            },
            "model_fn": get_cp_resnet
        }
    if args.model_name in ["passt_dirfms_1", "passt_dirfms_2", "passt_dirfms_3", 
                            "passt_fms_1", "passt_fms_2", "passt_fms_3"]:
        model_config = {
            "mel": {
                "sr": 32000,
                "n_mels": 128,
                "win_length": 800,
                "hopsize": 320,
                "n_fft": 1024,
                "fmax": None,
                "fmax_aug_range": 1000,
                "fmin": 0,
                "fmin_aug_range": 1
            },
            "net": {
                "arch": "passt_s_swa_p16_128_ap476",
                "n_classes": 10,
                "input_fdim": 128,
                "s_patchout_t": 0,
                "s_patchout_f": 6
            },
            "model_fn": get_passt  # This should be the function that creates the model.
        }
    else:
        raise NotImplementedError(f"No model configuration for {args.model_name}")
    args.use_teacher = True
    args.teacher_checkpoint = [args.teacher_checkpoint_1, args.teacher_checkpoint_2]
    
    if args.evaluate:
        evaluate(args)
    else:
        train(args)

