import hyperparameter
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import torch.nn.functional as F
import jammy_flows
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnext101_32x8d


"""
Every class is a different 'class' of model that can be tuned and optimized by changing the hyperparameter in hyperparameter.py.
After training the model is saved in results/RunXXX/model.
"""
def get_model():
    print("Using ResNet34 model")
    return Split()

class Split(pl.LightningModule):

    def __init__(self):
        """
        Defines the layes used in the model.
        """

        super().__init__()

        self.automatic_optimization = hyperparameter.auto_opt
        device = torch.device(f"cuda:0")

        flow_options_overwrite = {}
        flow_options_overwrite['f'] = dict()
        flow_options_overwrite['f']['add_vertical_rq_spline_flow'] = 1
        flow_options_overwrite['f']["vertical_smooth"] = 1
        flow_options_overwrite['f']["vertical_fix_boundary_derivative"] = 1
        flow_options_overwrite['f']["spline_num_basis_functions"] = 2
        flow_options_overwrite["f"]["boundary_cos_theta_identity_region"] = 0

        self.pdf_energy = jammy_flows.pdf("e1", "ggt", conditional_input_dim=hyperparameter.cond_input, options_overwrite=flow_options_overwrite).to(device)
        self.pdf_direction = jammy_flows.pdf("s2", "fffffffffffffff", conditional_input_dim=hyperparameter.cond_input, options_overwrite=flow_options_overwrite).to(device)

        self.class_criterion = nn.BCELoss()
        self.classifier = nn.Sequential(
                            nn.Linear(1000, 1),
                            nn.Sigmoid() 
                            )
        
        self.model1D = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 2)),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),

            nn.BatchNorm2d(256),
            )
        
        self.resnet34_5 = ResNet(ResidualBlock, [3, 4, 6, 3], 1000)

    def forward(self, x):
        x = self.model1D(x)
        x = x.permute((0, 2, 1, 3))
        x = self.resnet34_5(x)
        return x

    def second_init(self, data):
        self.pdf.init_params(data)

    def configure_optimizers(self): 
        """
        Defines the optimizer used in the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=hyperparameter.learning_rate, eps=hyperparameter.opt_eps) 
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=hyperparameter.sch_factor, patience=hyperparameter.rd_patience, min_lr=hyperparameter.min_lr, verbose=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_total_loss'}


    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_total_loss"])


        
    def training_step(self, train_batch, batch_idx):

        try:

            x, direction, energy, flavor = train_batch

            conv_out = self.forward(x)
            # self.log('conv_out_shape', conv_out.shape, on_step=True, on_epoch=True, prog_bar=True)
        
            flavor_pred = self.classifier(conv_out) 

            log_pdf_energy, _,_= self.pdf_energy(energy.to(torch.double), conditional_input=conv_out.to(torch.double))
            log_pdf_direction, _,_= self.pdf_direction(direction.to(torch.double), conditional_input=conv_out.to(torch.double))
            loss_classifier = self.class_criterion(flavor_pred.squeeze(), flavor.squeeze())

            final_loss = loss_classifier - log_pdf_energy.mean() - log_pdf_direction.mean()

            self.log('train_class_loss', loss_classifier)
            self.log('train_pdf_energy', -log_pdf_energy.mean())
            self.log('train_pdf_direction', -log_pdf_direction.mean())
            self.log('train_total_loss', final_loss)

            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(final_loss)

            grads = []

            for param in Split.parameters(self):

                if param.grad == None:
                    continue

                grads.append(param.grad.view(-1))

            grads = torch.cat(grads)

            if torch.isnan(grads).any():

                torch.set_printoptions(threshold=10_000)
                f = open("output.txt", "a")

                print('########################')
                print('nan occured here')
                print('########################')

                for name, param in Split.named_parameters(self):

                    if torch.isnan(param).any():
                        print(name, file=f)
                        print(param, file=f)

                f.close()

            assert(not torch.isnan(grads).any())
        
            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()

        except:
            print('error in training step')
        
            
    def validation_step(self, val_batch, batch_idx):

        x, direction, energy, flavor = val_batch
        conv_out = self.forward(x)

        flavor_pred = self.classifier(conv_out) 

        log_pdf_energy, _,_= self.pdf_energy(energy.to(torch.double), conditional_input=conv_out.to(torch.double))
        log_pdf_direction, _,_= self.pdf_direction(direction.to(torch.double), conditional_input=conv_out.to(torch.double))
        loss_classifier = self.class_criterion(flavor_pred.squeeze(), flavor.squeeze())

        final_loss = loss_classifier - log_pdf_energy.mean() - log_pdf_direction.mean()

        self.log('val_class_loss', loss_classifier)
        self.log('val_pdf_energy', -log_pdf_energy.mean())
        self.log('val_pdf_direction', -log_pdf_direction.mean())
        self.log('val_total_loss', final_loss)
        

class AdaptiveConcatPool2d(nn.Module):
    "Concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self):
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], 1)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(5, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
