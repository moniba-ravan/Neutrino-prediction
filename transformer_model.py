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
import numpy as np
import math



"""
A class representing the Original transformer model.
"""

def get_model():
    print("Using Original Transformer model")
    return TransTLightningModule()

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
import numpy as np
import math



"""
Every class is a different 'class' of model that can be tuned and optimized by changing the hyperparameter in hyperparameter.py.
After training the model is saved in results/RunXXX/model.
"""

class TransTLightningModule(pl.LightningModule):

    def __init__(self):
        
        #Defines the layers used in the model.
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

        #This is why we need to change cond_input to 512 in hyperparameter.py
        self.pdf_energy = jammy_flows.pdf("e1", "ggt", conditional_input_dim=hyperparameter.cond_input, options_overwrite=flow_options_overwrite).to(device)
        self.pdf_direction = jammy_flows.pdf("s2", "fffffffffffffff", conditional_input_dim=hyperparameter.cond_input, options_overwrite=flow_options_overwrite).to(device)

        self.class_criterion = nn.BCELoss()
        self.classifier = nn.Sequential(    
                            nn.Linear(512, 1),
                            nn.Sigmoid() 
                            )
    
        self.model1D = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AvgPool2d(kernel_size=(1, 2)),

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 16), padding='same'),
            nn.ReLU(),

            nn.BatchNorm2d(256),
            )
        
        #self.transformer = Trans(embed_dim=520, num_heads=8, num_encoders=4, num_classes=520)
        #self.transformer = ViT(embed_dim=256, patch_size=(16, 16), num_patches=256, dropout=0.001, in_channels=5, num_heads=8, num_encoders=4, expansion=128, num_classes=512)
    
        self.transformer = Trans(embed_dim=512, num_heads=8, num_encoders=4, num_classes=512, seq_len = 5)
        #Trans(embed_dim=embed_dim, num_heads=num_heads, num_encoders=num_encoders, num_classes=num_classes, seq_len=seq_len)
    def forward(self, x):
        #print(x.shape)            #torch.Size([128, 1, 5, 520])
        x = self.transformer(x)
        #print(x.shape)            #torch.Size([128, num_classes])

        #print(x.shape)              #torch.Size([128, 1, 5, 520])
        #x = self.model1D(x)              
        #x = x.permute((0, 2, 1, 3))
        #print(x.shape)              #torch.Size([128, 5, 256, 256])  
        #x = self.transformer(x)
        #print(x.shape)              #torch.Size([128, 512])
        return x

    def second_init(self, data):
        self.pdf.init_params(data) 
    
    def configure_optimizers(self): 

        #Defines the optimizer used in the model.
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

            for param in TransTLightningModule.parameters(self):

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

                for name, param in TransTLightningModule.named_parameters(self):

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


class Trans(nn.Module):
    def __init__(self, embed_dim, num_heads, num_encoders, num_classes, seq_len=5):
        super().__init__()

        # Positional encoding is added to help the model understand the order of the sequence.
        self.register_buffer('positional_encoding', self._generate_positional_encoding(seq_len=seq_len, embed_dim=embed_dim))

        # Save the sequence length and embedding size for later use.
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Transformer Encoder layer with attention and feedforward components.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  # Size of embeddings
            nhead=num_heads,    # Number of attention heads
            dim_feedforward=1024,  # Hidden layer size in feedforward network
            activation="gelu",     # Activation function
            batch_first=True,      # Input shape: [batch, seq_len, embed_dim]
            norm_first=True        # Apply normalization before attention/FFN
        )

        # Stack multiple encoder layers to form the Transformer Encoder.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        # Fully connected layer to convert embeddings to class predictions.
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Ensure positional encoding is on the same device as input.
        positional_encoding = self.positional_encoding.to(x.device)

        # If sequence length changes, regenerate positional encodings.
        if x.size(2) != self.seq_len:
            positional_encoding = self._generate_positional_encoding(seq_len=x.size(2), embed_dim=self.embed_dim).to(x.device)

        # Remove the extra dimension from input: [batch, 1, seq_len, embed_dim] -> [batch, seq_len, embed_dim]
        x = x.squeeze(1)

        # Add positional encoding to the input.
        x = x + positional_encoding

        # Pass the input through the Transformer Encoder.
        x = self.encoder(x)

        # Pool the sequence by taking the mean across the sequence dimension.
        x = x.mean(dim=1)

        # Pass through the fully connected layer to get the final output.
        x = self.fc(x)

        return x

    def _generate_positional_encoding(self, seq_len, embed_dim):
        # Generate positions and embedding indices.
        pos = torch.arange(seq_len).unsqueeze(1)  # Positions: [seq_len, 1]
        i = torch.arange(embed_dim).unsqueeze(0)  # Embedding dims: [1, embed_dim]

        # Compute scaling factors for positional encodings.
        div_term = torch.exp(-math.log(10000.0) * (2 * (i // 2)) / embed_dim)

        # Apply sine to even dimensions and cosine to odd dimensions.
        pos_enc = torch.zeros(seq_len, embed_dim)
        pos_enc[:, 0::2] = torch.sin(pos * div_term[:, 0::2])  # Even dimensions
        pos_enc[:, 1::2] = torch.cos(pos * div_term[:, 1::2])  # Odd dimensions

        # Add batch dimension: [1, seq_len, embed_dim]
        return pos_enc.unsqueeze(0)
