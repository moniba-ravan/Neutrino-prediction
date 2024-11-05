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


"""
Every class is a different 'class' of model that can be tuned and optimized by changing the hyperparameter in hyperparameter.py.
After training the model is saved in results/RunXXX/model.
"""


#The patch embedding from the article that me and Sissi where reading, it includes positional encoding as well
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
          # We use conv for doing the patching
            nn.Conv2d(
                  in_channels=in_channels,
                  out_channels=embed_dim,
                  # if kernel_size = stride -> no overlap
                  kernel_size=patch_size,
                  stride=patch_size
          ),
          # Linear projection of Flattened Patches. We keep the batch and the channels (b,c,h,w)
            nn.Flatten(2))
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Create a copy of the cls token for each of the elements of the BATCH
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # Create the patches
        x = self.patcher(x).permute(0, 2, 1)
        # Unify the position with the patches
        x = torch.cat([cls_token, x], dim=1)
        # Patch + Position Embedding
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x
    
#The fancy class itself, a lot of parameters to keep track of here, 
#I know what half of them mean and I am confused about the other half

class ViT(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels, num_heads, num_encoders, expansion, num_classes):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=int(embed_dim*expansion), activation="gelu", batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])  # Apply MLP on the CLS token only
        return x
    

    #From this line and below I included the optimizers and training that they already provided for us, 
    #but I'm not using it yet (I don't think atleast) so this can be ignored or maybe even removed 

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
