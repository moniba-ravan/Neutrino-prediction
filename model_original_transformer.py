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

# ----------------------------------------------
# Transformer Components
# ----------------------------------------------

class AddNormalization(nn.Module):
    def __init__(self, d_model):
        super(AddNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_x):
        return self.layer_norm(x + sublayer_x)

class FeedForward(nn.Module):
    def __init__(self, d_ff, d_model):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, keys, values, d_k, mask=None):
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, values)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_k, d_v, d_model):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.W_q = nn.Linear(d_model, h * d_k)
        self.W_k = nn.Linear(d_model, h * d_k)
        self.W_v = nn.Linear(d_model, h * d_v)
        self.W_o = nn.Linear(h * d_v, d_model)
        self.attention = DotProductAttention()

    def reshape_for_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.size(0)
        q = self.reshape_for_heads(self.W_q(queries), batch_size)
        k = self.reshape_for_heads(self.W_k(keys), batch_size)
        v = self.reshape_for_heads(self.W_v(values), batch_size)
        attn_output = self.attention(q, k, v, torch.tensor(self.d_k, dtype=torch.float32), mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_v)
        return self.W_o(attn_output)

class EncoderLayer(nn.Module):
    def __init__(self, h, d_k, d_v, d_model, d_ff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.add_norm1 = AddNormalization(d_model)
        self.feed_forward = FeedForward(d_ff, d_model)
        self.add_norm2 = AddNormalization(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask=None):
        attn_output = self.multihead_attention(x, x, x, padding_mask)
        x = self.add_norm1(x, self.dropout(attn_output))
        ffn_output = self.feed_forward(x)
        return self.add_norm2(x, self.dropout(ffn_output))

class PositionEmbedding(nn.Module):
    def __init__(self, seq_length, d_model):
        super(PositionEmbedding, self).__init__()
        self.embedding = nn.Parameter(self.get_position_encoding(seq_length, d_model), requires_grad=False)

    def get_position_encoding(self, seq_len, d_model, n=10000):
        P = np.zeros((seq_len, d_model))
        for k in range(seq_len):
            for i in np.arange(d_model // 2):
                denominator = np.power(n, 2 * i / d_model)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return torch.tensor(P, dtype=torch.float32)

    def forward(self, x):
        seq_len = x.size(1)  # Get actual sequence length of the input x
        return x + self.embedding[:seq_len, :]  # Slice to match input sequence length

# ----------------------------------------------
# Transformer Model Definition
# ----------------------------------------------

import jammy_flows

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

class TransformerModel(pl.LightningModule):
    def __init__(self, embed_dim=512, num_heads=8, num_encoders=6, dropout=0.1, learning_rate=1e-4):
        super(TransformerModel, self).__init__()
        
        # Position embedding to encode sequence information
        self.position_encoding = PositionEmbedding(seq_length=5, d_model=embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=2048, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Adjust input shape from [batch_size, 1, 5, 512] -> [batch_size, 5, 512]
        x = x.squeeze(1)  # Remove the singleton dimension
        
        # Apply position encoding and transformer encoder
        x = self.position_encoding(x)
        x = self.encoder(x)
        
        # Optionally, you can perform pooling here to reduce the output to a single vector per batch
        x = x.mean(dim=1)  # Mean pooling over sequence dimension

        return x
    
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