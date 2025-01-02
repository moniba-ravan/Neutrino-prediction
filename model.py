import vit_model 
import resnet34   # Your existing ResNet34 model class
import transformer_model

def get_model(model_name, **kwargs):
    """
    Factory method to return the desired model class.
    """
    if model_name == "resnet34":
        print("Using ResNet34 model")
        return resnet34.Split()
    elif model_name == "vit_model": 
        print("Using Vision Transformer (ViT) model")
        return vit_model.ViTLightningModule()
    elif model_name == "transformer_model":
        print("Using Original Transformer model")
        return transformer_model.TransTLightningModule()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
