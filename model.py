import vit_model 
import resnet34   # Your existing ResNet34 model class

def get_model(model_name, **kwargs):
    """
    Factory method to return the desired model class.
    """
    if model_name == "resnet34":
        return resnet34.Split()
    elif model_name == "vit":
        return vit_model.ViTLightningModule()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
