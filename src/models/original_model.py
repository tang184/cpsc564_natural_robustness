from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
import torchvision.models as torch_models
import torch
from collections import OrderedDict 
def classifier_loader():
    model = torch_models.resnet18()
    
    checkpoint = torch.load('/home/jtang/Desktop/original_model/checkpoint.pth.tar')

    
    prefix = 'module.'
    n_clip = len(prefix)
    adapted_dict = {k[n_clip:]: v for k, v in checkpoint['state_dict'].items()
                    if k.startswith(prefix)}

    for key, value in adapted_dict.items():
        print (key)
    model.load_state_dict(adapted_dict)
    
    # model.load_state_dict(new_state_dict)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    return model

registry.add_model(
    Model(
        name = 'original_model',
        transform = StandardTransform(img_resize_size=256, img_crop_size=224),
        normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        classifier_loader = classifier_loader,
        eval_batch_size = 128,
        # OPTIONAL
        arch = 'resnet18',
    )
)