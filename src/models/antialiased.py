from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization
from mldb.utils import load_model_state_dict
from models import models_lpf


model_params = {
'alexnet_lpf2': {   'arch': 'alexnet',
                    'eval_batch_size': 256,
                    'filter_size': 2,
                    'img_crop_size': 224,
                    'img_resize_size': 256,
                    'input_space': 'RGB',
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]},
'alexnet_lpf3': {   'arch': 'alexnet',
                    'eval_batch_size': 256,
                    'filter_size': 3,
                    'img_crop_size': 224,
                    'img_resize_size': 256,
                    'input_space': 'RGB',
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]},
'alexnet_lpf5': {   'arch': 'alexnet',
                    'eval_batch_size': 256,
                    'filter_size': 5,
                    'img_crop_size': 224,
                    'img_resize_size': 256,
                    'input_space': 'RGB',
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]},
'densenet121_lpf2': {   'arch': 'densenet121',
                        'eval_batch_size': 256,
                        'filter_size': 2,
                        'img_crop_size': 224,
                        'img_resize_size': 256,
                        'input_space': 'RGB',
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]},
'densenet121_lpf3': {   'arch': 'densenet121',
                        'eval_batch_size': 256,
                        'filter_size': 3,
                        'img_crop_size': 224,
                        'img_resize_size': 256,
                        'input_space': 'RGB',
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]},
'densenet121_lpf5': {   'arch': 'densenet121',
                        'eval_batch_size': 256,
                        'filter_size': 5,
                        'img_crop_size': 224,
                        'img_resize_size': 256,
                        'input_space': 'RGB',
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]},
'mobilenet_v2_lpf2': {   'arch': 'mobilenet_v2',
                         'eval_batch_size': 64,
                         'filter_size': 2,
                         'img_crop_size': 224,
                         'img_resize_size': 256,
                         'input_space': 'RGB',
                         'mean': [0.485, 0.456, 0.406],
                         'std': [0.229, 0.224, 0.225]},
'mobilenet_v2_lpf3': {   'arch': 'mobilenet_v2',
                         'eval_batch_size': 64,
                         'filter_size': 3,
                         'img_crop_size': 224,
                         'img_resize_size': 256,
                         'input_space': 'RGB',
                         'mean': [0.485, 0.456, 0.406],
                         'std': [0.229, 0.224, 0.225]},
'mobilenet_v2_lpf5': {   'arch': 'mobilenet_v2',
                         'eval_batch_size': 64,
                         'filter_size': 5,
                         'img_crop_size': 224,
                         'img_resize_size': 256,
                         'input_space': 'RGB',
                         'mean': [0.485, 0.456, 0.406],
                         'std': [0.229, 0.224, 0.225]},
'resnet101_lpf2': {   'arch': 'resnet101',
                      'eval_batch_size': 256,
                      'filter_size': 2,
                      'img_crop_size': 224,
                      'img_resize_size': 256,
                      'input_space': 'RGB',
                      'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225]},
'resnet101_lpf3': {   'arch': 'resnet101',
                      'eval_batch_size': 256,
                      'filter_size': 3,
                      'img_crop_size': 224,
                      'img_resize_size': 256,
                      'input_space': 'RGB',
                      'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225]},
'resnet101_lpf5': {   'arch': 'resnet101',
                      'eval_batch_size': 256,
                      'filter_size': 5,
                      'img_crop_size': 224,
                      'img_resize_size': 256,
                      'input_space': 'RGB',
                      'mean': [0.485, 0.456, 0.406],
                      'std': [0.229, 0.224, 0.225]},
'resnet18_lpf2': {   'arch': 'resnet18',
                     'eval_batch_size': 256,
                     'filter_size': 2,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'resnet18_lpf3': {   'arch': 'resnet18',
                     'eval_batch_size': 256,
                     'filter_size': 3,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'resnet18_lpf5': {   'arch': 'resnet18',
                     'eval_batch_size': 256,
                     'filter_size': 5,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'resnet34_lpf2': {   'arch': 'resnet34',
                     'eval_batch_size': 256,
                     'filter_size': 2,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'resnet34_lpf3': {   'arch': 'resnet34',
                     'eval_batch_size': 256,
                     'filter_size': 3,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'resnet34_lpf5': {   'arch': 'resnet34',
                     'eval_batch_size': 256,
                     'filter_size': 5,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'resnet50_lpf2': {   'arch': 'resnet50',
                     'eval_batch_size': 256,
                     'filter_size': 2,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'resnet50_lpf3': {   'arch': 'resnet50',
                     'eval_batch_size': 256,
                     'filter_size': 3,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'resnet50_lpf5': {   'arch': 'resnet50',
                     'eval_batch_size': 256,
                     'filter_size': 5,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'vgg16_bn_lpf2': {   'arch': 'vgg16_bn',
                     'eval_batch_size': 32,
                     'filter_size': 2,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'vgg16_bn_lpf3': {   'arch': 'vgg16_bn',
                     'eval_batch_size': 32,
                     'filter_size': 3,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'vgg16_bn_lpf5': {   'arch': 'vgg16_bn',
                     'eval_batch_size': 32,
                     'filter_size': 5,
                     'img_crop_size': 224,
                     'img_resize_size': 256,
                     'input_space': 'RGB',
                     'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]},
'vgg16_lpf2': {   'arch': 'vgg16',
                  'eval_batch_size': 32,
                  'filter_size': 2,
                  'img_crop_size': 224,
                  'img_resize_size': 256,
                  'input_space': 'RGB',
                  'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]},
'vgg16_lpf3': {   'arch': 'vgg16',
                  'eval_batch_size': 32,
                  'filter_size': 3,
                  'img_crop_size': 224,
                  'img_resize_size': 256,
                  'input_space': 'RGB',
                  'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]},
'vgg16_lpf5': {   'arch': 'vgg16',
                  'eval_batch_size': 32,
                  'filter_size': 5,
                  'img_crop_size': 224,
                  'img_resize_size': 256,
                  'input_space': 'RGB',
                  'mean': [0.485, 0.456, 0.406],
                  'std': [0.229, 0.224, 0.225]}}


def gen_classifier_loader(name, d):
    def classifier_loader():
        model = getattr(models_lpf, d['arch'])(filter_size=d['filter_size'])
        load_model_state_dict(model, name)
        return model
    return classifier_loader


for name, d in model_params.items():
    registry.add_model(
        Model(
            name = name,
            arch = d['arch'],
            transform = StandardTransform(d['img_resize_size'], d['img_crop_size']),
            normalization = StandardNormalization(d['mean'], d['std'], d['input_space']),
            classifier_loader = gen_classifier_loader(name, d),
            eval_batch_size = d['eval_batch_size'],
            adversarial_batch_size = d['adversarial_batch_size'] if 'adversarial_batch_size' in d else None
        )
    )
