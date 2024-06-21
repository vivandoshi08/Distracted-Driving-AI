import sys
import torchvision.models as tv_models
import torch.nn as nn
from timm.models.ghostnet import ghostnet_050, ghostnet_100, ghostnet_130

def update_classifier(args, model):
    """Update the classifier of a pre-trained network to match the required number of classes."""
    classifiers = {
        'dense': ['dense121', 'dense161', 'dense201', 'ghost1_0'],
        'vgg': ['vgg19bn', 'vgg16bn'],
        'inception': ['inceptionv3', 'inceptionv4'],
        'mobile': ['mobilenet_v2', 'mobilenetv3_large', 'mobilenetv3_small', 'efficientnet_b0'],
        'squeeze': ['squeezenet1_0', 'squeezenet1_1']
    }

    if args.net in classifiers['dense']:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, args.num_classes)
    elif args.net in classifiers['mobile']:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, args.num_classes)
    elif args.net in classifiers['inception']:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, args.num_classes)
    elif args.net in classifiers['vgg']:
        model.classifier[-1] = nn.Linear(4096, args.num_classes)
    elif args.net in classifiers['squeeze']:
        model.classifier[1] = nn.Conv2d(512, args.num_classes, kernel_size=1)
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, args.num_classes)
    return model

def load_mobilenet_v3_small(opt, num_classes, level=1):
    model = tv_models.mobilenet_v3_small(pretrained=True)
    model.requires_grad_(False)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes * level)
    model.classifier.requires_grad_(True)
    return model, level

def load_mobilenet_v3_large(opt, num_classes, level=1):
    model = tv_models.mobilenet_v3_large(pretrained=True)
    model.requires_grad_(False)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes * level)
    model.classifier.requires_grad_(True)
    return model, level

def load_resnet18(opt, num_classes, level=1):
    model = tv_models.resnet18(pretrained=True)
    model.requires_grad_(False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes * level)
    model.fc.requires_grad_(True)
    return model, level

def load_resnet50(opt, num_classes, level=1):
    model = tv_models.resnet50(pretrained=True)
    model.requires_grad_(False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes * level)
    model.fc.requires_grad_(True)
    return model, level

def load_shufflenet_v2(opt, num_classes, level=1):
    model = tv_models.shufflenet_v2_x1_0(pretrained=True)
    model.requires_grad_(False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes * level)
    model.fc.requires_grad_(True)
    return model, level

def load_squeezenet(opt, num_classes, level=1):
    model = tv_models.squeezenet1_0(pretrained=True)
    model.requires_grad_(False)
    in_channels = model.classifier[1].in_channels
    model.classifier[1] = nn.Conv2d(in_channels, num_classes * level, kernel_size=1)
    model.classifier.requires_grad_(True)
    return model, level

def load_efficientnet_b0(opt, num_classes, level=1):
    model = tv_models.efficientnet_b0(pretrained=True)
    model.requires_grad_(False)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes * level)
    model.classifier.requires_grad_(True)
    return model, level

def load_ghostnet_050(opt, num_classes, level=1):
    model = ghostnet_050(pretrained=True)
    model.requires_grad_(False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes * level)
    model.classifier.requires_grad_(True)
    return model, level

def load_ghostnet_100(opt, num_classes, level=1):
    model = ghostnet_100(pretrained=True)
    model.requires_grad_(False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes * level)
    model.classifier.requires_grad_(True)
    return model, level

if __name__ == '__main__':
    import argparse
    import torch
    import copy
    from thop import profile, clever_format

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1000, help='number of classes')
    parser.add_argument('--image_size', type=tuple, default=(224, 224), help='input image size (w, h)')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--model_names', nargs='+', default=["mobileVGG"], help='model names')
    parser.add_argument('--input_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--device', type=str, default="cpu", help='device to use (cpu or cuda)')
    parser.add_argument('--dtype', type=torch.dtype, default=torch.float32, help='tensor data type')

    opt = parser.parse_args()

    levels = [1] + [i * 5 for i in range(1, 7)]
    class_counts = [i * 10 for i in range(1, 101)]

    writers = []
    for metric in ['macs', 'params']:
        csv_file = open(f'{metric}.csv', mode='w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=map(str, levels))
        csv_writer.writeheader()
        writers.append(csv_writer)

    for num_classes in class_counts:
        opt.num_classes = num_classes
        macs_info, params_info = {}, {}
        for level in levels:
            model, level = load_resnet18(opt, num_classes, level)
            inputs = torch.randn(1, opt.input_channels, opt.image_size[0], opt.image_size[1])
            test_model = copy.deepcopy(model).to(opt.device)
            macs, params = profile(test_model, inputs=(inputs,))
            macs, params = clever_format([macs, params], "%.6f")
            print(level, macs, params)

            macs_info[str(level)] = float(macs[:-1])
            params_info[str(level)] = float(params[:-1])

        writers[0].writerow(macs_info)
        writers[1].writerow(params_info)
