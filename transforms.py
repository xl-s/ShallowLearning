from torchvision.transforms import (
    Compose,
    ToTensor,
    Grayscale,
    Normalize,
    ColorJitter,
    RandomHorizontalFlip,
    RandomApply,
    Resize,
    RandomRotation,
)
from torch.nn import ModuleList

# Saturation/Brightness/Contrast (ColorJitter) [Photometric Distortion]
# HorizontalFlip
# Can try a small amount of rotation?

aug_transform = Compose(
    [
        Grayscale(),
        RandomApply(
            ModuleList([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)]),
            p=0.5,
        ),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(10),
        ToTensor(),
        Normalize((0.5), (0.5)),
    ]
)

clean_transform = Compose([Grayscale(), ToTensor(), Normalize((0.5), (0.5))])

aug_ext_transform = Compose(
    [
        Resize(224),
        RandomApply(
            ModuleList([ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)]),
            p=0.5,
        ),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(10),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

ext_transform = Compose(
    [
        Resize(224),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

