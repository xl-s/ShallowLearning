from torchvision.transforms import (
    Compose,
    ToTensor,
    Grayscale,
    Normalize,
    ColorJitter,
    RandomHorizontalFlip,
    RandomApply,
    RandomRotation,
)
from torch.nn import ModuleList


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
