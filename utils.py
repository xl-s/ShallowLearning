import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from enum import Enum


class ImageSubfolder(ImageFolder):
	"""
	Assign classes by the deepest subfolder instead of by the subfolders of the root
	"""
	def _find_classes(self, dir):
		def has_dir(d):
			for item in os.scandir(d):
				if item.is_dir():
					return True
			return False
		classes = []
		base_length = len(dir)
		def add_dirs(root):
			for d in os.scandir(root):
				if not d.is_dir(): continue
				if has_dir(d):
					add_dirs(f"{root}/{d.name}")
				else:
					classes.append(f"{root}/{d.name}"[base_length+1:])
		add_dirs(dir)
		classes.sort()
		class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
		return classes, class_to_idx


class ClassType(Enum):
	THREE_CLASS = 0
	NORMAL_INFECTED = 1
	COVID_NONCOVID = 2

def _load_data(loc="dataset", kind="train", suf=None, sub=False, transform=None, **kwargs):
	base = f"{loc}/{kind}{'' if not suf else '/' + suf}"
	dataset = ImageSubfolder(base, transform=transform) if sub else ImageFolder(base, transform=transform)
	return DataLoader(dataset, **kwargs)

def load_data(loc="dataset", kind="train", class_type=ClassType.THREE_CLASS, transform=None, **kwargs):
	if class_type == ClassType.THREE_CLASS:
		return _load_data(loc=loc, kind=kind, sub=True, transform=transform, **kwargs)
	elif class_type == ClassType.NORMAL_INFECTED:
		return _load_data(loc=loc, kind=kind, sub=False, transform=transform, **kwargs) 
	elif class_type == ClassType.COVID_NONCOVID:
		return _load_data(loc=loc, kind=kind, sub=False, suf="infected", transform=transform, **kwargs)
	else:
		raise ValueError("Invalid class_type")