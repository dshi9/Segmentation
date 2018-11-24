"""CamVid Dataset Segmentation Dataloader"""
"""http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
import torch
from torch.utils.data import Dataset
from PIL import Image
import pdb

CAMVID_CLASSES = [  'Animal',
                    'Archway',
                    'Bicyclist',
                    'Bridge',
                    'Building',
                    'Car',
                    'CartLuggagePram',
                    'Child',
                    'Column_Pole',
                    'Fence',
                    'LaneMkgsDriv',
                    'LaneMkgsNonDriv',
                    'Misc_Text',
                    'MotorcycleScooter',
                    'OtherMoving',
                    'ParkingBlock',
                    'Pedestrian',
                    'Road',
                    'RoadShoulder',
                    'Sidewalk',
                    'SignSymbol',
                    'Sky',
                    'SUVPickupTruck',
                    'TrafficCone',
                    'TrafficLight',
                    'Train',
                    'Tree',
                    'Truck_Bus',
                    'Tunnel',
                    'VegetationMisc',
                    'Void',
                    'Wall'
                    ]
CAMVID_CLASS_COLOR = [
                    [64,128,64],
                    [192,0,128],
                    [0,128,192],
                    [0,128,64],
                    [128,0,0],
                    [64,0,128],
                    [64,0,192],
                    [192,128,64],
                    [192,192,128],
                    [64,64,128],
                    [128,0,192],
                    [192,0,64],
                    [128,128,64],
                    [192,0,192],
                    [128,64,64],
                    [64,192,128],
                    [64,64,0],
                    [128,64,128],
                    [128,128,192],
                    [0,0,192],
                    [192,128,128],
                    [128,128,128],
                    [64,128,192],
                    [0,0,64],
                    [0,64,64],
                    [192,64,128],
                    [128,128,0],
                    [192,128,192],
                    [64,0,64],
                    [192,192,0],
                    [0,0,0],
                    [64,192,0]
                    ]

class CamVid(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.images = list(map(lambda x: x.split(".")[0], listdir(img_dir)))
        self.transform = transform

        self.extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

        #  self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.extension)
        mask_path = os.path.join(self.mask_root_dir, name + "_L" + self.extension)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)

        data = {
                'image': torch.FloatTensor(image),
                'mask' : torch.LongTensor(gt_mask)
                }

        return data

    #  def __compute_class_probability(self):
    #      counts = dict((i, 0) for i in range(len(CAMVID_CLASSES)))
    #
    #      for name in self.images:
    #          mask_path = os.path.join(self.mask_root_dir, name + "_L" + self.extension)
    #
    #          raw_image = Image.open(mask_path).resize((224, 224))
    #          pdb.set_trace()
    #          imx_t = np.array(raw_image).reshape(224*224)
    #          #  imx_t[imx_t==255] = len(VOC_CLASSES)
    #
    #          for i in range(len(CAMVID_CLASSES)):
    #              counts[i] += np.sum(imx_t == i)
    #
    #      return counts
    #
    #  def get_class_probability(self):
    #      values = np.array(list(self.counts.values()))
    #      p_values = values/np.sum(values)
    #
        #  return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = np.transpose(raw_image.resize((224, 224)), (2,1,0))
        imx_t = np.array(raw_image, dtype=np.float32)/255.0

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = np.array(raw_image.resize((224, 224)))
        imx_t = np.zeros((224,224))
        for i in range(224):
            for j in range(224):
                imx_t[i, j] = CAMVID_CLASS_COLOR.index(list(raw_image[i,j]))
        return imx_t


if __name__ == "__main__":
    img_dir = "701_StillsRaw_full"
    mask_dir = "LabeledApproved_full"


    dataset = CamVid(img_dir=img_dir, mask_dir=mask_dir)

    #  print(objects_dataset.get_class_probability())

    sample = dataset[0]
    image, mask = sample['image'], sample['mask']

    image.transpose_(0, 2)

    fig = plt.figure()

    a = fig.add_subplot(1,2,1)
    plt.imshow(image)

    a = fig.add_subplot(1,2,2)
    plt.imshow(mask)

    plt.show()
