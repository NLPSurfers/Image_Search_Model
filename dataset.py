import json
import torch
import torch.utils.data as data
from tqdm import tqdm
from PIL import Image
import base64
import io
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.transforms as T
#from db_control import query_db_for_restaurant_table, query_db_for_images
#from db_control import RESTAURANT_TABLE_URL, IMAGE_URL

def get_transforms(is_train, size=224, degrees=(-180, 180), translate=(.5, .5), scale=(.8, 1.2), shear=.2):
    transforms = []
    transforms.append(SquarePad())
    if is_train:
        transforms.append(T.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear))
        transforms.append(T.RandomVerticalFlip())
        transforms.append(T.RandomHorizontalFlip())
    transforms.append(T.Resize(size=(size, size)))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
    
class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')
        
class RestaurantImageDataset(data.Dataset):
    
    def __init__(self, table_path, info_path, cluster_path, transform=None):
        
        super(RestaurantImageDataset, self).__init__()
        self.table_path = table_path
        self.info_path = info_path
        self.cluster_path = cluster_path
        self.transform = transform
        self.initialize()
    
    def initialize(self):
        
        if self.transform == None:
            self.transform = get_transforms(True)
        
        print("Loading Raw Data ...")
        with open(self.table_path, "r") as read_file:
            table = json.load(read_file)
        with open(self.info_path, "r") as read_file:
            infos = json.load(read_file)
        with open(self.cluster_path, "r") as read_file:
            clusters = json.load(read_file)
        
        self.images = []
        self.labels = []
        for n in range(len(clusters)):
            print("Preprocess for cluster - {}".format(n))
            for restaurant_id in tqdm(clusters[n]):
                images = infos[str(restaurant_id)]
                
                labels = [n] * len(images)
                
                self.images += images
                self.labels += labels
        
    def __getitem__(self, index):
    
        image_data = self.images[index]
        label = self.labels[index]
        
        image = Image.open(io.BytesIO(base64.b64decode(image_data.encode("utf-8"))))
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    table_path = "./table.json"
    cluster_path = "./clusters.json"
    image_path = "./images.json"
    
    dataset = RestaurantImageDataset(table_path, image_path, cluster_path)
    for x, y in dataset:
        print(x.shape, y)
        break # temp