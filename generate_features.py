import os
import glob
import json
import io
import base64
from PIL import Image
import torch
import geffnet
from tqdm import tqdm
from dataset import get_transforms

def get_features(images, model, device, size=224):
    model.eval()
    features = []
    
    transform = get_transforms(False, size=size)
    
    images = [Image.open(io.BytesIO(base64.b64decode(image_data.encode("utf-8")))) for image_data in images]
    images = [transform(image) for image in images]
    images = [torch.unsqueeze(image, 0) for image in images]        
    
    images = torch.cat(images)
    images = images.to(device)
    
    features = [model.features(torch.unsqueeze(image, 0)) for image in images]
    features = torch.cat(features)
    
    return features

def do_feature_generating(cluster_path, image_path, load_model_path, output_dir, skip_existed=False, use_cpu=True):

    exist_features = glob.glob(os.path.join(output_dir, "*.pt"))
    exist_features = [int(os.path.basename(filename).replace(".pt", "")) for filename in exist_features]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # not posible for GPU (memory not enough)
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(cluster_path, "r") as read_file:
        clusters = json.load(read_file)
    
    print("Load Model")
    #model = geffnet.efficientnet_b0(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2, num_classes=39)
    model = geffnet.efficientnet_b0(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2, num_classes=len(clusters))
    model.load_state_dict(torch.load(load_model_path))
    model.to(device)
    model.eval()

    with open(image_path, "r") as read_file:
        infos = json.load(read_file)
    with open(cluster_path, "r") as read_file:
        clusters = json.load(read_file)
    
    restaurants = [restaurant for cluster in clusters for restaurant in cluster]
    
    print("n_unique: {}".format(len(set(restaurants))))
    print("n_total: {}".format(len(restaurants)))
    print("n_has_image: {}".format(len([restaurant_id for restaurant_id in restaurants if len(infos[restaurant_id]) != 0])))
    
    for restaurant_id in tqdm(restaurants):
        
        if skip_existed and int(restaurant_id) in exist_features:
            continue
        
        images = infos[restaurant_id]
        if len(images) != 0:
            features = get_features(images, model, device)
        
            save_path = os.path.join(output_dir, "{:08d}.pt".format(int(restaurant_id)))
            torch.save(features, save_path)

if __name__ == "__main__":
    
    cluster_path = "./clusters.json"
    image_path = "./images.json"
    load_model_path = "./efficientnet_b0_v2.pth"
    
    output_dir = "./features"
    
    skip_existed = False
    use_cpu = True
    
    do_feature_generating(cluster_path, image_path, load_model_path, output_dir, skip_existed=skip_existed, use_cpu=use_cpu)