from PIL import Image
import geffnet
import torch
import json
import os
from dataset import get_transforms
from torch.nn import CosineSimilarity
import time
import io
import base64

def do_recommend(query, 
                 top_n=None,
                 size=224,
                 top_k=1,
                 load_model_path="./efficientnet_b0_v1.pth",
                 cluster_path="./clusters.json",
                 feature_dir="./features"):
                
    with open(cluster_path, "r") as read_file:
        clusters = json.load(read_file)
    num_classes = len(clusters)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Load Image")
    t0 = time.time()
    
    image = Image.open(io.BytesIO(base64.b64decode(query.encode("utf-8"))))
    transform = get_transforms(False)
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    
    print("Load Model")
    model = geffnet.efficientnet_b0(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2, num_classes=num_classes)
    model.load_state_dict(torch.load(load_model_path))
    model.to(device)
    model.eval()
    
    print("Predict Category")
    outputs = model(image)
    #_, predictions = torch.max(outputs.data, 1)
    #category_id = predictions.item()
    _, predictions = torch.topk(outputs.data, top_k, dim=1)
    category_ids = predictions.tolist()[0]

    print("category ids: {}".format(category_ids))
        
    cluster_ids = [restaurant_id for category_id in category_ids for restaurant_id in clusters[category_id]]
    
    #print(cluster_id)
    print("Feature Matching")
    target_feature = model.features(image)
    cluster_features = [torch.load(os.path.join(feature_dir, "{:08d}.pt".format(int(cluster_id))))
                        if os.path.exists(os.path.join(feature_dir, "{:08d}.pt".format(int(cluster_id)))) else None
                        for cluster_id in cluster_ids
                        ]
    
    cos = CosineSimilarity(dim=1, eps=1e-6)
    
    cluster_sims = [torch.max(cos(torch.flatten(target_feature, start_dim=1).to(device), 
                                                 torch.flatten(cluster_feature, start_dim=1).to(device))).item()
                    if cluster_feature != None else -1
                    for cluster_feature in cluster_features
                    ]
    
    ranked_cluster_ids = [x for x, _ in sorted(zip(cluster_ids, cluster_sims))][::-1]
    ranked_cluster_sims = sorted(cluster_sims)[::-1]
    
    print("Time Elapsed: {:.2f} sec.".format(time.time() - t0))
    
    if top_n is None:
        return ranked_cluster_ids, ranked_cluster_sims
    else:
        return ranked_cluster_ids[:top_n], ranked_cluster_sims[:top_n]

if __name__ == "__main__":
    
    # temp for show
    image_path = "./images/sushi2.jpg"
    
    image = Image.open(image_path)
    image_query = io.BytesIO()
    image.save(image_query, format="JPEG")
    image_query = image_query.getvalue()
    image_query = base64.b64encode(image_query).decode("utf-8")
    
    ranked_cluster_ids, ranked_cluster_sims = do_recomend(image_query)
    
    #print(ranked_cluster_ids)
    #print(ranked_cluster_sims)
    """temp for show"""
    table_path = "./table.json"
    with open(table_path, "r") as read_file:
        table = json.load(read_file)
    name_table = {v: k for k, v in table.items()}
    
    cluster_names = [name_table[int(restaurant_id)] for restaurant_id in ranked_cluster_ids]
    ranked_cluster_names = [x for x, _ in sorted(zip(cluster_names, ranked_cluster_sims))][::-1]
    print(ranked_cluster_names)