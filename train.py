import json
import torch
from torch.nn import CrossEntropyLoss
import geffnet
from dataset import RestaurantImageDataset
from sampler import ImbalancedDatasetSampler
import numpy as np
from db_control import download_table, download_images

def do_training(table_path,
                image_path,
                cluster_path,
                save_model_path,
                num_epochs,
                batch_size,
                learning_rate,
                load_model_path=None,
                download_enable=True):
    
    if download_enable:
        download_table(table_path)
        download_images(image_path)
    
    with open(cluster_path, "r") as read_file:
        clusters = json.load(read_file)    
    num_classes = len(clusters)
    
    print("*** Prepare Dataset ***")
    dataset = RestaurantImageDataset(table_path, image_path, cluster_path)
    train_loader = torch.utils.data.DataLoader(dataset, sampler=ImbalancedDatasetSampler(dataset), batch_size=batch_size, shuffle=False)
    
    print("*** Setup Efficientnet Model ***")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = geffnet.efficientnet_b0(pretrained=True, drop_rate=0.25, drop_connect_rate=0.2, num_classes=num_classes)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
    model.to(device)
    model.train()
    
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    
    print("*** Train Efficientnet Model ***")
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            # Run the forward pass
            images=images.float().to(device)
            labels=labels.float().to(device)
            
            outputs = model(images)
            
            #print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels.long())            
            loss_list.append(loss.item())
            # Backprop and perform Adam optimisation
            
            loss.backward()
            optimizer.step()
            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.long()).sum().item()
            acc_list.append(correct / total)

            if (i+1) % 5 == 0 or i==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

    print("*** Save Efficientnet Model ***")
    # Save model
    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
        
if __name__ == "__main__":
    download_enable = False
    table_path = "./table.json"
    image_path = "./images.json"
    cluster_path = "./clusters.json"
    
    load_model_path = None
    save_model_path = "./efficientnet_b0_v0.pth"
    
    num_epochs = 20
    batch_size = 32    
    learning_rate = 1e-4
    
    do_training(table_path,
                         images_path,
                         cluster_path,
                         save_model_path,
                         num_epochs,
                         batch_size,
                         learning_rate,
                         load_model_path=load_model_path,
                         download_enable=save_model_path)