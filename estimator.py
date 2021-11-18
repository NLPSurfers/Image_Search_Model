import os
from recommend import do_recommend
from clustering import do_clustering
from train import do_training
from generate_features import do_feature_generating

class ModelOutput:
    
    def __init__(self, status, outputs):
        self.status = status
        self.outputs = outputs

class ImageSearchModel:

    def __init__(self, 
                 size=224,
                 top_k=1,
                 min_samples=10,
                 num_epochs=40,
                 batch_size=32,  
                 learning_rate=1e-4,
                 characteristics_path="./data/characteristics.json",
                 stopwords_path="./data/stopwords.json",
                 load_model_path="./weights/efficientnet_b0_v1.pth",
                 cluster_path="./data/clusters.json",
                 feature_dir="./data/features",
                 table_path="./data/table.json",
                 image_path="./data/images.json",
                 download_enable=True):
        self.size = size
        self.top_k = top_k
        self.min_samples = min_samples
        self.characteristics_path = characteristics_path
        self.stopwords_path = stopwords_path
        self.load_model_path = load_model_path
        self.cluster_path = cluster_path
        self.feature_dir = feature_dir
        self.table_path = table_path
        self.image_path = image_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.download_enable = download_enable
        
        # check for directories
        if not os.path.exists(os.path.dirname(self.characteristics_path)):
            os.makedirs(os.path.dirname(self.characteristics_path))
        if not os.path.exists(os.path.dirname(self.stopwords_path)):
            os.makedirs(os.path.dirname(self.stopwords_path))
        if not os.path.exists(os.path.dirname(self.load_model_path)):
            os.makedirs(os.path.dirname(self.load_model_path))
        if not os.path.exists(os.path.dirname(self.cluster_path)):
            os.makedirs(os.path.dirname(self.cluster_path))
        if not os.path.exists(os.path.dirname(self.feature_dir)):
            os.makedirs(os.path.dirname(self.feature_dir))
        if not os.path.exists(os.path.dirname(self.table_path)):
            os.makedirs(os.path.dirname(self.table_path))
        if not os.path.exists(os.path.dirname(self.image_path)):
            os.makedirs(os.path.dirname(self.image_path))
        
    def inference(self, query, top_n=None):
        ranked_cluster_ids, ranked_cluster_sims = do_recommend(query,
                                                               top_n,
                                                               size=self.size,
                                                               top_k=self.top_k,
                                                               load_model_path=self.load_model_path,
                                                               cluster_path=self.cluster_path,
                                                               feature_dir=self.feature_dir)
        
        outputs = [(int(id_), score_) for id_, score_ in zip(ranked_cluster_ids, ranked_cluster_sims)]
        return ModelOutput(None, outputs)
        
    def update(self):
        # clustering
        do_clustering(self.characteristics_path, 
                                self.stopwords_path, 
                                self.cluster_path, 
                                self.min_samples, 
                                self.download_enable)
        # train classifier
        do_training(table_path=self.table_path,
                    image_path=self.image_path,
                    cluster_path=self.cluster_path,
                    save_model_path=self.load_model_path,
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate, 
                    download_enable=self.download_enable)                    
        # generate features
        do_feature_generating(cluster_path=self.cluster_path, 
                              image_path=self.image_path,
                              load_model_path=self.load_model_path,
                              output_dir=self.feature_dir)

if __name__ == "__main__":
    from PIL import Image
    import io
    import base64
    
    image_path = "./images/sushi.jpg"
    
    image = Image.open(image_path)
    image_query = io.BytesIO()
    image.save(image_query, format="JPEG")
    image_query = image_query.getvalue()
    image_query = base64.b64encode(image_query).decode("utf-8")
    
    model = ImageSearchModel()
    model.update()
    #output = model.inference(image_query, 5)
    #print(output.outputs)