import requests
import json
from tqdm import tqdm

IP_PORT = "172.21.45.8:8000"
#IP_PORT = "192.168.1.13:8000"
TEST_URL = "http://{}/restaurant_api/test_connect".format(IP_PORT)
RESTAURANT_TABLE_URL = "http://{}/restaurant_api/get_restaurant_table".format(IP_PORT)
IMAGE_URL = "http://{}/restaurant_api/get_images".format(IP_PORT)
RESTAURANT_CHARACTERISTICS_URL = "http://{}/restaurant_api/get_restaurant_characteristics".format(IP_PORT)
STOPWORDS_URL = "http://{}/restaurant_api/get_stopWord_file".format(IP_PORT)

def test_connect(url):
    print("test connection")
    s = requests.session()
    headers = {'Content-Type': 'application/json'}
    
    connect_res = 'Fail'
    while 'Fail' in connect_res:
        # connection test
        response = s.post(url = url,
                          headers = headers,
                          data = json.dumps({}))
        connect_res = response.text

def query_db_for_restaurant_table(restaurant_table_url):
    print('get restaurant table...')
    s = requests.session()
    headers = {'Content-Type': 'application/json'}
    
    response = s.post(url = restaurant_table_url,
                      headers = headers,
                      data = json.dumps({}))
    table = json.loads(response.json().encode().decode('unicode_escape')) # {name: id}
    
    return table

def query_db_for_images(restaurant_id, image_url):
    print('get restaurant images...')
    s = requests.session()
    headers = {'Content-Type': 'application/json'}
    
    data = {"id": restaurant_id}
    response = s.post(
                            url = image_url,
                            headers = headers,
                            data = json.dumps(data)
                         )
    images = json.loads(response.json())
    
    return images

def query_db_for_characteristics(restaurant_id, characteristics_url):
    print('get restaurant characteristics...')
    s = requests.session()
    headers = {'Content-Type': 'application/json'}

    ### get single restaurant characteristics
    data = {'id':restaurant_id}
    response = s.post(
                        url = characteristics_url,
                        headers = headers,
                        data = json.dumps(data)
                     )
    
    characteristics = response.json()
    return characteristics

def download_stopwords(stopwords_path):
    print("downloading stopwords...")
    s = requests.session()
    headers = {'Content-Type': 'application/json'}
    
    response = s.post(url = STOPWORDS_URL,
                                   headers = headers,
                                   data = json.dumps({}))
    stopwords = json.loads(response.json())
    
    with open(stopwords_path, "w") as write_file:
        json.dump(stopwords, write_file)
        
def download_images(image_path):
    print("downloading images...")
    table = query_db_for_restaurant_table(RESTAURANT_TABLE_URL)
    info = {}
    for restaurant_id in tqdm(table.values()):
        images = query_db_for_images(restaurant_id, IMAGE_URL)
        info[restaurant_id] = images
    with open(image_path, "w") as write_file:
        json.dump(info, write_file)

def download_table(table_path):
    print("downloading table...")
    table = query_db_for_restaurant_table(RESTAURANT_TABLE_URL)
    with open(table_path, "w") as write_file:
        json.dump(table, write_file)

def download_characteristics(characteristic_path):
    print("downloading characteristics...")
    table = query_db_for_restaurant_table(RESTAURANT_TABLE_URL)
    info = {}
    for restaurant_id in tqdm(table.values()):
        characteristics = query_db_for_characteristics(restaurant_id, RESTAURANT_CHARACTERISTICS_URL)
        info[restaurant_id] = characteristics
    with open(characteristic_path, "w") as write_file:
        json.dump(info, write_file)

if __name__ == "__main__":
    download_characteristics("./characteristics.json")
    #download_images("./images.json")
    #download_table("./table.json")