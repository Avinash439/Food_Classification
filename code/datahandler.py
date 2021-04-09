import glob
import shutil
import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from google.cloud import storage



path_to_credentials='./credentials/training-deep-learning-models-2c295c344fe7.json'

food_classes=['bread','dairy_product','desserts','egg','fried_food',
              'meat','noodle_pasta','rice','seafood','soup','vegetable']

def split_data_into_class_folders(path_to_data,class_id):
    
    img_path = glob.glob(path_to_data + '*jpg')
    
    for path in img_path:
        basename= os.path.basename(path)
        if basename.startswith(str(class_id)+ '_'):
            path_to_save = os.path.join(path_to_data,food_classes[class_id])
            
            if not os.path.isdir(path_to_save):
                os.makedirs(path_to_save)
                
            shutil.move(path,path_to_save)



def visualize(path_to_data):
    
    image_path=[]# image paths
    labels=[]# labels for the image
    
    for r,d,f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):
                image_path.append(os.path.join(r,file))
                labels.append(os.path.basename(r))
                
    fig=plt.figure()
    
    for i in range(16):
        choosen_index=random.randint(0,len(image_path)-1)
        choosen_img=image_path[choosen_index]
        choosen_label=labels[choosen_index]
        
        ax=fig.add_subplot(4,4,i+1)
        ax.title.set_text(choosen_label)
        ax.imshow(Image.open(choosen_img))
        
    fig.tight_layout(pad=0.05)    
    plt.show()            
                

def get_images_sizes(path_to_data):
    image_path=[]
    widths= []
    heights=[]
    
    for r,d,f in os.walk(path_to_data):
        for file in f:
            if file.endswith(".jpg"):
                img=Image.open(os.path.join(r,file))
                widths.append(img.size[0])
                heights.append(img.size[1])
                img.close()
                
    mean_width=sum(widths) /len(widths) 
    mean_height= sum(heights) /len(heights)
    median_widht=np.median(widths) 
    median_height=np.median(heights)  
    return mean_height,mean_width,median_height,median_widht                   


def list_blobs(bucket_name):
    storage_client=storage.Client.from_service_account_json(path_to_credentials)#we need to make a client to access.
    blobs=storage_client.list_blobs(bucket_name)
    
    return blobs          


def download_data_to_local_Directory(bucket_name,local_directory):
    storage_client=storage.Client.from_service_account_json(path_to_credentials)
    blobs=storage_client.list_blobs(bucket_name)
    
    if not os.path.isdir(local_directory):
        os.makedirs(local_directory)       
        
        
    for blob in blobs:
        joined_path =os.path.join(local_directory,blob.name)   
        
        if os.path.basename(joined_path) == '':
            if not os.path.isdir(joined_path):# verify directory or not
                os.makedirs(joined_path)
            
        else:
            if not os.path.isfile(joined_path):
                if not os.path.isdir(os.path.dirname(joined_path)):#verifies folders
                    os.makedirs(os.path.dirname(joined_path))
                blob.download_to_filename(joined_path)
            
    
if __name__=='__main__':
    
    split_data_switch= False
    visualize_switch= False
    size_switch=False
    list_blobs_switch=False
    download_data_switch=True
    
    
    path_to_train_data='/home/avinash/udemy course/code/food-11/training/'
    path_to_validation_data='/home/avinash/udemy course/code/food-11/validation/'
    path_to_test_data='/home/avinash/udemy course/code/food-11/evaluation/'
    
    
    if split_data_switch:
        for i in range(11):
            split_data_into_class_folders(path_to_train_data,i)
            split_data_into_class_folders(path_to_validation_data,i)
            split_data_into_class_folders(path_to_test_data,i)  
    if visualize_switch:
        visualize(path_to_train_data)
    if size_switch:
        mean_width, mean_height,median_width,median_height = get_images_sizes(path_to_train_data)
        print(f"mean_width={mean_width}")
        print(f"mean_height={mean_height}")
        print(f"median_width={median_width}")
        print(f"median_heigth={median_heigth}")
        
    if list_blobs_switch:
        blobs=list_blobs('fooddata-bucket')
    
    # for blob in blobs:
    #     print(blob.name)
                   
    if download_data_switch:
        download_data_to_local_Directory('fooddata-bucket',"./data")               