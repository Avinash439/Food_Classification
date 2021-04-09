from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report, confusion_matrix 
from datahandler import download_data_to_local_Directory
import os
import numpy as np
import argparse
from tensorflow.python.client import device_lib


print("Tensor flow running on the following devices")
print(device_lib.list_local_devices())#show the devices available for  training, tensorflow is running on whether docker or virtual environment,cpu or gpu

def build_model(classes):
    base_model=InceptionV3(weights="imagenet",include_top=False, input_tensor=Input(shape=(229,229,3)))
    
    output_model=base_model.output
    
    output_model=Flatten()(output_model)
    
    output_model=Dense(512)(output_model)# hidden units are of ur choice
    
    output_model=Dropout(0.5)(output_model)# 0.5 is hyperparameter we have to choose according to our results
    
    output_model=Dense(classes,activation="softmax")(output_model)# activation fucntion that maps input to output since it is multiple classes we used softmax, if it is binary we can use sigmoid activation.
    
    model=Model(inputs=base_model.input,outputs=output_model)
    
    
    for layer in base_model.layers:
        layer.trainable = False
        

    
    return model


def data_pipeline(batch_size,train_Data_path,validation_Data_path,evaluation_Data_path):
    
    train_augmentor= ImageDataGenerator(rescale=1./255, rotation_range=25,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,
                                        shear_range=0.15,horizontal_flip=True,fill_mode="nearest") #inceptionv3 model actually takes input of images that have values between 0 and 1, and our original images have values between 0 and 255 so we have to divide by 255.  
    
    

    
    val_augmentor= ImageDataGenerator(rescale=1./255)
    
    
    eval_augmentor= ImageDataGenerator(rescale=1./255) 
    train_generator=train_augmentor.flow_from_directory(
        train_Data_path,
        class_mode="categorical",
        target_size=(229,229),                                                                                                                                              
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
        
    )
    
    
    validation_generator=val_augmentor.flow_from_directory(
        validation_Data_path,
        class_mode="categorical",
        target_size=(229,229),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )


     
    evaluation_generator=eval_augmentor.flow_from_directory(
        evaluation_Data_path,
        class_mode="categorical",
        target_size=(229,229),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )
        
    return train_generator,evaluation_generator,validation_generator
    
 
 
 
def get_no_of_images_inside_folder(directory):
    
    count=0
    for root,dirnames,filesnames in os.walk(directory):
        for filename in filesnames:
            _, ext=os.path.splitext(filename)
            if ext in [".png",".jpg",".jpeg"]:
                count=count+1
    return count             
    
 
    
    
def Train(path_to_data,batch_size,epochs):
    path_to_train_data=os.path.join(path_to_data,'training')
    path_to_validation_data=os.path.join(path_to_data,'validation')
    path_to_evaluation_data=os.path.join(path_to_data,'evaluation')
    
    
    total_train_images=get_no_of_images_inside_folder(path_to_train_data)
    total_validation_images=get_no_of_images_inside_folder(path_to_validation_data)
    total_evaluation_images=get_no_of_images_inside_folder(path_to_evaluation_data)
    
    
    print("[INFO] train images : ", total_train_images)
    
    
    
    train_generator,evaluation_generator,validation_generator=data_pipeline(batch_size=batch_size,
                                                                            train_Data_path=path_to_train_data,
                                                                            validation_Data_path=path_to_validation_data,
                                                                            evaluation_Data_path=path_to_evaluation_data)
    
    
    classes_dic=train_generator.class_indices
    model=build_model(classes=len(classes_dic.keys()))  
    
    optimizer=Adam(lr=1e-5)#small learning rate we can play with this to see how values are changed.
    
    model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=['accuracy'])#classification tast all classes are independent from each other. but if we have classes that colapses each other then  it changes.
    
    model.fit_generator(train_generator,steps_per_epoch=total_train_images//batch_size,
                        validation_data=validation_generator,
                        validation_steps=total_validation_images//batch_size,
                        epochs=epochs)#version 2.2 tensorflow. epoch means the model has finished going to the data set once.
    
    print("[INFO] Evaluation phase...")

    predictions = model.predict_generator(evaluation_generator)
    predictions_idxs = np.argmax(predictions, axis=1)

    my_classification_report = classification_report(evaluation_generator.classes, predictions_idxs, 
                                                     target_names=evaluation_generator.class_indices.keys())

    my_confusion_matrix = confusion_matrix(evaluation_generator.classes, predictions_idxs)

    print("[INFO] Classification report : ")
    print(my_classification_report)

    print("[INFO] Confusion matrix : ")
    print(my_confusion_matrix)     
        
    
          
if __name__=='__main__':
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--bucket_name",type=str,help="bucket name on google cloud storage",default="fooddata-bucket")
    parser.add_argument("--batch_size",type=int,help="batch size used for training deeplearning model",default=2)
    
    args=parser.parse_args()
    print("downloading data")
    download_data_to_local_Directory(args.bucket_name,"./data")
    print("downloading finished")   
            
    path_to_data='/home/avinash/udemy course/code/food-11/'
    Train(path_to_data,args.batch_size,1)
