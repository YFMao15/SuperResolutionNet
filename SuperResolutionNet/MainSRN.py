import os
import re

import numpy as np
import tensorflow as tf

from shutil import copy
from SRNModel import SRN_model



if __name__=='__main__':
    """
    Main Function of Super Resolution Net
    Fixed parameters listed below
    """
    # Current mode of SRN, 'train' or 'test' or 'validate'
    # validation -- predicting HR image using LR image in training dataset
    # test --  predicting HR image using LR image from new dataset
    running_mode='train'
    # Net chosen for running, 'vanilla_SRN' or 'Res_SRN'
    net_chosen='vanilla_SRN'
    # Folders saving splitted data, data==blurred images, label==clear images
    # classified folder: for training and validation
    # test folder: for testing only
    original_data_folder='Original'
    label_folder='Label'
    classified_data_folder='Class'
    test_data_folder='Test'
    classified_test_folder='Test_Class'
    # Checkpoint folder
    checkpoint_folder='checkpoint'
    # Reconstruction folder
    reconstruction_folder='reconstruction'
    # Color channel of input pictures, 1 for mono-chromatic
    color_channels=1


    """
    Hyper-parameters of SRN listed below
    Change the values of hyper-parameters for better performance
    """
    # Learning rate of SRN
    learning_rate=1e-03
    decay_rate=0.95
    # List of Conv-windows
    windows=[[5,5,color_channels,128],[1,1,128,32],[3,3,32,color_channels]] 
    # size of LR sub-images, here images and labels are the same size
    # full LR image will be cropped into several sub-images for reconstruction
    # Y_size is determined by X_size and window sizes in model
    X_size=20
    # Strides of image cropping in data reading section, pixelwise
    train_crop_strides=10
    test_crop_strides=X_size-windows[0][0]-windows[1][0]-windows[2][0]+len(windows)
    # Strides of conv-layers vanilla-SRN
    conv_strides=[1,1,1,1]
    # size of full images
    # HR image size is decided by HR image size and conv strides
    LR_image_size=254
    # Number of images in batch processing, and the training iterations of data in the same batch
    batch_size=32
    batch_training_time=1000
    # Optimizer chosen for the net, 'SGD or'Adam'
    optimizer_name='SGD'
    # Maximum epoch of training
    max_epoch=1000

    """
    Testing parameters
    """
    # val_HR_image should input a tuple of length 1-3
    # Indicate the coordinate of designated tile
    val_HR_image=(10)
    


    """
    Make directories and preprocess data
    """
    home_directory=os.path.dirname(__file__)
    data_directory=os.path.dirname(__file__)
    #data_directory='/data/myf/SuperResolutionNet'


    # Classify the original data to faciliate the process
    if not os.path.exists(os.path.join(home_directory,checkpoint_folder)):
        os.makedirs(os.path.join(home_directory,checkpoint_folder))
        print('Checkpoint Folder Created')
    if not os.path.exists(os.path.join(data_directory,reconstruction_folder)):
        os.makedirs(os.path.join(data_directory,reconstruction_folder))
        print('Reconstruction Folder Created')
    if not os.path.exists(os.path.join(data_directory,reconstruction_folder,'validate')):
        os.makedirs(os.path.join(data_directory,reconstruction_folder,'validate'))
        print('Validation Reconstruction Folder Created')
    if not os.path.exists(os.path.join(data_directory,reconstruction_folder,'test')):
        os.makedirs(os.path.join(data_directory,reconstruction_folder,'test'))
        print('Test Reconstruction Folder Created')

    if not os.path.exists(os.path.join(data_directory,classified_data_folder)):
        print('Classified Data Folder Created')
        os.makedirs(os.path.join(data_directory,classified_data_folder))
        for image in os.listdir(os.path.join(data_directory,original_data_folder)):
            original_path=os.path.join(data_directory,original_data_folder,image)
            # save the splitting symbol in the result
            temp_class=image.split('_')
            if not os.path.exists(os.path.join(data_directory,classified_data_folder,temp_class[0],temp_class[1])):
                os.makedirs(os.path.join(data_directory,classified_data_folder,temp_class[0],temp_class[1]))
            classified_path=os.path.join(data_directory,classified_data_folder,temp_class[0],temp_class[1],temp_class[2])
            copy(original_path,classified_path)
        for image in os.listdir(os.path.join(data_directory,label_folder)):
            label_path=os.path.join(data_directory,label_folder,image)
            # save the splitting symbol in the result
            temp_class=image.split('_')
            if not os.path.exists(os.path.join(data_directory,classified_data_folder,temp_class[0],temp_class[1])):
                os.makedirs(os.path.join(data_directory,classified_data_folder,temp_class[0],temp_class[1]))
            classified_path=os.path.join(data_directory,classified_data_folder,temp_class[0],temp_class[1],temp_class[2])
            copy(label_path,classified_path)

    if (running_mode=='test') & (not os.path.exists(os.path.join(data_directory,test_data_folder))):
        print('Test Data Not Found')
        print('Put testing data into the new folder named %s.' % test_data_folder)
    elif (running_mode=='test') & (not os.path.exists(os.path.join(data_directory,classified_test_folder))):
        print('Classified Test Data Folder Created')
        os.makedirs(os.path.join(data_directory,classified_test_folder))
        for image in os.listdir(os.path.join(data_directory,test_data_folder)):
            test_path=os.path.join(data_directory,test_data_folder,image)
            # save the splitting symbol in the result
            temp_class=image.split('_')
            if not os.path.exists(os.path.join(data_directory,classified_test_folder,temp_class[0],temp_class[1])):
                os.makedirs(os.path.join(data_directory,classified_test_folder,temp_class[0],temp_class[1]))
            classified_path=os.path.join(data_directory,classified_test_folder,temp_class[0],temp_class[1],temp_class[2])
            copy(test_path,classified_path)

    """
    GPU Management
    """
    os.environ["CUDA_VISIBLE_DEVICES"]='1'  
    # The second GPU card is allowed to use
    config=tf.ConfigProto()  
    config.gpu_options.per_process_gpu_memory_fraction=0.7
    # 70% of GiBs are aloowed to take in the training
    config.gpu_options.allow_growth=True 


    """
    Run SRN and test its performance
    """

   
    with tf.Session(config=config) as sess:
        my_SRN=SRN_model(sess,
                running_mode,
                train_crop_strides,
                test_crop_strides,
                conv_strides,
                windows,
                learning_rate,
                decay_rate,
                optimizer_name,
                net_chosen,
                batch_size,
                batch_training_time,
                max_epoch,
                color_channels,
                X_size,
                LR_image_size,
                val_HR_image,
                checkpoint_folder,
                reconstruction_folder,
                classified_data_folder,
                classified_test_folder,
                home_directory,
                data_directory)

        my_SRN.run()



