import os 
import re
import math
import numpy as np

import matplotlib.image as img
import scipy.misc as misc

from skimage import io
from skimage import img_as_float32
from skimage import img_as_uint
from skimage.exposure import rescale_intensity


def crop_image(data,image_size,sub_size,crop_strides):
    block_list=[]
    max_range=crop_strides*((image_size-sub_size)//crop_strides)
    scanning_num=max_range//crop_strides
    for row in range(0,max_range,crop_strides):
        for col in range(0,max_range,crop_strides): 
            crop=data[row:row+sub_size,col:col+sub_size]
            block_list.append(crop)
    block_array=np.array(block_list)
    block_array=np.reshape(block_array,(int(scanning_num*scanning_num),sub_size,sub_size,1))
    return block_array

# Merge the croppped reconstruction sub-images to reconstruction result
def reconstruct_image(data_directory,classified_data_folder,reconstruction_folder,running_mode,image_size,sub_size,crop_strides,*args):
    if running_mode=='validate':
        HR_image=args[0]
        prediction=args[1]
        # mask reconstructs overlapped image
        mask=np.zeros((image_size,image_size,prediction.shape[3]))
        sub_mask=np.ones((sub_size,sub_size,prediction.shape[3]))
        max_range=crop_strides*((image_size-sub_size)//crop_strides)
        for row in range(0,max_range,crop_strides):
            for col in range(0,max_range,crop_strides): 
                mask[row:row+sub_size:,col:col+sub_size,:]+=sub_mask
        mask=np.true_divide(np.ones((image_size,image_size,prediction.shape[3])),mask)
        # Reconstruct images of different sizes based on differeent levels of validated image
        if isinstance(HR_image,tuple):
            if len(HR_image)==3:
                temp_image=np.zeros((image_size,image_size,prediction.shape[3]))
                scanning_num=max_range//crop_strides
                for count in range(prediction.shape[0]):
                    row=count//scanning_num
                    col=count%scanning_num
                    temp_image[row*crop_strides:row*crop_strides+sub_size,col*crop_strides:col*crop_strides+sub_size,:]+=prediction[count,:,:,:]
                # Neutralize overlapping effects
                output_image=mask*temp_image
                if prediction.shape[3]==1:
                    output_image=np.reshape(output_image,(image_size,image_size))
                reconstruction_directory=os.path.join(data_directory,reconstruction_folder)
                io.imsave(os.path.join(reconstruction_directory,'validate',str(HR_image[0])+'_'+str(HR_image[1])+'_'+str(HR_image[2])+'_predict.tif'),img_as_uint(rescale_intensity(output_image)))

    
            elif len(HR_image)==2:
                subimage_directory=os.path.join(data_directory,classified_data_folder,str(HR_image[0]),str(HR_image[1]))
                scanning_num=max_range//crop_strides
                tile_count=0
                for tile in os.listdir(subimage_directory):
                    if tile.split('.')[0][-1]=='a':
                        temp_image=np.zeros((image_size,image_size,prediction.shape[3]))
                        for count in range(scanning_num*scanning_num):
                            row=count//scanning_num
                            col=count%scanning_num
                            temp_image[row*crop_strides:row*crop_strides+sub_size,col*crop_strides:col*crop_strides+sub_size,:]+=prediction[tile_count*scanning_num*scanning_num+count,:,:,:]
                        output_image=mask*temp_image
                        if prediction.shape[3]==1:
                            output_image=np.reshape(output_image,(image_size,image_size))
                        reconstruction_directory=os.path.join(data_directory,reconstruction_folder)
                        io.imsave(os.path.join(reconstruction_directory,'validate',str(HR_image[0])+'_'+str(HR_image[1])+'_'+tile.split('.')[0][0:-1]+'_predict.tif'),img_as_uint(rescale_intensity(output_image)))
                        tile_count+=1


        elif isinstance(HR_image,int):
            image_directory=os.path.join(data_directory,classified_data_folder,str(HR_image))
            subimage_list=sorted([int(x) for x in os.listdir(image_directory)])
            scanning_num=max_range//crop_strides
            tile_count=0
            for subimage in subimage_list:
                subimage_directory=os.path.join(image_directory,str(subimage))
                for tile in os.listdir(subimage_directory):
                    if tile.split('.')[0][-1]=='a':
                        temp_image=np.zeros((image_size,image_size,prediction.shape[3]))
                        for count in range(scanning_num*scanning_num):
                            row=count//scanning_num
                            col=count%scanning_num
                            temp_image[row*crop_strides:row*crop_strides+sub_size,col*crop_strides:col*crop_strides+sub_size,:]+=prediction[tile_count*scanning_num*scanning_num+count,:,:,:]
                        output_image=mask*temp_image
                        if prediction.shape[3]==1:
                            output_image=np.reshape(output_image,(image_size,image_size))
                        reconstruction_directory=os.path.join(data_directory,reconstruction_folder)
                        io.imsave(os.path.join(reconstruction_directory,'validate',str(HR_image)+'_'+str(subimage)+'_'+tile.split('.')[0][0:-1]+'_predict.tif'),img_as_uint(rescale_intensity(output_image)))
                        tile_count+=1

    elif running_mode=='test':
        classified_test_folder=args[0]
        prediction=args[1]
        max_range=crop_strides*((image_size-sub_size)//crop_strides)
        mask=np.zeros((image_size,image_size,prediction.shape[3]))
        sub_mask=np.ones((sub_size,sub_size,prediction.shape[3]))
        for row in range(0,max_range,crop_strides):
            for col in range(0,max_range,crop_strides): 
                mask[row:row+sub_size:,col:col+sub_size,:]+=sub_mask
        mask=np.true_divide(np.ones((image_size,image_size,prediction.shape[3])),mask)

        test_directory=os.path.join(data_directory,classified_test_folder)
        image_list=sorted([int(x) for x in os.listdir(test_directory)])
        scanning_num=max_range//crop_strides
        tile_count=0
        for image in image_list:
            image_directory=os.path.join(test_directory,str(image))
            if not os.path.exists(os.path.join(data_directory,reconstruction_folder,'test',str(image))):
                os.makedirs(os.path.join(data_directory,reconstruction_folder,'test',str(image)))
            subimage_list=sorted([int(x) for x in os.listdir(image_directory)])        
            for subimage in subimage_list:
                subimage_directory=os.path.join(image_directory,str(subimage))
                if not os.path.exists(os.path.join(data_directory,reconstruction_folder,'test',str(image),str(subimage))):
                    os.makedirs(os.path.join(data_directory,reconstruction_folder,'test',str(image),str(subimage)))
                for tile in os.listdir(subimage_directory):
                    temp_image=np.zeros((image_size,image_size,prediction.shape[3]))
                    for count in range(scanning_num*scanning_num):
                        row=count//scanning_num
                        col=count%scanning_num
                        temp_image[row*crop_strides:row*crop_strides+sub_size,col*crop_strides:col*crop_strides+sub_size,:]+=prediction[tile_count*scanning_num*scanning_num+count,:,:,:]
                    output_image=mask*temp_image
                    if prediction.shape[3]==1:
                        output_image=np.reshape(output_image,(image_size,image_size))
                    reconstruction_directory=os.path.join(data_directory,reconstruction_folder)
                    io.imsave(os.path.join(reconstruction_directory,'test',str(image),str(subimage),tile.split('.')[0][0:-1]+'_predict.tif'),img_as_uint(rescale_intensity(output_image)))
                    tile_count+=1
        


def read_data(data_directory,LR_image_size,HR_image_size,X_size,Y_size,crop_strides,running_mode,*args):
    if running_mode=='train':
        classified_data_folder=args[0]
        curr_epoch=args[1]
        classified_directory=os.path.join(data_directory,classified_data_folder)
        
        # label image <=> HR image
        # data image <=> LR image
        LR_full_list=[]
        HR_full_list=[]
        #image=str((curr_epoch//1024)%24+1)
        #subimage=str(curr_epoch//32)
        #subimage_directory=os.path.join(classified_directory,image,subimage)
        subimage_directory=os.path.join(classified_directory,'10','0')
        for tile in os.listdir(subimage_directory):
            if tile.split('.')[0][-1]=='a':
                HR_tile=img_as_float32(img.imread(os.path.join(subimage_directory,tile),format='.tif'))
                HR_tile_array=crop_image(HR_tile,HR_image_size,Y_size,crop_strides)
                HR_full_list.append(HR_tile_array)
            elif tile.split('.')[0][-1]=='b':
                LR_tile=img_as_float32(img.imread(os.path.join(subimage_directory,tile),format='.tif'))
                LR_tile_array=crop_image(LR_tile,LR_image_size,X_size,crop_strides)
                LR_full_list.append(LR_tile_array)

        LR_shape=LR_tile_array.shape
        HR_shape=HR_tile_array.shape
        LR_full_array=np.reshape(np.array(LR_full_list),(len(LR_full_list)*LR_shape[0],LR_shape[1],LR_shape[2],LR_shape[3]))
        HR_full_array=np.reshape(np.array(HR_full_list),(len(HR_full_list)*HR_shape[0],HR_shape[1],HR_shape[2],HR_shape[3]))
        return LR_full_array,HR_full_array
    
    elif running_mode=='validate':
        classified_data_folder=args[0]
        val_image=args[1]
        # Validating different levels of images
        if isinstance(val_image,tuple):
            if len(val_image)==3:
                val_directory=os.path.join(data_directory,classified_data_folder)
                tile_directory=os.path.join(val_directory,str(val_image[0]),str(val_image[1]))
                HR_tile=img_as_float32(img.imread(os.path.join(tile_directory,str(val_image[2])+'a.tif'),format='.tif'))
                HR_tile_array=crop_image(HR_tile,HR_image_size,Y_size,crop_strides)
                LR_tile=img_as_float32(img.imread(os.path.join(tile_directory,str(val_image[2])+'b.tif'),format='.tif'))
                LR_tile_array=crop_image(LR_tile,LR_image_size,X_size,crop_strides)
                return LR_tile_array,HR_tile_array
            
            elif len(val_image)==2:
                val_directory=os.path.join(data_directory,classified_data_folder)
                subimage_directory=os.path.join(val_directory,str(val_image[0]),str(val_image[1]))
                HR_subimage_list=[]
                LR_subimage_list=[]
                for tile in os.listdir(subimage_directory):
                    tile_name=tile.split('.')
                    if tile_name[0][-1]=='a':
                        HR_tile=img_as_float32(img.imread(os.path.join(subimage_directory,str(tile)),format='.tif'))
                        HR_tile_array=crop_image(HR_tile,HR_image_size,Y_size,crop_strides)
                        HR_subimage_list.append(HR_tile_array)
                    elif tile_name[0][-1]=='b':
                        LR_tile=img_as_float32(img.imread(os.path.join(subimage_directory,str(tile)),format='.tif'))
                        LR_tile_array=crop_image(LR_tile,LR_image_size,X_size,crop_strides)
                        LR_subimage_list.append(LR_tile_array)
                HR_shape=HR_tile_array.shape
                LR_shape=LR_tile_array.shape
                HR_subimage_array=np.reshape(np.array(HR_subimage_list),(len(HR_subimage_list)*HR_shape[0],HR_shape[1],HR_shape[2],HR_shape[3]))
                LR_subimage_array=np.reshape(np.array(LR_subimage_list),(len(LR_subimage_list)*LR_shape[0],LR_shape[1],LR_shape[2],LR_shape[3]))
                return LR_subimage_array,HR_subimage_array

        elif isinstance(val_image,int):
            val_directory=os.path.join(data_directory,classified_data_folder)
            image_directory=os.path.join(val_directory,str(val_image))
            HR_image_list=[]
            LR_image_list=[]
            subimage_list=sorted([int(x) for x in os.listdir(image_directory)])
            for subimage in subimage_list:
                subimage_directory=os.path.join(image_directory,str(subimage))
                for tile in os.listdir(subimage_directory):
                    tile_name=tile.split('.')
                    if tile_name[0][-1]=='a':
                        HR_tile=img_as_float32(img.imread(os.path.join(subimage_directory,str(tile)),format='.tif'))
                        HR_tile_array=crop_image(HR_tile,HR_image_size,Y_size,crop_strides)
                        HR_image_list.append(HR_tile_array)
                    elif tile_name[0][-1]=='b':
                        LR_tile=img_as_float32(img.imread(os.path.join(subimage_directory,str(tile)),format='.tif'))
                        LR_tile_array=crop_image(LR_tile,LR_image_size,X_size,crop_strides)
                        LR_image_list.append(LR_tile_array)
            HR_shape=HR_tile_array.shape
            LR_shape=LR_tile_array.shape
            HR_image_array=np.reshape(np.array(HR_image_list),(len(HR_image_list)*HR_shape[0],HR_shape[1],HR_shape[2],HR_shape[3]))
            LR_image_array=np.reshape(np.array(LR_image_list),(len(LR_image_list)*LR_shape[0],LR_shape[1],LR_shape[2],LR_shape[3]))
            return LR_image_array,HR_image_array

    elif running_mode=='test':
        # Testing on the vivo image of 100+ million pixels
        classified_test_folder=args[0]
        test_directory=os.path.join(data_directory,classified_test_folder)
        image_list=sorted([int(x) for x in os.listdir(test_directory)])
        LR_image_list=[]
        for image in image_list:
            image_directory=os.path.join(test_directory,str(image))
            subimage_list=sorted([int(x) for x in os.listdir(image_directory)])
            for subimage in subimage_list:
                subimage_directory=os.path.join(image_directory,str(subimage))
                for tile in os.listdir(subimage_directory):
                    tile_name=tile.split('.')
                    LR_tile=img_as_float32(img.imread(os.path.join(subimage_directory,str(tile)),format='.tif'))
                    LR_tile_array=crop_image(LR_tile,LR_image_size,X_size,crop_strides)
                    LR_image_list.append(LR_tile_array)
        LR_shape=LR_tile_array.shape
        LR_image_array=np.reshape(np.array(LR_image_list),(len(LR_image_list)*LR_shape[0],LR_shape[1],LR_shape[2],LR_shape[3]))
        return LR_image_array


