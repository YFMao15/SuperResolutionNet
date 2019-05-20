import os
import time
import numpy as np
import tensorflow as tf

from UtilFunction import read_data
from UtilFunction import reconstruct_image
from sklearn.model_selection import KFold

class SRN_model(object):
    def __init__(self, 
            sess,
            running_mode, 
            train_crop_strides=1,  
            test_crop_strides=2,
            conv_strides=[1,1,1,1],
            windows=[[3,3,1,64],[2,2,64,32],[3,3,32,1]] ,
            learning_rate=1e-03,
            decay_rate=0.95,
            optimizer_name='SGD',
            net_chosen='vanilla_SRN',
            batch_size=32,
            batch_training_time=1000,
            max_epoch=100000,
            color_channels=1,
            X_size=64,
            LR_image_size=254,
            val_HR_image=(10,0,0),
            checkpoint_folder='checkpoint',
            reconstruction_folder='reconstruction',
            classified_data_folder='Class2',
            classified_test_folder='Test_Class2',
            home_directory=os.path.dirname(__file__),
            data_directory='/data/myf/SuperResolutionNet'):

        """
        In-class parameter mapping
        """
        self.sess=sess   
        self.running_mode=running_mode     
        self.learning_rate=learning_rate
        self.decay_rate=decay_rate
        self.windows=windows
        self.train_crop_strides=train_crop_strides
        self.test_crop_strides=test_crop_strides
        self.conv_strides=conv_strides
       
        self.optimizer_name=optimizer_name
        self.net_chosen=net_chosen
        self.X_size=X_size
        self.Y_size=self.X_size-self.windows[0][0]-self.windows[1][0]-self.windows[2][0]+len(self.windows)
        self.batch_size=batch_size
        self.batch_training_time=batch_training_time
        self.max_epoch=max_epoch
        
        self.color_channels=color_channels
        self.windows[0][2]=self.color_channels
        self.LR_image_size=LR_image_size
        self.HR_image_size=self.LR_image_size
        if self.running_mode=='train':
            self.HR_image_size=self.LR_image_size-self.windows[0][0]-self.windows[1][0]-self.windows[2][0]+len(self.windows)
        else:
            self.HR_image_size=self.test_crop_strides*((self.LR_image_size-self.windows[0][0]-self.windows[1][0]-self.windows[2][0]+len(self.windows))//self.test_crop_strides)

        self.checkpoint_folder=checkpoint_folder
        self.reconstruction_folder=reconstruction_folder
        self.classified_data_folder=classified_data_folder
        self.classified_test_folder=classified_test_folder
        self.home_directory=home_directory
        self.data_directory=data_directory

        self.val_HR_image=val_HR_image
        
        
        """
        SRN Construction
        """
        # Input placeholder saving files for SRN
        self.X=tf.placeholder(tf.float32,[None,self.X_size,self.X_size,self.color_channels],name='X')
        self.Y=tf.placeholder(tf.float32,[None,self.Y_size,self.Y_size,self.color_channels], name='Y')
        
        # Weight tensor: length, width, depth, conv_core_numbers
        self.weights={
            'w1': tf.Variable(tf.random_normal(self.windows[0], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal(self.windows[1], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal(self.windows[2], stddev=1e-3), name='w3')}
        
        # Bias tensor: conv_core_numbers
        self.biases={
            'b1': tf.Variable(tf.zeros([windows[0][3]]), name='b1'),
            'b2': tf.Variable(tf.zeros([windows[1][3]]), name='b2'),
            'b3': tf.Variable(tf.zeros([windows[2][3]]), name='b3')}
        # Y_hat is the prediction of given X under current weight and bias
        self.Y_hat=self.NN_model()

        # Mean absolute error as SRN loss function
        self.loss_function=tf.reduce_mean(tf.abs(self.Y-self.Y_hat))

        if self.optimizer_name=='SGD':
            self.optimizer=tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_function)
        elif self.optimizer_name=='Adam':
            self.optimizer=tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function)
        
        self.saver=tf.train.Saver()


    def NN_model(self):
        if self.net_chosen=='vanilla_SRN':
            # Y_hat=weights*X+biases
            # Y_hat and Y has different dimensionlities
            conv1=tf.nn.relu(tf.nn.conv2d(self.X,self.weights['w1'],strides=self.conv_strides,padding='VALID')+self.biases['b1'])
            conv2=tf.nn.relu(tf.nn.conv2d(conv1,self.weights['w2'],strides=self.conv_strides,padding='VALID')+self.biases['b2'])
            conv3=tf.nn.conv2d(conv2,self.weights['w3'],strides=self.conv_strides,padding='VALID')+self.biases['b3']
            return conv3
        elif self.net_chosen=='Res_SRN':
            # Under construction
            print('hello')

    def run(self):
        if self.running_mode=='train':
            # Initialization
            tf.global_variables_initializer().run()
            start_time=time.time()

            print("\n----- Training Process Starts -----")
            # Count on iterations
            iteration=0
            for curr_epoch in range(self.max_epoch):
                # Load new training images in every epoch
                # Return the cross-validation set of data and labels
                data,label=read_data(self.data_directory,self.LR_image_size,self.HR_image_size,self.X_size,\
                    self.Y_size,self.train_crop_strides,self.running_mode,self.classified_data_folder,curr_epoch)

                 # Cross-validation
                kf=KFold(n_splits=10,shuffle=True)
                split=list(kf.split(X=data,y=label))
                
                # Run by batch images
                fold=split[curr_epoch%10]
                training_data=data[fold[0],:,:,]
                validating_data=data[fold[1],:,:,:]
                training_label=label[fold[0],:,:,:]
                validating_label=label[fold[1],:,:,:]
                batch_num=len(training_data)//self.batch_size
                
                # Count on current sample of validating set
                valid_num=0
                for curr_batch in range(0,batch_num):
                # Train the same data repeatedly
                    for _ in range(0,self.batch_training_time):
                        X_batch=training_data[curr_batch*self.batch_size:(curr_batch+1)*self.batch_size,:,:,:]
                        Y_batch=training_label[curr_batch*self.batch_size:(curr_batch+1)*self.batch_size,:,:,:]
                        _,loss_value=self.sess.run([self.optimizer,self.loss_function],feed_dict={self.X:X_batch, self.Y:Y_batch})   
                        iteration+=1   
                        if iteration % 50==0:
                            # Here loss_value is presented with reduce_sum, which is easier to judge the convergence of SRN
                            print('Training Info--Epoch: [%d], Iteration: [%d], Pixelwise Loss: [%.6f], Time span: [%.2f]' \
                                % ((curr_epoch+1),iteration,loss_value,time.time()-start_time))
                            self.save(self.home_directory,self.checkpoint_folder,iteration)                     
                            # Here we only get the reconstruction of sub-image
                            validating_data=np.array([validating_data[valid_num]])
                            validating_label=np.array([validating_label[valid_num]])
                            val_result=self.Y_hat.eval({self.X:validating_data,self.Y:validating_label})  
                            # Show the sum of absolute errors  
                            score=np.mean(np.abs(val_result-validating_label))
                            print('Inner-training validation Info--Epoch: [%d], Iteration: [%d], Pixelwise Loss: [%.6f], Time span: [%.2f]\n' \
                                % ((curr_epoch+1),iteration,score,time.time()-start_time))
            
        elif self.running_mode=='validate':
            if not self.load(self.checkpoint_folder):
                print('Program Terminated. Double-check the ckpt files.\n')
            else:
                print('\n----- Validating Process Starts -----')
                if isinstance(self.val_HR_image,tuple):
                    if len(self.val_HR_image)==3:
                        print('Tile image reconstruction')
                        val_data,val_label=read_data(self.data_directory,self.LR_image_size,self.HR_image_size,self.X_size,\
                            self.Y_size,self.test_crop_strides,self.running_mode,self.classified_data_folder,self.val_HR_image)
                        # Y_hat.eval() will deplete all memory if testing data is not splitted into batches
                        # Process the tesing data in different batches, and merge the result together
                        # test_batch_num is the number of blocks in the prediction
                        test_batch_num=self.HR_image_size//self.test_crop_strides
                        prediction=[]
                        test_score=0
                        for curr_batch in range(test_batch_num):
                            temp_result,temp_score=self.sess.run([self.Y_hat,self.loss_function],
                                            feed_dict={self.X:val_data[curr_batch*test_batch_num:(curr_batch+1)*test_batch_num,:,:,:],
                                                    self.Y:val_label[curr_batch*test_batch_num:(curr_batch+1)*test_batch_num,:,:,:]})
                            test_score+=temp_score
                            prediction.append(temp_result)
                        batch_shape=prediction[0].shape
                        print('\nValidation Info--Pixelwise Loss: [%.6f]' % (test_score/len(prediction)))
                        print('Reconstructing the predicted tile') 
                        # Reconsturct predicted high resolution image  
                        prediction=np.reshape(np.array(prediction),(len(prediction)*batch_shape[0],batch_shape[1],batch_shape[2],batch_shape[3]))
                        reconstruct_image(self.data_directory,self.classified_data_folder,self.reconstruction_folder,self.running_mode,self.HR_image_size,self.Y_size,self.test_crop_strides,self.val_HR_image,prediction) 

                    elif len(self.val_HR_image)==2:
                        print('Sub-image reconstruction')
                        val_data,val_label=read_data(self.data_directory,self.LR_image_size,self.HR_image_size,self.X_size,\
                            self.Y_size,self.test_crop_strides,self.running_mode,self.classified_data_folder,self.val_HR_image)
                        test_batch_num=self.HR_image_size//self.test_crop_strides
                        prediction=[]
                        test_score=0
                        subimage_directory=os.path.join(self.data_directory,self.classified_data_folder,str(self.val_HR_image[0]),str(self.val_HR_image[1]))
                        tile_num=len(os.listdir(subimage_directory))//2
                        for curr_tile in range(tile_num):
                            tile_base=curr_tile*test_batch_num*test_batch_num
                            for curr_batch in range(test_batch_num):
                                temp_result,temp_score=self.sess.run([self.Y_hat,self.loss_function],
                                                feed_dict={self.X:val_data[tile_base+curr_batch*test_batch_num:tile_base+(curr_batch+1)*test_batch_num,:,:,:],
                                                        self.Y:val_label[tile_base+curr_batch*test_batch_num:tile_base+(curr_batch+1)*test_batch_num,:,:,:]})
                                test_score+=temp_score
                                prediction.append(temp_result)
                        batch_shape=prediction[0].shape
                        print('\nValidation Info--Pixelwise Loss: [%.6f]' % (test_score/len(prediction)))
                        print('Reconstructing the predicted tile') 
                        prediction=np.reshape(np.array(prediction),(len(prediction)*batch_shape[0],batch_shape[1],batch_shape[2],batch_shape[3]))
                        reconstruct_image(self.data_directory,self.classified_data_folder,self.reconstruction_folder,self.running_mode,self.HR_image_size,self.Y_size,self.test_crop_strides,self.val_HR_image,prediction) 

                elif isinstance(self.val_HR_image,int):
                        print('Full image reconstruction')
                        val_data,val_label=read_data(self.data_directory,self.LR_image_size,self.HR_image_size,self.X_size,\
                            self.Y_size,self.test_crop_strides,self.running_mode,self.classified_data_folder,self.val_HR_image)
                        test_batch_num=self.HR_image_size//self.test_crop_strides
                        prediction=[]
                        test_score=0
                        image_directory=os.path.join(self.data_directory,self.classified_data_folder,str(self.val_HR_image))
                        subimage_count=0
                        for subimage in os.listdir(image_directory):
                            subimage_directory=os.path.join(image_directory,subimage)
                            tile_num=len(os.listdir(subimage_directory))//2
                            subimage_base=subimage_count*tile_num*test_batch_num*test_batch_num
                            for curr_tile in range(tile_num):
                                tile_base=curr_tile*test_batch_num*test_batch_num
                                for curr_batch in range(test_batch_num):
                                    temp_result,temp_score=self.sess.run([self.Y_hat,self.loss_function],
                                            feed_dict={self.X:val_data[subimage_base+tile_base+curr_batch*test_batch_num:subimage_base+tile_base+(curr_batch+1)*test_batch_num,:,:,:],
                                                    self.Y:val_label[subimage_base+tile_base+curr_batch*test_batch_num:subimage_base+tile_base+(curr_batch+1)*test_batch_num,:,:,:]})
                                    test_score+=temp_score
                                    prediction.append(temp_result)
                            subimage_count+=1
                        batch_shape=prediction[0].shape
                        print('\nValidation Info--Pixelwise Loss: [%.6f]' % (test_score/len(prediction)))
                        print('Reconstructing the predicted tile') 
                        prediction=np.reshape(np.array(prediction),(len(prediction)*batch_shape[0],batch_shape[1],batch_shape[2],batch_shape[3]))
                        reconstruct_image(self.data_directory,self.classified_data_folder,self.reconstruction_folder,self.running_mode,self.HR_image_size,self.Y_size,self.test_crop_strides,self.val_HR_image,prediction) 

        elif self.running_mode=='test':
            if not self.load(self.checkpoint_folder):
                print('Program Terminated. Double-check the ckpt files.\n')
            else:
                print('\n----- Testing Process Starts -----')
                print('Vivo image reconstruction')
                test_data=read_data(self.data_directory,self.LR_image_size,self.HR_image_size,self.X_size,\
                        self.Y_size,self.test_crop_strides,self.running_mode,self.classified_test_folder)
                test_batch_num=(self.LR_image_size-self.X_size+self.test_crop_strides)//self.test_crop_strides
                prediction=[]
                subimage_count=0
                test_directory=os.path.join(self.data_directory,self.classified_test_folder)
                image_list=sorted([int(x) for x in os.listdir(test_directory)])
                for image in image_list:
                    image_directory=os.path.join(test_directory,str(image))
                    subimage_list=sorted([int(x) for x in os.listdir(image_directory)])   
                    for subimage in subimage_list:
                        subimage_directory=os.path.join(image_directory,str(subimage))
                        tile_num=len(os.listdir(subimage_directory))
                        subimage_base=subimage_count*tile_num*test_batch_num*test_batch_num
                        for curr_tile in range(tile_num):
                            tile_base=curr_tile*test_batch_num*test_batch_num
                            for curr_batch in range(test_batch_num):
                                temp_result=self.sess.run(self.Y_hat,
                                    feed_dict={self.X:test_data[subimage_base+tile_base+curr_batch*test_batch_num:subimage_base+tile_base+(curr_batch+1)*test_batch_num,:,:,:]})
                                prediction.append(temp_result)
                        subimage_count+=1
                batch_shape=prediction[0].shape
                print('Reconstructing the predicted tile') 
                prediction=np.reshape(np.array(prediction),(len(prediction)*batch_shape[0],batch_shape[1],batch_shape[2],batch_shape[3]))
                reconstruct_image(self.data_directory,self.classified_test_folder,self.reconstruction_folder,self.running_mode,self.HR_image_size,self.Y_size,self.test_crop_strides,self.classified_test_folder,prediction) 


    def save(self,home_directory,checkpoint_folder,iteration):
        print('Now saving to checkpoint of %d iteration' % iteration)
        # Naming: (X_size Y_size batch_size) op_name
        model_name='('+str(self.X_size)+'_'+str(self.windows[0][0])+'_'+str(self.windows[1][0])+'_'+str(self.windows[2][0])+')'+self.optimizer_name+'.model'
        model_folder=self.net_chosen
        checkpoint_directory=os.path.join(home_directory,checkpoint_folder,model_folder)

        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        self.saver.save(self.sess,
                os.path.join(checkpoint_directory,model_name),
                global_step=iteration)
        print('Checkpoint sucessfully created\n')

    def load(self,checkpoint_folder):
        print('\nLoading Checkpoint in folder %s' % checkpoint_folder)
        checkpoint_directory=os.path.join(self.home_directory,checkpoint_folder,self.net_chosen)
        ckpt=tf.train.get_checkpoint_state(checkpoint_directory)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name=os.path.basename(ckpt.model_checkpoint_path)
            print('Restore Checkpoint file %s\n' % ckpt_name)
            self.saver.restore(self.sess,os.path.join(checkpoint_directory,ckpt_name))
            print('Checkpoint sucessfully restored')
            return True
        else:
            print('Checkpoint NOT FOUND!')
            return False




        