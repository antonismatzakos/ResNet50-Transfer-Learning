#!/usr/bin/env python
# coding: utf-8

# In[2]:



# In[3]:


ResNet50_model = tensorflow.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), classes=5)

for layers in ResNet50_model.layers:
    layers.trainable=False


resnet50_x = Flatten()(ResNet50_model.output)
resnet50_x = Dense(256,activation='relu')(resnet50_x)
resnet50_x = Dense(5,activation='softmax')(resnet50_x)
model = Model(inputs=ResNet50_model.input, outputs=resnet50_x)


adam = optimizers.SGD(lr=0.001,momentum=0.7)
model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics=['acc'])


# In[4]:


train_path = 'C:/Users/anton/Desktop/MSc Data Science/Data Systems Project/DataSet/KneeXrayData/ClsKLData/kneeKL224/train'
valid_path = 'C:/Users/anton/Desktop/MSc Data Science/Data Systems Project/DataSet/KneeXrayData/ClsKLData/kneeKL224/val'
test_path  = 'C:/Users/anton/Desktop/MSc Data Science/Data Systems Project/DataSet/KneeXrayData/ClsKLData/kneeKL224/test'


# In[5]:


train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# In[6]:


test_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    
    horizontal_flip=True,
    fill_mode='nearest')


# In[7]:


train_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[8]:


test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[10]:


from datetime import datetime
from keras.callbacks import ModelCheckpoint



checkpoint = ModelCheckpoint(filepath='Res.h5',
                               verbose=2, save_best_only=True)

callbacks = [checkpoint]

start = datetime.now()

model_history=model.fit_generator(
  train_set,
  validation_data=test_set,
  epochs=250,
  steps_per_epoch=25,
  validation_steps=32,
    callbacks=callbacks ,verbose=2)


duration = datetime.now() - start
print("Training time: ", duration)




# Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_set, 1656 // 32+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, y_pred))
print('Classification Report')
target_names = ['0','1','2','3','4']
print(classification_report(test_set.classes, y_pred, target_names=target_names))


# In[22]:


_# Plot training & validation loss values
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('CNN Model accuracy values')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



