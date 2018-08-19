
# coding: utf-8

# In[1]:


import numpy as np #module pour faire des maths et manipuler des tenseurs
import itertools #truc random pour faire marcher les graphs

import keras #le module de deep learning
from keras import backend as K #support de Keras (Tensorflow)
from keras.models import Sequential, load_model, Model # de quoi créer un modèle (et en charger un, au besoin)

from keras.layers.core import Dense, Flatten # des couches de neurones simples
from keras.layers.normalization import BatchNormalization # du preprocessing d'image
from keras.layers.convolutional import * # les résaux convolutifs pour l'analyse d'image

from keras.applications import imagenet_utils # du preprocessing

#import tensorflowjs as tfjs # tensorflow js (conversion à la fin pour le web)

from keras.optimizers import Adam # sous programme responsable de la backpropagation
from keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy #différente fonctions de coûts
from keras.preprocessing.image import ImageDataGenerator # du preprocessing

from matplotlib import pyplot as plt # des graphiques
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix # des graphiques

from keras.utils.generic_utils import CustomObjectScope

print("MODULES : IMPORTATION COMPLETE")


# In[2]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[3]:


def plots(ims,figsize=(12,6),rows=1,interp=False,titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if(ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
            
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims)%2 == 0 else len(ims)// rows+1
    for i in range(len(ims)):
        sp = f.add_subplot(rows,cols,i+1)
        sp.axis("Off")
        if titles is not None:
            sp.set_title(titles[i],fontsize=16)
        plt.imshow(ims[i],interpolation=None if interp else "none")


# In[4]:


with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = load_model('deepSoviet_next.h5')
model.summary() # on regarde son architecture
print("DEEPSOVIET : IMPORTATION COMPLETE")


# In[5]:


train_path = "./train"
valid_path = "./valid"
test_path = "./test"

train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path,target_size=(224,224), classes=["Communist","Other"],batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(valid_path,target_size=(224,224),classes=["Communist","Other"],batch_size=5)
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path,target_size=(224,224),classes=["Communist","Other"],batch_size=10, shuffle=False)


# In[6]:


model.fit_generator(train_batches,steps_per_epoch=69,validation_data=valid_batches,validation_steps=5,epochs=7,verbose=1)


# In[7]:


test_imgs, test_labels = next(test_batches) #affichage des images de test
plots(test_imgs, titles=test_labels)


# In[8]:


predictions = model.predict_generator(test_batches,steps=1,verbose=2) #test
print("PREDICTION SUR LE TESTSET :")
print(np.round(np.multiply(predictions,100)))


# In[9]:


test_labels = test_labels[:,0]
cm = confusion_matrix(test_labels,np.round(predictions[:,0]))
cm_plot_labels = ["Communist","Other"]
plot_confusion_matrix(cm,cm_plot_labels,title="Confusion Matrix") #matrice de confusion pour évaluer l'IA


# In[10]:


model.save("deepSoviet_next.h5")


# In[11]:


#tfjs.converters.save_keras_model(model,"TFJS_NEXT")

