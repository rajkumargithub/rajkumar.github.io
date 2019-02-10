---
layout: post
title: Abnormality Detection in Musculoskeletal Radiographs
---
## Objective
The objective is to build a machine learning model that can detect an abnormality in the X-Ray radiographs. These models can help towards providing healthcare access to the parts of the world where access to skilled radiologists is limited. According to a study on the Global Burden of Disease and the worldwide impact of all diseases found that, _“musculoskeletal conditions affect more than 1.7 billion people worldwide. They are the 2nd greatest cause of disabilities, and have the 4th greatest impact on the overall health of the world population when considering both death and disabilities”_. (www.usbji.org, n.d.).

Stanford University Machine Learning Group has published a paper related to this problem and provided one of the world’s largest public radiographic images dataset called MURA. MURA is short for Musculoskeletal Radiographs. Stanford university ML group has used DenseNet169 algorithm to train a deep neural network which can detect abnormalities in the radiographs with the accuracy closer to the top radiologists. This project attempts to implement deep neural network using DenseNet169 inspired from the Stanford Paper [Rajpurkar, et al., 2018](https://arxiv.org/abs/1712.06957).

## Dataset
As per the paper _“MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs”_, the data is collected de-identified, HIPAA-compliant images from the Pictures Archive and Communication System (PACS) of Stanford Hospital. It consists of 14,863 studies from 12,173 patients, with a total of 40,561 multi-view radiographic images. Each belongs to one of seven Stanford upper extremity radiographic study types: elbow, finger, forearm, humerus, shoulder and wrist. Each study was manually labeled as normal or abnormal by board-certified radiologists from the Stanford Hospital at the time of clinical radiographic interpretation in the diagnostic radiology environment between 2001 and 2012. The labeling was performed during interpretation on DICOM images presented on at least 3 megapixel PACS medical grade display and max luminance 400 cd/m2 and min luminance 1 cd/m2 with pixel size of 0.2 and native resolution of 1500 x 2000 pixels. The clinical images vary in resolution and in aspect ratios. The dataset has 9,045 normal and 5,818 abnormal musculoskeletal radiographic studies. [Rajpurkar, et al., 2018](https://arxiv.org/abs/1712.06957).

## Data Collection
Stanford review board approved study collected de-idenified, HIPAA-compliant images from the Picture Archive and Communication Systems (PACS) for Stanford Hospital. The assembled a dataset of musculoskeletal radiographs consists of 14,863 studies from 12,173 patients, with the total of 40,561 multi-view radiographic images. Each belongs to one of seven standard upper extremity radiographic study types: elbow, finger, forearm, hand, humerus, shoulder, and wrist. Each study was manually labeled as normal or abnormal by board-certified radiologist from the Stanford Hospital at the time of clinical radiographic interpretation in the diagnostic radiology environment between 2001 and 2012. The labeling was performed during interpretation on DICOM images presented on at least 3 megapixel PACS medical grade display with max luminance 400 cd/m<sup>2</sup> and min luminance 1 cd/m<sup>2</sup> with pixel size of 0.2 and naive resolution of 1500 x 2000 pixels. The clinical images vary in resolution and in aspect ratios. The dataset has been split into training (11,184 patients, 13,457 studies, 36,808 images), validation (783 patients, 1,199 studies, 3,197 images), and test (206 patients, 207 studies, 556 images) sets. There is no overlap in patients between any of the sets. To evaluate the models and get a robust estimate of radiologist performance, additional labels from the board-certified Stanford radiologists on the test set, consisting of 207 musculoskeletal studies. The radiologists individually retrospectively reviewed and labeled each study in the test set as a DICOM file as normal or abnormal in the clinical reading room environment using the PACS. The radiologists have 8.83 years of experience on average ranging from 2 to 25 years. The radiologists did not have access to any clinical information. Labels were entered into a standardized data entry program.[Rajpurkar, et al., 2018](https://arxiv.org/abs/1712.06957).

I downloaded the dataset from stanford machine learning group website  - https://stanfordmlgroup.github.io/competitions/mura. This data comes as split dataset (training & validation in separate directories). Each data categories such as training & validation has sub-directories (pls see below).

```
.
├── train {data category}
│   ├── XR_ELBOW {study type}
│   │   ├── patient00011 {patient}
│   │   │   └── study1_negative {study with label}
│   │   │       ├── image1.png {radiographs}
│   │   │       ├── image2.png
│   │   │       └── image3.png

.
├── valid {data category}
│   ├── XR_HUMERUS {study type}
│   │   ├── patient11216 {patient}
│   │   │   └── study1_negative {study with label}
│   │   │       ├── image1.png {radiographs}
│   │   │       └── image2.png
```

I have wrote few simple functions to extract the label information from the study directory and transform the data so that it can be processed for data visualizations, training & validation.

{% highlight python %}
def create_studies_metadata_csv(category):
    """
    This function creates a csv file containing the path of studies, count of images & label.
    """
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    study_types = ['XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS','XR_SHOULDER','XR_WRIST']
    i = 0
    study_data[category] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
    for study_type in study_types: # Iterate throught every study types
        DATA_DIR = '../data/MURA-v1.1/%s/%s/' % (category, study_type)
        patients = list(os.walk(DATA_DIR))[0][1]  # list of patient folder names
        for patient in tqdm(patients):  # for each patient folder
            for study in os.listdir(DATA_DIR + patient):  # for each study in that patient folder
                if(study != '.DS_Store'):
                    label = study_label[study.split('_')[1]]  # get label 0 or 1
                    path = DATA_DIR + patient + '/' + study + '/'  # path to this study
                    study_data[category].loc[i] = [path, len(os.listdir(path)), label]  # add new row
                    i += 1
    study_data[category].to_csv("../data/"+category+"_study_data.csv",index = None, header=False)
{% endhighlight %}

{% highlight python %}
def create_images_metadata_csv(category):
    """
    This function creates a csv file containing the path of images, label.
    """
    image_data = {}
    study_label = {'positive': 1, 'negative': 0}
    study_types = ['XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS','XR_SHOULDER','XR_WRIST']
    i = 0
    image_data[category] = pd.DataFrame(columns=['Path', 'Label'])
    for study_type in study_types: # Iterate throught every study types
        DATA_DIR = '../data/MURA-v1.1/%s/%s/' % (category, study_type)
        patients = list(os.walk(DATA_DIR))[0][1]  # list of patient folder names
        for patient in tqdm(patients):  # for each patient folder
            for study in os.listdir(DATA_DIR + patient):  # for each study in that patient folder
                if(study != '.DS_Store'):
                    label = study_label[study.split('_')[1]]  # get label 0 or 1
                    path = DATA_DIR + patient + '/' + study + '/'  # path to this study
                    for j in range(len(os.listdir(path))):
                        image_path = path + 'image%s.png' % (j + 1)
                        image_data[category].loc[i] = [image_path, label]  # add new row
                        i += 1
    image_data[category].to_csv("../data/"+category+"_image_data.csv",index = None, header=False)
{% endhighlight %}

## Abnormality Analysis
To investigate the types of abnormalities present in the dataset, the radiologist reports have been reviewed to manually label 100 abnormal studies with the abnormality finding: 53 studies were labeled with fractures, 48 with hardware, 35 with degenerative join diseases, and 29 with other miscellaneous abnormalities, including lesions and subluxations.[Rajpurkar, et al., 2018](https://arxiv.org/abs/1712.06957).

## Exploratory Data analysis
I have created a 2 types of metadata. One containing information for each image with its corresponding label & study type. Another one information containing for each study type, label & image count. So that i can plot the data to understand various characteristics. Also, i have plotted some images from both the classes (Normal & Abnormal) to visualize it.

{% highlight python %}
fig, ax = plt.subplots(figsize=(10, 5))
for i, phase in enumerate(data_cat):
    counts = patients_count[phase].values()
    m = max(counts)
    for i, v in enumerate(counts):
        if v==m: ax.text(i-0.1, v+3, str(v))
        else: ax.text(i-0.1, v + 20, str(v))
    x_pos = np.arange(len(study_types))
    plt.bar(x_pos, counts, alpha=0.5)
    plt.xticks(x_pos, study_types)

plt.xlabel('Study types')
plt.ylabel('Number of patients')
plt.legend(['train', 'valid'])
plt.show()
fig.savefig('../images/pcpst.jpg', bbox_inches='tight', pad_inches=0)
{% endhighlight  %}

![image](/public/images/pcpst.jpg)
Figure 1 [Rishabh Agrahari, 2018](https://medium.com/@pyaf/implementing-densenet-on-mura-using-pytorch-f39e92566815)

Figure 1 shows data from each study types. XR_WRIST has largest number of data compared to other study types. But one important thing is that, there is no significant variation in the validation data volumes among on the study types. It may cause some issue when predicting the classes for each study types. Because some study types don't have enough training data compared to others. For example _"XR-HUMERUS"_

![image](/public/images/pcpsc.jpg)
Figure 2 [Rishabh Agrahari, 2018](https://medium.com/@pyaf/implementing-densenet-on-mura-using-pytorch-f39e92566815)

Figure 2 shows the distribution of studies counts in each study (patient level). Sometimes a diagnosis requires multiple studies of different radiographs. It shows how many patients have to under-go multiple studies. It seems like more than 90% of the time, a patient require just one study. However, sometimes they under-go multiple studies.

![image](/public/images/pcplc.jpg)
Figure 3 [Rishabh Agrahari, 2018](https://medium.com/@pyaf/implementing-densenet-on-mura-using-pytorch-f39e92566815)

Figure 3 represents class distribution in the data for each study types. Ofcourse, it is clear that negative (0) class is dominating the dataset, compared to positive (1) class. However, I don't find the imbalance is of any concern.

![image](/public/images/pcpvc.jpg)
Figure 4 [Rishabh Agrahari, 2018](https://medium.com/@pyaf/implementing-densenet-on-mura-using-pytorch-f39e92566815)

Figure 4 visualizes the distribution of number of images in studies. As we discussed earlier, some patients may required to under-go multiple studies. It is also possible that each study may involve multiple radiographic images. The patient may required to take multiple radiographs during each study depending on the complexity of the problem. From this chart, we can infer, XR_FOREARM, XR_SHOULDER, XR_HAND, XR_WRIST & XR_FINGER study types have 3 images per study on majority of its studies. XR_HUMERUS, XR_ELBOW study type has 2 images per study on majority of its studies.


![image](/public/images/normal.png)
Figure 5 (Normal radiographs)

![image](/public/images/abnormal.png)
Figure 6 (Abnormal radiographs)
Figure 5 & 6 shows some sample radiographs.

{% highlight python %}
#show sample images
fig = plt.figure(1,figsize=(15,15))
positive = train_mat[train_mat[:,1]==1,:]
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    ix = np.random.randint(0, len(positive)) # randomly select a index
    img_path = positive[ix][0]
    print(img_path)
    ax.imshow(io.imread(img_path), cmap='inferno')
    cat = img_path.split('/')[2] # get the radiograph category
    plt.title('Category: %s & Label: %d ' %(cat, positive[ix][1]))
plt.show()
{% endhighlight %}

## Pre-preprocessing
As per the paper, i have normalized the each image to have same mean & std of the images in the ImageNet training set. In the paper, they have used variable-sized images to 320 x 320. But i have chosen to scale 224 x 224.  Then i have augmented the data during the training by applying random lateral inversions and rotations of up to 30 degrees using `ImageDataGenerator`

{% highlight python %}
img = cv2.imread(data['Path']) #read the image
img = cv2.resize(img,(img_width,img_height))  #scale to 224 x 224  
img = img[...,::-1].astype(np.float32) #convert the array into float32 dtype
{% endhighlight %}

{% highlight python %}
mean = np.mean(/public/images[:, :, :]) #normalization
std = np.std(/public/images[:, :, :])
images[:, :, :] = (/public/images[:, :, :] - mean) / std
{% endhighlight %}

{% highlight python %}
train_datagen = ImageDataGenerator( #rotations & lateral inversions
    rotation_range=30,
    horizontal_flip=True
    )

test_datagen = ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True
    )
{% endhighlight %}

## Model & Network Architecture
DenseNet or Densely Connected Convolutional Neural Network is another flavor of neural network. The DenseNet have several compelling advantages: they alleviate the vanishing-gradient
problem, strengthen feature propagation, encourage feature
reuse, and substantially reduce the number of parameters. https://github.com/liuzhuang13/DenseNet
According to the [paper](https://arxiv.org/pdf/1608.06993.pdf),
> The traditional convolutional networks with L layers have L
connections. whereas DenseNet, the l<sup>th</sup> layer has l inputs, consisting of the feature-maps of all preceding convolutional blocks. Its own feature-maps are passed on to all L-l subsequent layers. This introduces
L(L+1)/2 connections in an L-layer network, instead of just
L, as in traditional architectures. Because of its dense connectivity
pattern, we refer to our approach as Dense Convolutional
Network (DenseNet).

![image](/public/images/dense_net.JPG)
Figure 7 (A 5-layer dense block with a growth rate of k = 4.
Each layer takes all preceding feature-maps as input. [Huan. G et.al](https://arxiv.org/pdf/1608.06993.pdf) )

The model leverages consecutive functions (BatchNorm->Relu->Conv) from Wide Res Net paper. I used Keras implementation of DenseNet from `keras.applications.densenet` module. As per the [MURA](https://arxiv.org/abs/1712.06957) paper, I bootstrapped the `DenseNet169` model with the pre-trained weights from ImageNet. The overall probability of a study is computed by taking a arithmetic mean of the abnormalities probabilities output by the network for each image.

As per the [MURA](https://arxiv.org/abs/1712.06957) paper, i replaced the fully connected layer with the one that has a single output, after that i applied a sigmoid nonlinearity. In the paper, the optimized weighted binary cross entropy loss. Please see below for the formula,

<center>L(X, y) = -W<sub>T,1</sub> * ylog p(Y = 1|X) -W<sub>T,0</sub> * (1 - y)log p(Y = 0|X);</center>

p(Y = 1|X) is the probability that the network assigns to the label
i, W<sub>T,1</sub> = |N<sub>T</sub>| / (|A<sub>T</sub>| + |N<sub>T</sub>|), and W<sub>T,0</sub> = |A<sub>T</sub>| / (|A<sub>T</sub>| + |N<sub>T</sub>|) where |A<sub>T</sub>|) and |N<sub>T</sub>|) are the
number of abnormal images and normal images of study type T in the training set, respectively.

But i choose to use the default binary cross entropy. The network is configured with Adam using default parameters, batch size of 8, initial learning rate = 0.0001 that is decayed by a factor of 10 each time the validation loss plateaus after an epoch.

{% highlight python %}
#Bootstrap the base model with pre-trained weights.
base_model = DenseNet169(input_shape=(None, None,3),
                         weights='imagenet',
                         include_top=False,
                         pooling='avg')
x = base_model.output
x = Dropout(0.1)(x)
predictions = Dense(n_classes,activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc', 'mse'])
{% endhighlight %}

{% highlight python %}
#Print the network architecture diagram
from keras.utils import plot_model
from keras.utils.vis_utils import *
import keras
import importlib
importlib.reload(keras.utils.vis_utils)
import pydot
plot_model(model, to_file='images/densenet_archi.png', show_shapes=True)
{% endhighlight %}

![image](/public/images/densenet_archi.png)

## Training
To be completed..

## Performance Fine-tuning
To be completed..

## Prediction
To be completed..

## Training Environment Setup
I used AWS EC2 _p3.2xlarge_ instance for my training. This specific instance comes with _Tesla V100_ GPU. I opted for _Deep Learning AMI (Ubuntu) Version 21.0 - ami-09a706a24845d072_ image to boot up the EC2 instance. This image already loaded with the CUDA, Tensorflow & Keras modules. I activated _tensorflow_p36_ virtual environment to spin up jupyter notebook with the _Python 3_ kernel. Then i followed this [link](https://aws.amazon.com/getting-started/tutorials/get-started-dlami/) to create a tunnel to access the notebook url from my laptop.
[AMS - DLAMI Get started!](https://aws.amazon.com/getting-started/tutorials/get-started-dlami/)


_**P.S.:** AWS EC2 Instance can be expensive, i normally use it to train the model only for few hours and then would immediately terminate it. I would save the models so that i can resume train them whenever needed, that way, i don't keep running these machines. There may cheaper options available, But I find AWS EC2 hassle free. It just works for me everytime!_

## Citations
