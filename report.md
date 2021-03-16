# Deep Learning Small Project 
By: Chan Luo Qi (1002983), Seow Xu Liang (xxxx)

## 1. Dataset 
### 1.1 Dataset Exploration 

#### 1.1.1 Data distribution 
-[x] distribution of images 
-[x] discussion of whether dataset is balanced / uniformly distributed etc - not sure if enough
-[x] graphs 
-[x] discuss typical data processing operations applied, why? 
-[ ] (Bonus) Data Augmentation techniques, why and proof of how it benefited the model 


| | Normal | Infected (Non-Covid) | Infected (Covid)|Total|
|:---:|:---:|:---:|:---:|:---:|
|Train|1341|2530|1345|5216|
|Test|234|242|138|614|
|Validation|8|8|8|24|
|Total|1583|2780|1491|5854|

|Overall distribution|Normal vs Infected|Infected Distribution|
|:---:|:---:|:---:|
|![dataset distribution](model_structures/dataset_distr.png)|![normal_distribution](model_structures/normal_distr.png)|![infected_distribution](model_structures/infected_distr.png)|
|Fig. 1A|Fig. 1B|Fig. 1C|

1. **Overall Distribution**: Dataset is unbalanced with double the number infected-non-Covid samples (47.49%)
than normal (25.47%) and infected-covid samples (27.04%). 
2. **Distribution of Normal vs Infected**: Dataset contains 72.96% infected samples (includes Covid and non-Covid samples) with the remaining 27.04% as normal samples. 
3. **Distribution of Infected** (Covid vs Non-Covid): 65.08% of infected samples are Covid while 34.92% are non-Covid. 

From the 3 different distributions as described above, we see that the dataset is unbalanced, 
with a higher proportion of samples being infected-Covid samples. 

### 1.1.2 Train-Test-Validation Split
![dataset_split](model_structures/datasetsplit.png)

### 1.2 Data Processing
1. **Normalisation.** Used to standardise input features to ensure better and faster convergence of the model. 

2. **Grayscale.** X-ray images are naturally quite monochrome to the human eye. However, the input images are saved "in colour" - 
that is, they contain 3 RGB channels. We decided to convert all inputs to grayscale, compressing input to a single channel. 
This can help make the model more generalisable (less detail) and also increase speed of computation. 

### 1.3 Data Augmentation
As discussed in Section 1.1, the dataset provided is not balanced. Data augmentation can help to generate new training samples 
for the model to learn. In our model, we make use of ```Torchvision.Transforms.Compose``` to augment our training samples. 
In every epoch, the transformations are randomly applied to the training dataset - that is, the model sees a set of slightly varied input 
each epoch.  

1. **Photometric distortions.** A quick visual scan of the dataset reveals that training samples vary in terms of brightness and saturation. 
Thus, we apply photometric distortions randomly to samples in their hue, saturation and brightness. This could help to better generalise the model. 

2. **Horizontal Flips.** X-rays of the chest are quite symmetrical, with the exception of the presence of a denser mass on 
the right-side of the radiograph (indicating the heart). Flipping samples horizontally provide a quick method of generating more 
training data within reasonable expectations. 

3. **Rotations.** A small rotation of 10 degrees was randomly applied to training samples. Similar to horizontal flips, 
this is a quick method of generation more training samples within reasonable expectations. 


## 2. Classifier 
-[x] Discuss difference between the two architectures above and defend which one we chose and why 
-[ ] Discuss reasons behind this choice of model structure (types of layers, # of params)
-[ ] Discuss value for mini-batch size 
-[ ] Explain choice of loss function and its parameters (if any)
-[ ] (Bonus) Implementing regularisation on loss function and discuss its appropriate choice of parameters and benefits for model 

### 2.1 Choice of Architecture

#### 2.1.1 Difference in architecture
The 2-part binary architecture contains 2 sequential classifiers. The first classifier discriminates normal samples 
against infected samples whereas the second classifier only takes into account "infected" samples and discriminates
against covid and non-covid samples within those found to be infected. 

In contrast, a single tri-class classifier completes the task in one step, distinguishing the 3 classes at the same time. 
 
#### 2.1.2 Hypothesis 
The team hypothesised that a 2-part binary architecture would be more suitable for the problem. We theorised that 
classification task of (1) Normal vs Infected and; (2) Covid vs Non-Covid is quite different. The model will probably have to
consider different sets of structures for task (1) and (2). For example, the model might be concerned with finer details 
in the radiograph in task (2) while considering larger structures for task (1). 

To this end, we chose to work with a 2-part binary architecture.

#### 2.1.3 Architecture Evaluation
Experiments using both architectures were carried out and our hypothesis was confirmed. Results are shown in the table below. 

|Architecture|Training Accuracy|Test Accuracy|
|:---:|:---:|:---:|
|2-part binary classifier|xx|xx|
|tri-class classifier|xx|xx|

Generally, the 2-part binary classifier gave better results. However, its improvement over the tri-class classifier was less
than expected and could be attributed to the unbalanced dataset. 

### 2.2 Model Design
The team primarily used convolutional layers in our model design, which is the most appropriate for an image-classification task. 

#### 2.3 Number of Layers 
#### 2.4 Number of features
#### 2.5 Mini-batch size 
#### 2.6 Loss function 
 

-[ ] Explain choice of optimiser and its parameters
-[ ] (Bonus) Implementing scheduler and discuss its appropriate choice of parameters and benefits 
-[ ] Explain choice of initialisation of model parameters
-[ ] Learning curves to show evolution of loss function and other performance metrics over epochs for both train and test sets

## 3. Results 
-[ ] Subplot on the validation set with ground truth, predicted labels + all performance metrics used 
-[ ] Discuss if we expected that COVID_NON-COVID was harder than INFECTED_NOT-INFECTED, why? 
-[ ] Would it be better to have high overall accuracy or low true negatives / false positive rates? Why?
-[ ] Does the model seem to replicate how doctors diagnose infections based on x-rays? 
-[ ] (Bonus) Show typical samples of failures and discuss what might be the reason? 
-[ ] Features 





