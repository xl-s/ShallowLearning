# Deep Learning Small Project 
By: Chan Luo Qi (1002983), Seow Xu Liang (1003324)

## 1. Dataset 
### 1.1 Dataset Exploration 

#### 1.1.1 Data distribution 
- [x] distribution of images 
- [x] discussion of whether dataset is balanced / uniformly distributed etc - not sure if enough
- [x] graphs 
- [x] discuss typical data processing operations applied, why? 
- [ ] (Bonus) Data Augmentation techniques, why and proof of how it benefited the model 


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

1. **Overall Distribution**: Dataset is unbalanced with double the number infected-non-Covid samples (47.49%) than normal (25.47%) and infected-covid samples (27.04%). 
2. **Distribution of Normal vs Infected**: Dataset contains 72.96% infected samples (includes Covid and non-Covid samples) with the remaining 27.04% as normal samples. 
3. **Distribution of Infected** (Covid vs Non-Covid): 65.08% of infected samples are Covid while 34.92% are non-Covid. 

From the 3 different distributions as described above, we see that the dataset is unbalanced, with a higher proportion of samples being infected-Covid samples. 

### 1.1.2 Train-Test-Validation Split
![dataset_split](model_structures/datasetsplit.png)

### 1.2 Data Processing
1. **Normalisation.** Used to standardise input features to ensure better and faster convergence of the model. 

2. **Grayscale.** X-ray images are naturally quite monochrome to the human eye. However, the input images are saved "in colour" - that is, they contain 3 RGB channels. We decided to convert all inputs to grayscale, compressing input to a single channel. This can help make the model more generalisable (less detail) and also increase speed of computation. 

### 1.3 Data Augmentation
As discussed in Section 1.1, the dataset provided is not balanced. Data augmentation can help to generate new training samples  for the model to learn. In our model, we make use of ```Torchvision.Transforms.Compose``` to augment our training samples.  In every epoch, the transformations are randomly applied to the training dataset - that is, the model sees a set of slightly varied input each epoch.  

1. **Photometric distortions.** A quick visual scan of the dataset reveals that training samples vary in terms of brightness and saturation.  Thus, we apply photometric distortions randomly to samples in their hue, saturation and brightness. This could help to better generalise the model. 

2. **Horizontal Flips.** X-rays of the chest are quite symmetrical, with the exception of the presence of a denser mass on the right-side of the radiograph (indicating the heart). Flipping samples horizontally provide a quick method of generating more training data within reasonable expectations. 

3. **Rotations.** A small rotation of 10 degrees was randomly applied to training samples. Similar to horizontal flips, this is a quick method of generation more training samples within reasonable expectations. 


## 2. Classifier 
- [x] Discuss difference between the two architectures above and defend which one we chose and why 
- [x] Discuss reasons behind this choice of model structure (types of layers, # of params)
- [x] Discuss value for mini-batch size 
- [x] Explain choice of loss function and its parameters (if any)
- [x] (Bonus) Implementing regularisation on loss function and discuss its appropriate choice of parameters and benefits for model 

- [x] Explain choice of optimiser and its parameters
- [ ] (Bonus) Implementing scheduler and discuss its appropriate choice of parameters and benefits 
- [ ] Explain choice of initialisation of model parameters
- [x] Learning curves to show evolution of loss function and other performance metrics over epochs for both train and test sets

### 2.1 Choice of Architecture
#### 2.1.1 Difference in architecture
The 2-part binary architecture contains 2 sequential classifiers. The first classifier discriminates normal samples against infected samples whereas the second classifier only takes into account "infected" samples and discriminates against covid and non-covid samples within those found to be infected. 

In contrast, a single tri-class classifier completes the task in one step, distinguishing the 3 classes at the same time. 

#### 2.1.2 Hypothesis 
The team hypothesised that a 2-part binary architecture would be more suitable for the problem. We theorised that classification task of (1) Normal vs Infected and; (2) Covid vs Non-Covid is quite different. The model will probably have to consider different sets of structures for task (1) and (2). For example, the model might be concerned with finer details in the radiograph in task (2) while considering larger structures for task (1). 

To this end, we chose to work with a 2-part binary architecture.

#### 2.1.3 Architecture Evaluation
Experiments using both architectures were carried out and our hypothesis was confirmed. Results are shown in the table below. 

|Architecture|Train Set Accuracy|Test Set Accuracy|Validation Set Accuracy|
|:---:|:---:|:---:|:---:|
|2-part binary classifier|79.87%|79.19%|72.00%|
|tri-class classifier|81.58%|71.54%|60.00%|

Generally, the 2-part binary classifier gave better results. However, its improvement over the tri-class classifier was less than expected and could be attributed to the unbalanced dataset. 

### 2.2 Model Design
The team primarily used convolutional layers in our model design, which is the most appropriate for an image-classification task. 

![model desgin](model_structures/model_design.PNG)

We experimented with a variety of model architectures by varying the number of convolutional layers (together with the number of channels) and fully connected layers. We also experimented with the inclusion of batchnorm and dropout layers, as well as residual and inception networks. The basic convolutional model showed the most promising performance with batch normalization, but without dropout layers.

Generally, we found that all of the models overfitted extremely quickly, which may be primarily attributed to the small size of the dataset. Every model routinely attained 95+% training accuracy within a few epochs, but showed significantly poorer and highly varied performance on the test set. More complex models had accuracies ranging as far as from 60% to 85%, often in consecutive epochs (despite there being little change in training set performance). Training a model beyond two to three epochs did not yield any significant improvement in overall test set accuracy (sometimes resulting in reduced performance as compared to earlier epochs). This phenomenon was observed even with strong regularization applied. A typical learning curve demonstrating this behaviour is shown in the figure below.

![](plots/learning_curve.png)

For this reason, we settled on a relatively basic convolutional model as a baseline model, with three convolutional layers and one fully connected hidden layer. Due to the small dataset, more complex architectures would only add overhead to the model without offering significant improvement in performance.

#### 2.2.1 Mini-batch size 
Theoretically, a smaller batch size might compute faster. However, as the model contains batchnorm layers, it is not ideal to have small batch sizes as the normalisation will not be stable. Based on conventions, a 32 or 4 batch size is normally used. We recorded the time taken for each batch size to complete 1 epoch:

|Batch size|Time Taken (s)|
|:---:|:---:|
|32|10.22|
|64|11.03|

Thus, we chose a batch size of 32 in the end. 

#### 2.2.2 Loss function 
The **cross-entropy** loss function ```nn.CrossEntropyLoss```was used as this is a classification problem. As the dataset is biased, we used cross-entropy weights as calculated: $ w_0 = (n_0 + n_1)/(2*n_0) $, where $n_0 = $ # of samples with label 0, and $w_0 =$ weight for class 0. 

|infected|normal|
|:---:|:---:|
|0.25|0.75|

|covid|non-covid|
|:---:|:---:|
|0.65|0.35|

Regularisation was also done on the loss function by using the ```weight_decay``` parameter of the optimiser. The value of weight decay was decided using hyperparameter tuning, further described in Section XX below.  

### 2.3 Choice of Optimiser
Initially, we used the Adam optimiser as it is the "default" choice in deep learning models. However, we realised that Adam did not generalise over the data well and resulted in overfitting quite quickly.

Reading pyTorch's documentation, we found that ```torch.optim.AdamW``` could alleviate overfitting by implementing a weight decay parameter that penalises the magnitude of weights. 

[need to show graph to prove less overfitting????]

Hyperparameters Learning Rate (LR) and Weight Decay (WD) of the AdamW optimiser was tuned separately for each classifier. This was done by changing the value of the hyperparameter while holding all other factors constant. The model was trained for 8 epochs and the last 4 values of each performance metric was averaged to obtain the final performance of the model for that hyperparameter value. 

#### 2.3.1 Learning Rate (LR)
| |Normal VS Infected|Covid VS Non-Covid|
|:---:|:---:|:---:|
|Graph of Performance against Log(LR)|![tune_LR_1](tuning_norm/test_LR_more.png)|![tune_LR_2](tuning_inf/test_LR.png)|
|Optimal LR|2.3e-6|1e-5|

#### 2.3.2 Weight Decay (WD)
| |Normal VS Infected|Covid VS Non-Covid|
|:---:|:---:|:---:|
|Graph of Performance against Log(WD)|![tune_LR_1](tuning_norm/test_wtdecay.png)|![tune_LR_2](tuning_inf/test_wtdecay.png)|
|Optimal WD|5e-3|5e-3|

### 2.4 Final Model Architecture
To combat the overfitting of the model and improve robustness, we settled on utilizing an ensemble method for our final model architecture. 

We constructed two ensemble models, one to predict on covid/non-covid and another on infected/normal, which was then assembled into a two-part binary classifier.

We also constructed an additional standalone ternary classification ensemble model, to evaluate our hypothesis that a two-part binary classifier would outperform a single ternary classifier.

Each ensemble consists of 10 distinct baseline models with the architecture described in 2.2. At prediction time, the input is run through all 10 models, and the majority output is treated as the prediction.

#### 2.4.1 Model Training and Selection

Each individual baseline model was trained on 75% of the training dataset for 20 epochs, to reduce the overfitting of each model on the training set.

As mentioned in 2.2, each model showed a high variance in test set accuracy between epochs. Besides tracking the model performance, we used the test set to produce model evaluation metrics such that we could select the model which is most generalizable. To this end, we saved the model that performed the best out of all 20 epochs, instead of naively saving the final model after 20 epochs, with the rationale that a model which generalizes well to the test set would also generalize well to all other data, as well as to account for variations in the generalizability of the model throughout the training process.

As this is a medical diagnosis, we factored in sensitivity as a model selection and evaluation metric, under the assumption that it is less costly to produce a false positive prediction than a false negative prediction (seeing as a false negative could potentially result in a pneumonia or covid carrier spreading the disease to others). The final evaluation metric was thus designed to take into account both accuracy and sensitivity, with an additional balancing term to ensure  that neither metric performs too poorly.
$$
\text{performance} = \text{Acc.} + \text{Sens.} - |\text{Acc.} - \text{Sens.}|
$$
The final training algorithm for an individual baseline model is as follows:

* For 20 epochs:
  * Perform mini-batch gradient descent using the training set.
  * Evaluate the model using the aforementioned performance metric.
    * If it is better than the currently best-performing model, save the model and note it as the new best-performing model.


## 3. Results 
- [ ] Subplot on the validation set with ground truth, predicted labels + all performance metrics used 
- [ ] Discuss if we expected that COVID_NON-COVID was harder than INFECTED_NOT-INFECTED, why? 
- [ ] Would it be better to have high overall accuracy or low true negatives / false positive rates? Why?
- [ ] Does the model seem to replicate how doctors diagnose infections based on x-rays? 
- [ ] (Bonus) Show typical samples of failures and discuss what might be the reason? 
- [ ] Feature maps





