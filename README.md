# Variational Autoencoder of MRI's of the Brain involving Alzheimer's Disease

---

**Author:** Jessica Sullivan

**Student Number:** 45799930

**Assignment:** COMP3710 Supplementary Assessment

---

## Description of Model

The model constructed was a Variation Autoencoder (VAE). A VAE is based off the structure of an autoencoder, with extra steps of normalisation inbetween. An autoencoder combines two types of Convolutional Nural Networks (CNN) togther; and encoder and decoder. CNN have muliple convolutional layers, usually with either a max pooling (decrease in the size of the data) or an upsampling (increasing in the size of the data) to connect layers. An encoder starts with the dataset and through multiple convolutional layers and max pooling reults with an output of much smaller size. A decoder is the opposite. Starting with the input, it will go through the convolutional layers and upsampling to create an output with a larger size. An autoencoder is an encoder which has its output fed through a decoder.

---

## Description of the DataSet

The ADNI dataset contains information of three different key groups (as referenced [here](https://adni.loni.usc.edu/)):

* completly normal healthy elders
* elders with mild memory problems or mild cognitive imparements
* elders with Alzheimer's disease dementia

The data that we will be analsying is of the magnetic resonanace images (MRI) of the brain ivolving these three categories of participants. 

---

## Description of the Files

**module.py:** Contains the code that creates the model for the autoencoder.

**dataset.py:** Imports and preprocesses the dataset for this project.

**train.py:** The main file used to run the project which will train the dataset and produce all outputs required.

---

## Outputs of Trained Model

---

## Plots

### Convergence Plots

### Loss function Equations

---

## Visualizations

---

## Answer to Report Question

**Question:**

```
Do your visulisations show any relationship or seperation with respect between the Alzheimert's siease and healthy groups?
```

**Answer:**

yes or no?

---
