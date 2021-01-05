# Cifar10_Bayesian_Classifier.
In this we use Bayesian Statistical principles to classify images present in 10  different cases such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

## Data Preprocessing:
<p align="justify">
There are total of 50,000 training images of dimensions 32X32X3 divided into 10 different classes. Testing set consisits of seperate 10,000 images. The only important feature of Cifar-10 images is their average color. That means that we calculate the mean color of 32X32 images and each of the i=1,...,50000 CIFAR-10 images is then represented by only three values xi=(ri,gi,bi).
</p>


## Implementation:
<p align="justify">
Bayesian rule to be used is stated as:
![image](https://user-images.githubusercontent.com/42828760/103663242-8d963280-4f79-11eb-9789-7a8f16fd5389.png)
To use the above rule, we computed the mean and variance of the three color channels for each class. The function def cifar10_naivebayes_learn(Xf,Y) computes the normal distribution parameters (mu,sigma,p) for all ten classes(mu and sigma are 10X3 and prior p is 10X1). Finally, the function def cifar10_classifier_naivebayes(x,mu,sigma,p) returns the Bayesian optimal class c for the sample x
</p>
