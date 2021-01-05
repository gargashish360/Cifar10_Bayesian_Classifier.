# Cifar10_Bayesian_Classifier.
In this we use Bayesian Statistical principles to classify images present in 10  different cases such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

##Data Preprocessing:
There are total of 50,000 training images of dimensions 32X32X3 divided into 10 different classes. Testing set consisits of seperate 10,000 images. The only important feature of Cifar-10 images is their average color. That means that we calculate the mean color of 32X32 images and each of the i=1,...,50000 CIFAR-10 images is then represented by only three values xi=(ri,gi,bi).
