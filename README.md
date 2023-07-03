# CXR-Siamese-Networks
An attempt to use Siamese networks to check whether there is a change or not in a before-and-after sequence of CXRs. 

The first attempt was to use the MIMIC-CXR dataset (default ResNet weights), then modified the architecture to use pre-trained weights and the ImaGenome dataset, then finally used MNIST dataset to see whether the architecture was wrong or not.

A Siamese network is two parallel and mirrored neural networks with the same weights and parameters. In this case, each network in the Siamese network is fed two images, A and B, which propagate through the identical networks separately and then eventually join into one output: the desired output (1 for "change" and 0 for "no change). For it to readjust its weights and parameters, it uses a loss function (cross-entropy loss). 

Here's a way to think of Siamese networks:
![Siamese Networks Diagram](https://github.com/tdserapio/CXR-Siamese-Networks/blob/main/Siamese%20Network%20Diagrams.png?raw=true)
**Cross-Entropy Loss** is used more often, but Binary-Entropy Loss isn't too bad in performance *if* you're planning to utilize two classes (eg. change versus no change). 
