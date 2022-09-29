torchunmix
==========

![](https://github.com/erikhagendorn/torchunmix-images/blob/main/augment-mixture-notitle.png?raw=true)
*Original (non-augmented) image is from [this](https://www.journalmc.org/index.php/JMC/article/view/2216) publication.*

## Title
TorchUnmix: automatic stain unmixing and augmentation for histopathology whole slide images in PyTorch

## Authors
<sup>1</sup>Erik Hagendorn  
AbbVie Bioresearch Center, Information Research, Worcester, MA, USA<sup>1</sup>

## Abstract
TorchUnmix is a library which aims to provide automatic stain unmixing and augmentation for histopathology whole slide images. Separation of histochemical stains (unmixing) is performed by orthonormal transformation of the RGB pixel data from pre-defined light absorption coefficients called stain vectors [1]. Precomputed publicly available stain vector definitions are often used, but inter-laboratory variation due to the histology and/or image acquisition process is common, yielding suboptimal unmixing results. Classical stain vector estimation methods rely on abundant distribution of stains, making them less practical for sparser distributions as observed from immunohistochemical stains. Geis et al. proposed a method based on k-means clustering of pixel values in the hue-saturation-density color space to determine optimal stain vectors which has been used in this work [2]. While stain vectors may be used for quantification of individual stains, TorchUnmix also provides functionalities to perform stain augmentation. Stain augmentation is a method used during the training process of deep learning models to improve generalization by unmixing the image, stochastically modifying the individual stains, and then compositing the stains into the final augmented image [3]. To our knowledge, no other libraries fully implement the above methods in PyTorch, utilizing GPU-acceleration. Additionally, TorchUnmix has extended all calculations used to perform the automatic stain unmixing and augmentation to operate on batches of images, drastically accelerating execution performance speeds in comparison to other libraries.

## References
1. Ruifrok, Arnout & Johnston, Dennis. (2001). Quantification of histochemical staining by color deconvolution. Anal Quant Cytol Histol. 23.
2. Geijs, D., Intezar, M., Litjens, G. J. S., & van der Laak, J. (2018). Automatic color unmixing of IHC stained whole slide images. In M. N. Gurcan & J. E. Tomaszewski (Eds.), Medical Imaging 2018: Digital Pathology (p. 20). SPIE. https://doi.org/10.1117/12.2293734. 
3. Balkenhol, M., Karssemeijer, N., Litjens, G. J. S., van der Laak, J., Ciompi, F., & Tellez, D. (2018). H&E stain augmentation improves generalization of convolutional networks for histopathological mitosis detection. In M. N. Gurcan & J. E. Tomaszewski (Eds.), Medical Imaging 2018: Digital Pathology (p. 34). SPIE. https://doi.org/10.1117/12.2293048

## Installation
1. Install [PyTorch](https://pytorch.org/get-started/locally/)
2. Install [kmeans-pytorch](https://github.com/subhadarship/kmeans_pytorch#installation)  
**_NOTE:_**  At the time of writing this, the kmeans-pytorch PyPi package is outdated from its master branch and may not work with TorchUnmix, we recommend you install this package from source
3. Install matplotlib (`pip install matplotlib`)
4. Clone this repository and install (`cd torchunmix && pip install .`)

## Quick Start
```python
from torchunmix.auto import Unmix
from torchunmix.augment import augment_stains_random

# iterate over mini-batches of images and cluster the pixel data
unmix = Unmix(dataloader, threshold=0.3, percentile=0.99, device='cuda:0', num_clusters=2)

# obtain the stain vectors after unmixing
rgb_to_stains, stains_to_rgb = unmix.stains()

# create random augmentation ranges for two stains (last tuple is an empty range since the stain doesn't exist)
rand_ranges = ((-0.1, 0.1), (-0.1, 0.1), (0.0, 0.0))

# use obtained stain vectors to augment a mini-batch of images
for batch in dataloader:
    batch = augment_stains_random(batch.to('cuda:0'), rgb_to_stains, stains_to_rgb, rand_ranges)
```
See [examples](https://github.com/abbvie-external/torchunmix/blob/master/examples) directory for guidance on more detailed usage.

## Performance
GPU acceleration has a large impact on processing speed, a critical feature for stain unmixing and augmentation of gigapixel whole slide images with TorchUnmix. The following performance benchmarks were obtained with a Nvidia Quadro P6000 GPU and an Intel Xenon E5-1620 v4 CPU.
### Stain Unmixing
![](https://github.com/erikhagendorn/torchunmix-images/blob/main/perfeval-unmix.png?raw=true)

### Stain Augmentation
![](https://github.com/erikhagendorn/torchunmix-images/blob/main/perfeval-augment.png?raw=true)

## Additional Notes
### Handling of non-deterministic results
Due to random initialization used by k-means clustering, the order of centroids (stains) returned is non-deterministic. The typical way to resolve this is by setting the `random_state` argument which can be passed to the `Unmix` class (and ultimately kmeans-pytorch), but this can lead to confusion if you forget to set it. To account for this, we sort the centroids based on their polar angle, with the origin at the mean of the centroids.

### Unmixing during training is not recommended
Geijs et al. describe a tile-based approach (although the actual tile size is never explicitly stated) for automatic stain unmixing using k-means clustering. In this work we extend this method by using mini-batches of images instead of single tiles. While our method does increase the amount of pixel data used for clustering, it is not recommended to perform an unmixing for each mini-batch during training as there is still no guarantee that all stains will be present within the mini-batch. We suggest computing the stain vectors by unmixing all tiles from the whole slide image(s) prior to using stain augmentation within the training loop.

### Number of supported stains is 2 or 3
As described in Ruifrok et al., there is a limitation to 2 or 3 stains when performing color deconvolution. In Geijs et al., they describe their methods  using only 2 stains by calculating the Euclidean distance between the cluster centers to find the largest distance. In this work, we extend this idea to three stains by determining the largest area of the triangle formed by the three cluster centers.

## Acknowledgements
Thank you to the Department of Pathology, Radboud University Medical Center, Nijmegen, The Netherlands for the publications enabling this work.

We'd also like to thank the authors and contributors to the following libraries which have provided insights for us to build upon:
- [scikit-image](https://github.com/scikit-image/scikit-image)
- [StainTools](https://github.com/Peter554/StainTools)
