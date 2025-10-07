## <p align=center> TripleNet：Exploiting Complementary Features and Pseudo-Labels for Semi-Supervised Salient Object Detection  (TIP 2025) </p>
####  Liyuan Chen，Ming-Hsuan Yang, Jian Pu，Zhonglong Zheng </sup>


<font size=7><div align='center' > <a href=https://ieeexplore.ieee.org/abstract/document/11142954>**Paper**</a> | [**Training**](#training) | [**Testing**](#Testing) | [**Pre-trained Model**](#training)  </div></font>

![模型框架](framework.png)


## Abstract
Due to the limited output categories, semi-supervised salient object detection faces challenges in adapting conventional semi-supervised strategies. To address this limitation, we propose a multi-branch architecture that extracts complementary features from labeled data. Specifically, we introduce TripleNet, a three-branch network architecture designed for contour, content, and holistic saliency prediction. The supervision signals for the contour and content branches are derived by decomposing the limited ground truths. After training on the labeled data, the model produces pseudo-labels for unlabeled images, including contour, content, and salient objects.
By leveraging the complementarity between the contour and content branches, we construct coupled pseudo-saliency labels by integrating the pseudo-contour and pseudo-content labels, which differ from the model-inferred pseudo-saliency labels. We further develop an enhanced pseudo-labeling mechanism that generates enhanced pseudo-saliency labels by combining reliable regions from both pseudo-saliency labels. Moreover, we incorporate a partial binary cross-entropy loss function to guide the learning of the saliency branch to focus on effective regions within the enhanced pseudo-saliency labels, which are identified through our adaptive thresholding approach. Extensive experiments demonstrate that the proposed method achieves state-of-the-art performance using only 329 labeled training images.
