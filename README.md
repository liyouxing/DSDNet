### Single Traffic Image Deraining via Similarity-Diversity Model

[[<u>Paper</u>]](https://ieeexplore.ieee.org/document/10359459)
****

```
@article{li2023single,
  title={Single Traffic Image Deraining via Similarity-Diversity Model},
  author={Li, Youxing and Lan, Rushi and Huang, Huiwen and Zhou, Huiyu and Liu, Zhenbing and Pang, Cheng and Luo, Xiaonan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2023},
  publisher={IEEE}
}
 ```

*We discovers that there are similar degradations between the existing rain model and the haze one in the low-frequency
components but diverse degradations in the high-frequency areas, and develops a Similarity-Diversity physics model to
describe these characteristics. A novel deep similarity-diversity network, guided by the developed physics model, is
then proposed for single-traffic image deraining. Extensive single-traffic image deraining experiments have been
conducted to evaluate our proposed method which outperforms the other state-of-the-art deraining techniques. In
addition, the paper deploys the proposed algorithm with Google Vision API for object recognition, which also obtains
satisfactory results both qualitatively and quantitatively.*

### Requirements

****

- torch ~= 1.12.1 + cu113
- scikit-image ~= 0.19.3
- Pillow ~= 9.4.0
- torchvision ~= 0.13.1 + cu113
- matplotlib ~= 3.7.1
- tensorboardX ~= 2.2

### Dataset structure

1. download the datasets and arrange the images in the following order
2. Save the image names into text file

****

```
data
└── <dataset_name>
    ├── test    
    │   ├── norain  # clean images
    │   ├── rain    # rain images
    │   ├── test_norain.txt
    │   └── test_rain.txt
    ├── train   
    │   ├── norain  # clean images
    │   ├── rain    # rain images
    │   ├── train_norain.txt
    │   └── train_rain.txt
    └── val
        ├── norain  # clean images
        ├── rain    # rain images
        ├── val_norain.txt
        └── val_rain.txt

```

### Train and Test DSDNet

****

```
python main_train_test.py
```
