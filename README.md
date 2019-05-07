# IRM based Speech Enhancement using DNN

Practice project, inspired by [funcwj](https://github.com/funcwj)'s [nn-ideal-mask](https://github.com/funcwj/nn-ideal-mask). 

Compared to his project:
- Increased visualization of validation data
- More structured
- Adjust the order of Batch Normalization and ReLU function
- Support new PyTorch version(v1.1)

## ToDo

- [x] Preparation of dataset
- [x] DNN models
- [x] Training logic
- [x] Validation logic
- [x] Visualization of validation set (waveform, audio, config)
- [x] PESQ and STOI metrics
- [x] Optimize Config parameters
- [ ] Test script