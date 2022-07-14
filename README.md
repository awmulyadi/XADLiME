# Estimating Explainable Alzheimer's Disease Likelihood Map via Clinically-guided Prototype Learning
![XADLiME](image/xadlime.png)

This repository provides the PyTorch implementation of our proposed XProtoADPM framework in addressing Alzheimer's Disease progression modeling.

## Datasets
We utilized Alzheimer's disease neuroimaging initiative dataset
* http://www.loni.usc.edu/ADNI

## Usage
### ADPEN
For pretraining the ADPEN, run:
``` 
python xadlime_adpen.py --fold=1 --gpu_id=0 --finetune=0
```
For finetuning the ADPEN, make a list the pretrained directory location and run:
``` 
python xadlime_adpen.py --fold=1 --gpu_id=0 --finetune=1
```

### ProgAE
For training the autoencoder for progression map, run:
``` 
python xadlime_progae.py --fold=1 --gpu_id=0
```

### XADLiME
After training all required networks, XADLiME can be executed through:
``` 
python xadlime_classification_clinicalstage.py --fold=1 --gpu_id=0
```
``` 
python xadlime_regression_mmse.py --fold=1 --gpu_id=0
```
``` 
python xadlime_regression_age.py --fold=1 --gpu_id=0
```

## Acknowledgements
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) No. 2022-0-00959 ((Part 2) Few-Shot Learning of Causal Inference in Vision and Language for Decision Making) and No. 2019-0-00079 (Artificial Intelligence Graduate School Program (Korea University)).