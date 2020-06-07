# PENet 
This repository contains the model (PENet) described in the paper *"PENet: A Scalable Deep-learning model for Automated Diagnosis of Pulmonary Embolism Using Volumetric CT Scan"* published on Nature Digital Medicine. [manuscript link](https://rdcu.be/b3Lll) 

![](./img/grad_cam.gif)

## Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Results](#results)
0. [Usage](#usage)

## Introduction

Pulmonary Embolism (PE) is responsible for 180,000 deaths per year in the US alone. The gold standard diagnostic modality for PE is computed tomography pulmonary angiography (CTPA) which is interpreted by radiologists. Studies have shown that prompt diagnosis and treatment can greatly reduce morbidity and mortality, yet PE remains among the diagnoses most frequently missed or delayed due to lack of radiologist availability and physician fatigue. Therefore, there is an urgency in automating accurate interpretation and timely reporting of CTPA examinations to the immediate attention of physicians. 

To address this issue, we have developed an end-to-end deep learning model, PENet, capable of detecting PE. Some notable implementation details of PENet include: 
- Pretraining the model with a video dataset (Kinetics-600) for transfer learning
- Using a sliding window of CT slices as inputs to increase the proportion of the target PE relative to the input. 

Our model also highlights regions in the original CT scans that contributed most to the model’s prediction using Class Activation Maps, which can potentially help draw Radiologists’ attention to the most relevant parts of the CT scans for more efficient and accurate diagnosis [See Example](https://www.youtube.com/watch?v=ZdOabYt4Cjo). 

For more information please see the full manuscript at this [link](https://rdcu.be/b3Lll).

## Citation

If you use these PENet in your research, please cite:

	@article{huang2020penet,
            title={PENet—a scalable deep-learning model for automated diagnosis of pulmonary embolism using volumetric CT imaging},
            author={Huang, Shih-Cheng and Kothari, Tanay and Banerjee, Imon and Chute, Chris and Ball, Robyn L and Borus, Norah and Huang, Andrew and Patel, Bhavik N and Rajpurkar, Pranav and Irvin, Jeremy and others},
            journal={npj Digital Medicine},
            volume={3},
            number={1},
            pages={1--9},
            year={2020},
            publisher={Nature Publishing Group}
        }

## Results
|                                    | Internal dataset: Stanford | External dataset: Intermountain |
|------------------------------------|----------------------------|---------------------------------|
| Metric (AUROC) [95% CI]            |                            |                                 |
| PENet kinetics pretrained          |      0.84 [0.82–0.87]      |         0.85 [0.81–0.88]        |
| PENet no pretraining               |      0.69 [0.74–0.65]      |         0.62 [0.57–0.88]        |
| ResNet3D-50 kinetics pretrained    |      0.78 [0.74–0.81]      |         0.77 [0.74–0.80]        |
| ResNeXt3D-101 kinetics pretrained  |      0.80 [0.77–0.82]      |         0.83 [0.81–0.85]        |
| DenseNet3D-121 kinetics pretrained |      0.69 [0.64–0.73]      |         0.67 [0.63–0.71]        |

Our results demonstrate robust and interpretable diagnosis including sustained cross-institutional AUROC performance on an external dataset. PENet also outperforms the current state-of-the-art 3D CNN models by a wide margin. Thus, this work supports that successful application of deep learning to the diagnosis of a difficult radiologic finding such as PE on volumetric imaging in CTPA is possible, and can generalize on data from an external institution despite that the external institution. Ultimately, clinical integration may aid in prioritizing positive studies by sorting CTPA studies for timely diagnosis of this important disease including in settings where radiological expertise is limited.

## Usage

#### Environment Setup 
1. Please install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) in order to create a Python environment.
2. Clone this repo (from the command-line: `git clone git@github.com:marshuang80/PENet.git`).
3. Create the environment: `conda env create -f environment.yml`.
4. Activate the environment: `source activate ctpe`.

#### Downlaod trained weights

The checkpoints and weights for PENet are stored [here](https://stanfordmedicine.box.com/s/uql0ikebseltkkntiwl5rrn6zzuww6jt). 

#### Training

To re-train the model, please modify **dir_dir**, **ckpt_path** and **save_dir** in `train.sh` and run `sh train.sh`

#### Testing

To test the model, please modify **dir_dir**, **ckpt_path** and **results_dir** in `test.sh` and run `sh test.sh`

#### Generate CAMs

The script to generate CAMs using trained model is `get_cams.sh`. Please modify **dir_dir**, **ckpt_path** and **cam_dir**

#### Testing on raw DICOM

To predict probability of PE for a single study using raw DICOM files, please use `test_from_dicom.sh`. Remember to modify the relavent arguments.  
