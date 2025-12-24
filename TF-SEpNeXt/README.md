# **TF-SEpNeXt:**

Paper: (A Study on Lightweight Acoustic Scene
Classification Using the ConvNeXt-V2 Architecture)

# Easy DCASE Task 1 - Train Your Model with a YAML file

Original Author: **Yiqiang Cai** (yiqiang.cai21@student.xjtlu.edu.cn), *Xi'an Jiaotong-Liverpool University*
<br>
**Modified by: minsiter02 (mswd81@chosun.ac.kr), *Chosun Univ***

## Task Description

The task 1 of DCASE Challenge focuses on Acoustic Scene Classification (ASC), recognizing the environment in which an audio recording was captured, such as streets, parks, or airports. For a detailed description of the challenge and this task, please visit the [DCASE website](https://dcase.community/challenge2024/). The main challenges of this task are summarized as below:

1. Domain Shift: Unseen devices exist in test set. (2020\~)
2. Short Duration: The duration of audio recordings reduced from 10s (\~2021) to 1s (2022\~).
3. Low Complexity: Limited model parameters (128K INT8) and computational overheads (30 MMACs). (2022\~)
4. Data Efficiency: Train model with fewer data, specifically 5%, 10%, 25%, 50% and 100%. (2024\~)

## System Description

This repository provides an easy way to train models on the DCASE Task 1 datasets, based on the framework by Yiqiang Cai. The original system (TF-SepNet + BEATs teacher) won the **Judges' Award** for DCASE2024 Challenge Task1. The corresponding paper is available [here](https://arxiv.org/abs/2408.14862).

**This version has been modified to develop and test the TF-SEpNeXt model, an enhanced architecture based on the original TF-SepNet. The key improvements and features of TF-SEpNeXt are documented in [Your Paper's Title or Section].**

1. All configurations of model, dataset and training can be done via a simple YAML file.
2. Entire system is implemented using [PyTorch Lightning](https://lightning.ai/).
3. Logging is implemented using [TensorBoard](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html#tensorboardlogger). ([Wandb API](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html) is also supported.)
4. Various task-related techniques have been included:
   * 3 Spectrogram Extractors: [Cnn3Mel](https://dcase-repo.github.io/dcase_util/generated/dcase_util.features.MelExtractor.html?highlight=mel#dcase_util.features.MelExtractor), [CpMel](https://github.com/fschmid56/cpjku_dcase23/tree/main), [BEATsMel](https://github.com/microsoft/unilm/tree/master/beats)
   * **4 High-performing Backbones:** [BEATs](https://arxiv.org/pdf/2212.09058), [TF-SepNet](https://ieeexplore.ieee.org/abstract/document/10447999), **TF-SEpNeXt**, [BC-ResNet](https://arxiv.org/abs/2106.04140).
   * 4 Plug-and-played Data Augmentation Techniques: [MixUp](https://arxiv.org/abs/1710.09412), [FreqMixStyle](https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Schmid_27.pdf), [SpecAugmentation](https://arxiv.org/abs/1904.08779), [Device Impulse Response Augmentation](https://arxiv.org/pdf/2305.07499).
   * 2 Model Compression Methods: [Post-training Quantization](https://lightning.ai/docs/pytorch/stable/advanced/post_training_quantization.html#model-quantization), [Knowledge Distillation](https://github.com/fschmid56/cpjku_dcase23/tree/main).

## Getting Started

1. Clone this repository.
2. Create and activate a [conda](https://docs.anaconda.com/free/miniconda/index.html) environment:

   ```bash
   conda create -n dcase_t1
   conda activate dcase_t1
   ```
3. Install [PyTorch](https://pytorch.org/get-started/previous-versions/) version that suits your system. For example:

   ```bash
   # For CUDA < 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   # For CUDA >= 12.1
   pip install torch torchvision torchaudio
   ```
4. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```
5. Download and extract the [TAU Urban Acoustic Scenes 2020 Mobile, Development dataset](https://zenodo.org/records/3819968), [TAU Urban Acoustic Scenes 2022 Mobile, Development dataset](https://zenodo.org/records/6337421) and [Microphone Impulse Response](https://micirp.blogspot.com/?m=1) according to your needs.  The directory should be placed in the **parent path** of code directory.

You should end up with a directory that contains, among other files, the following:

* ../TAU-urban-acoustic-scenes-2020-mobile-development/development/audio/: A directory containing 230,35 audio files in *wav* format.
* ../TAU-urban-acoustic-scenes-2022-mobile-development/development/audio/: A directory containing 230,350 audio files in *wav* format.
* ../microphone_impulse_response/: A directory containing 67 impulse response files in *wav* format.

6. The seed can be configured in the `main.py` file.

   ```python

   import time
   import random
   from pytorch_lightning import seed_everything

   dynamic_seed = int(time.time()*random.random()) 
   seed_everything(dynamic_seed, workers=True)
   ```
   
7. (Optional) You can load the train/validation data (25% split) or test data into shared memory before running the model training command.

   ```bash
   python shared_memory_process_split25.py
   python shared_memory_process_test.py
   ```
8. Start the training procedure by running the following command with the desired YAML configuration file:

   ```bash
   python main.py fit --config config/TFSEpNeXt_train.yaml # python main.py fit --config config/shm_TFSEpNeXt_train.yaml 
   ```

   You can override arguments directly from the command line:

   ```bash
   python main.py fit --config config/TFSEpNeXt_train.yaml --trainer.max_epochs 30
   python main.py fit --config config/TFSEpNeXt_train.yaml --optimizer.lr 0.006
   ```
9. Test your trained model:

   ```bash
   python main.py test --config config/TFSEpNeXt_test.yaml --ckpt_path path/to/your/model.ckpt
   ```
10. View results using TensorBoard:

    ```bash
    # Check training results
    tensorboard --logdir log/TFSEpNeXt_train
    # Check testing results
    tensorboard --logdir log/TFSEpNeXt_test
    ```
11. Quantize your model:

    ```bash
    python main.py validate --config config/TFSEpNeXt_quant.yaml --ckpt_path path/to/your/model.ckpt
    ```

## Knowledge Distillation

This framework supports knowledge distillation. To distill knowledge from fine-tuned BEATs to **TF-SEpNeXt**, first generate logits from a teacher model. Then, update the `logits_files` path in your configuration YAML (`config/tfsepnet_kd.yaml` or a custom one) and run the training command.

Example of distilling knowledge to **TF-SEpNeXt**:

```bash
python main.py fit --config config/TFSEpNeXt_kd.yaml
```

## Citation

**It is important to acknowledge both the original framework and the modifications made in this work.**

If you use the original framework or concepts from the DCASE 2024 workshop paper, please cite:

```bibtex
@inproceedings{Cai2024workshop,
    author = "Cai, Yiqiang and Li, Shengchen and Shao, Xi",
    title = "Leveraging Self-Supervised Audio Representations for Data-Efficient Acoustic Scene Classification",
    booktitle = "Proceedings of the Detection and Classification of Acoustic Scenes and Events 2024 Workshop (DCASE2024)",
    month = "October",
    year = "2024",
    pages = "21--25",
}
```

**If you use the TF-SEpNeXt model or refer to the specific modifications in this repository for your research, please cite our paper:**

```bibtex
@inproceedings{YourLastName2025,
    author = "{Your Name and Co-authors}",
    title = "{Your Paper Title}",
    booktitle = "{Conference or Journal Name}",
    year = "{2025}",
    pages = "{xx--xx}",
}
```
