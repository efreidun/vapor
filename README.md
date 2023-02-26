# VaPoR
PyTorch source code for the _**Va**riational **Po**se **R**egression_ method proposed in the ICRA 2023 paper [A Probabilistic Framework for Visual Localization in Ambiguous Scenes](https://arxiv.org/abs/2301.02086).

## Dependencies
All python dependencies of the project are listed in the `requirement.txt` file, installed by `pip install -r requirements-dev.txt`. In addition to these, and optionally
1. [Deep Bingham Networks](https://github.com/Multimodal3DVision/torch_bingham) package is required for computing sample likelihoods, 
2. [Instant Neural Graphics Primitives](https://github.com/NVlabs/instant-ngp) package is required for modelling scenes and view synthesis from sampled poses, and
3. [Weights & Biases](https://wandb.ai/) for logging of results.

If you do not wish to install the latter packages, do not run/comment out the relevant sections that use those functionalities.


## Usage
Clone the repository and install it, for example, with
```bash
pip install .
```

The project codebase assumes the following file structure,
```
+── ~/data
│   +── AmbiguousReloc
│       +── blue_chairs
│       +── meeting_table
│       +── seminar
│       +── staircase
│       +── staircase_ext
│   +── CambridgeLandmarks
│       +── KingsCollege
│       +── OldHospital
│       +── ShopFacade
│       +── StMarysChurch
│       +── Street
│   +── SevenScenes
│       +── chess
│       +── fire
│       +── heads
│       +── office
│       +── pumpkin
│       +── redkitchen
│       +── stairs
│   +── Rig
│       +── Ceiling
+── ~/code
│   +── vapor (clone root)
│       +── vapor
│           +── scripts
│       +── runs
```
where each dataset follows its own structure for images and ground truth pose labels. Refer to `vapor/data.py` for the exact structures.

## Training and Evaluation

The project contains various tools for evaluation and analysis of results, including many tools that were not included in the paper. A minimal guide to train and evaluate the pipeline is provided below. However, feel free to explore the available tools.

#### Training
You can train the pipeline by the training script
```
python vapor/scripts/train_pipeline.py --dataset AmbiguousReloc --sequence blue_chairs
```
and provide the appropriate settings by flags. The results will be saved at `vapor/runs/RUNNAME`.

#### Evaluation
You can evaluate a pretrained model by the evaluation script
```
python vapor/scripts/evaluate_pipeline.py RUNNAME
```
You can then visualize the samples predicted by the pretrained model by the visualization script
```
python vapor/scripts/visualize_samples.py RUNNAME
```
The results will be saved at `vapor/runs/RUNNAME/plots/`.

## Citation
If you find this library useful in your research, consider citing our publication:
```
@article{zangeneh2023vapor,
  title={A Probabilistic Framework for Visual Localization in Ambiguous Scenes},
  author={Zangeneh, Fereidoon and Bruns, Leonard and Dekel, Amit and Pieropan, Alessandro and Jensfelt, Patric},
  journal={arXiv preprint arXiv:2301.02086},
  year={2023}
}
```
