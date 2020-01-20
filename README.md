# Aerial Depth Completion

This work is described in the letter "Aerial Single-View Depth Completion with Image-Guided Uncertainty Estimation", by Lucas Teixeira, Martin R.
Oswald, Marc Pollefeys, Margarita Chli, published in the IEEE
Robotics and Automation Letters (RA-L) [IEEE link](https://doi.org/10.1109/LRA.2020.2967296).

#### Video:
<a href="https://www.youtube.com/embed/IzfFNlYCFHM" target="_blank"><img src="http://img.youtube.com/vi/IzfFNlYCFHM/0.jpg" 
alt="Mesh" width="240" height="180" border="10" /></a>

#### Citations:
If you use this Code or Aerial Dataset, please cite the following publication:

```
@article{Teixeira:etal:RAL2020,
    title   = {{Aerial Single-View Depth Completion with Image-Guided Uncertainty Estimation}},
    author  = {Lucas Teixeira and Martin R. Oswald and Marc Pollefeys and Margarita Chli},
    journal = {{IEEE} Robotics and Automation Letters ({RA-L})},
    doi     = {10.1109/LRA.2020.2967296},
    year    = {2020}
}
```
NYUv2, CAB and PVS datasets require further citation from their authors. 
During our research, we reformat and created ground-truth depth for the CAB and PVS datasets. 
This code also contains thirt-party networks used for comparison. 
Please also cite their authors properly in case of use. 


#### Acknowledgment:
The authors thanks [Fangchang Ma](https://github.com/fangchangma) and [Abdelrahman Eldesokey](https://github.com/abdo-eldesokey) for sharing their code that is partially used here. The authors also thanks the owner of the 3D models used to build the dataset. They are identified in each 3D model file.

-----------------------------------------------------------------------

### Prerequisites

#### Packages
* PyTorch 1.0.1
* Python 3.6
* Plus dependencies

#### Trained Models

Several trained models are available - [here](https://datasets.v4rl.ethz.ch/datasets/ral20-models/)

#### Datasets
* Aerial+NYUv2 Dataset - [link](https://datasets.v4rl.ethz.ch/datasets/ral20-SynDepthInspection/)
* CAB Dataset - [link](https://datasets.v4rl.ethz.ch/datasets/ral20-cab/)
* PVS Dataset - [link](https://datasets.v4rl.ethz.ch/datasets/ral20-pvs/)

#### Simulator
The Aerial Dataset was created using this simulator [link](https://github.com/VIS4ROB-lab/visensor_simulator)

### Running the code

#### Testing  Example

```bash
python3 main.py --evaluate "/media/lucas/lucas-ds2-1tb/tmp/model_best.pth.tar" --data-path "/media/lucas/lucas-ds2-1tb/dataset_big_v12"
```

#### Training Example

```bash
python3 main.py --data-path "/media/lucas/lucas-ds2-1tb/dataset_big_v12" --workers 8 -lr 0.00001 --batch-size 1 --dcnet-arch gudepthcompnet18 --training-mode dc1_only --criterion l2
```

```bash
python3 main.py --data-path "/media/lucas/lucas-ds2-1tb/dataset_big_v12" --workers 8 --criterion l2 --training-mode dc0-cf1-ln1 --dcnet-arch ged_depthcompnet --dcnet-pretrained /media/lucas/lucas-ds2-1tb/tmp/model_best.pth.tar:dc_weights --confnet-arch cbr3-c1 --confnet-pretrained /media/lucas/lucas-ds2-1tb/tmp/model_best.pth.tar:conf_weights --lossnet-arch ged_depthcompnet --lossnet-pretrained /media/lucas/lucas-ds2-1tb/tmp/model_best.pth.tar:lossdc_weights
```

#### Parameters

Parameter | Description
------------ | -------------
  --help            | show this help message and exit
  --output NAME       | output base name in the subfolder results
  --training-mode ARCH  | this variable indicating the training mode. Our framework has up to tree parts the dc (depth completion net), the cf (confidence estimation net) and the ln (loss net). The number 0 or 1 indicates whether the network should be updated during the back-propagation. All the networks can be pre-load using other parameters. training_mode: dc1_only ; dc1-ln0 ; dc1-ln1 ; dc0-cf1-ln0 ; dc1-cf1-ln0 ; dc0-cf1-ln1 ; dc1-cf1-ln1 (default: dc1_only)
  --dcnet-arch ARCH     | model architecture: resnet18 ; udepthcompnet18 ; gms_depthcompnet ; ged_depthcompnet ; gudepthcompnet18 (default: resnet18)
  --dcnet-pretrained PATH | path to pretraining checkpoint for the dc net (default: empty). Each checkpoint can have multiple network. So it is necessary to define each one. the format is **path:network_name**. network_name can be: dc_weights, conf_weights, lossdc_weights. 
  --dcnet-modality MODALITY | modality: rgb ; rgbd ; rgbdw (default: rgbd)
  --confnet-arch ARCH   | model architecture: cbr3-c1 ; cbr3-cbr1-c1 ; cbr3-cbr1-c1res ; join ; none (default: cbr3-c1)
  --confnet-pretrained PATH | path to pretraining checkpoint for the cf net (default: empty). Each checkpoint can have multiple network. So it is necessary to define each one. the format is **path:network_name**. network_name can be: dc_weights, conf_weights, lossdc_weights.
  --lossnet-arch ARCH   | model architecture: resnet18 ; udepthcompnet18 (uresnet18) ; gms_depthcompnet (nconv-ms) ; ged_depthcompnet (nconv-ed) ; gudepthcompnet18 (nconv-uresnet18) (default: ged_depthcompnet)
  --lossnet-pretrained PATH | path to pretraining checkpoint for the ln net (default: empty). Each checkpoint can have multiple network. So it is necessary to define each one. the format is **path:network_name**. network_name can be: dc_weights, conf_weights, lossdc_weights.
  --data-type DATA      | dataset: visim ; kitti (default: visim)
  --data-path PATH      | path to data folder - this folder has to have inside a **val** folder and a **train** folder if it is not in evaluation mode.
  --data-modality MODALITY | this field define the input modality in the format colour-depth-weight. kfd and fd mean random sampling in the ground-truth. kgt means keypoints from slam with depth from ground-truth. kor means keypoints from SLAM with depth from the landmark. The weight can be binary (bin) or from the uncertanty from slam (kw). The parameter can be one of the following: rgb-fd-bin ; rgb-kfd-bin ; rgb-kgt-bin ; rgb-kor-bin ; rgb-kor-kw (default: rgb-fd-bin)
  --workers N     | number of data loading workers (default: 10)
  --epochs N            | number of total epochs to run (default: 15)
  --max-gt-depth D      | cut-off depth of ground truth, negative values means infinity (default: inf [m])
  --min-depth D         | cut-off depth of sparsifier (default: 0 [m])
  --max-depth D         | cut-off depth of sparsifier, negative values means infinity (default: inf [m])
  --divider D           | Normalization factor - zero means per frame (default: 0 [m])
  --num-samples N | number of sparse depth samples (default: 500)
  --sparsifier SPARSIFIER | sparsifier: uar ; sim_stereo (default: uar)
  --criterion LOSS | loss function: l1 ; l2 ; il1 (inverted L1) ; absrel (default: l1)
  --optimizer OPTIMIZER | Optimizer: sgd ; adam (default: adam)
  --batch-size BATCH_SIZE | mini-batch size (default: 8)
  --learning-rate LR | initial learning rate (default 0.001)
  --learning-rate-step LRS | number of epochs between reduce the learning rate by 10 (default: 5)
  --learning-rate-multiplicator LRM | multiplicator (default 0.1)
  --momentum M          | momentum (default: 0)
  --weight-decay W | weight decay (default: 0)
  --val-images N        | number of images in the validation image (default: 10)
  --print-freq N  | print frequency (default: 10)
  --resume PATH         | path to latest checkpoint (default: empty)
  --evaluate PATH | evaluates the model on validation set, all the training parameters will be ignored, but the input parameters still matters (default: empty)
  --precision-recall | enables the calculation of precision recall table, might be necessary to ajust the bin and top values in the ConfidencePixelwiseThrAverageMeter class. The result table shows for each confidence threshold the error and the density (default:false)
  --confidence-threshold VALUE | confidence threshold , the best way to select this number is create the precision-recall table. (default: 0)

