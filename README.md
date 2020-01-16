#Aerial Depth Completion

This work is described in the letter "Aerial Single-View Depth Completion with Image-Guided Uncertainty Estimation", by Lucas Teixeira, Martin R.
Oswald, Marc Pollefeys, Margarita Chli, published in the IEEE
Robotics and Automation Letters (RA-L) (to appear).

#### Video:
<a href="https://www.youtube.com/embed/IzfFNlYCFHM" target="_blank"><img src="http://img.youtube.com/vi/IzfFNlYCFHM/0.jpg" 
alt="Mesh" width="240" height="180" border="10" /></a>

#### Citations:
If you use this Code or Dataset, please cite the following publication:

```
@article{Teixeira:etal:RAL2020,
    title   = {{Aerial Single-View Depth Completion with Image-Guided Uncertainty Estimation}},
    author  = {Lucas Teixeira and Martin R. Oswald and Marc Pollefeys and Margarita Chli},
    journal = {{IEEE} Robotics and Automation Letters ({RA-L})},
    doi     = {10.1109/LRA.2020.2967296},
    year    = {2020}
}
```

#### Acknowledgment:
The authors thanks @github/fangchangma and @github/abdo-eldesokey for sharing their code that is partially used here. The authors also thanks the owner of the 3D models used to build the dataset. They are identified in each 3D model file.

-----------------------------------------------------------------------

###Prerequisites

#### Packages
* PyTorch 1.0.1
* Python 3.6
* Plus dependencies

#### Trained Models

**PVS Dataset** - [Bagfile](https://drive.google.com/open?id=0B82ekrhU9sDmTTdIeFJXTlBBLVE)

#### Datasets

**Aerial+NYUv2 Dataset** - [Bagfile](https://drive.google.com/open?id=0B82ekrhU9sDmTTdIeFJXTlBBLVE)
**CAB Dataset** - [Bagfile](https://drive.google.com/open?id=0B82ekrhU9sDmTTdIeFJXTlBBLVE)
**PVS Dataset** - [Bagfile](https://drive.google.com/open?id=0B82ekrhU9sDmTTdIeFJXTlBBLVE)

### Running the code

####Testing

```bash
python3 main.py --evaluate [path_to_trained_model]
```

####Training

```bash
python3 main.py --evaluate [path_to_trained_model]
```

#### Parameters

