## Introduction
This is the source code of the paper entitled [*Meta knowledge assisted Evolutionary Neural Architecture Search*]

-------------------------------------------- **Current Version Notes**------------------------------------------

The current version is just a baseline approach for reference, the detailed explanation and useage will be given when this paper is accepted.
Of course, you can study this source code yourself to help your work.

To plot the architecture of found cells, you have to install *pygraphviz* package and use *Plot_network* function in **utils.py**. 


## Dependency Install
```
pytorch>=1.4.0
pygraphviz # used for plotting neural architectures
```

## Usage
```
# Search process
python EMO_v3.py

# Training process
python train_cifar.py # validate on CIFAR-10 and CIFAR-100 datasets
python train_imagenet.py # validate on ImageNet dataset

# Plotting
Given a solution in EMO_v3.py
the solution's normal cell and reduction cell can be plotted by
executing 'utils.Plot_network(solution.dag[0], path)' and 'utils.Plot_network(solution.dag[1], path)',
where 'path' is the path to save figures.

```




## Citation
If you find this work helpful in your research, please use the following BibTex entry to cite our paper.
```
@article{li2025meta,
  title={Meta knowledge assisted Evolutionary Neural Architecture Search},
  author={Li, Yangyang and Liu, Guanlong and Shang, Ronghua and Jiao, Licheng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  doi={10.1109/TCSVT.2025.3565562}
}
```

## Acknowledgement
Thanks for the help of [NAO](https://github.com/renqianluo/NAO_pytorch/tree/master/NAO_V2), [NSGA-Net](https://github.com/ianwhale/nsga-net) and [ACE-NAS](https://github.com/anonymone/ACE-NAS).

