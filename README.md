# An inductive graph neural network model for compound-protein interaction prediction based on a homogeneous graph.
---
### Installation with conda

If you don't have conda installed, please install it following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

```git clone https://github.com/wanxiaozhe/CPI-IGAE```

```cd CPI-IGAE```

```conda env create -f environment.yml```

### Datasets
All datasets are stored in `~/dataset`

### Usage

To check the results of test set, please run:
```
$ python test.py
```
To check the results of two external datasets ```DrugBank``` and ```TTD```, please run:
```
$ python outtest.py --outtest xxx
```
```xxx``` can be ```drugbank``` or ```ttd```

```
optional arguments:
  -h, --help            show this help message and exit
  --batch_size	        default is 512
  --device              which cuda device to use (-1 for cpu training)
                            default is 0, cpu is not recommended due to to cpu 
                            due to the long runing time
  --model               default is the trained model in `~/best_model`
  --outtest             only in the `outtest.py`, can be `drugbank` or `ttd`
```

Our trained model is `~/best_model/final_model.pth`
