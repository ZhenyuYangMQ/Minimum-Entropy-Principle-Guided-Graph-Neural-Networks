# Second-Order Pooling for Graph Neural Networks + MEDE
 
This is the code for MEDE with the graph-level representation method SOPOOL implementation. It is based on the code from [SOPOOL](This is the code arisen from paper "Second-Order Pooling for Graph Neural Networks". It is based on the code from [SOPOOL](https://github.com/divelab/sopool). Many thanks!

## Download & Citation

If you use the code or results, please kindly cite these papers.

```
@article{wang2020second,
  author={Wang, Zhengyang and Ji, Shuiwang},
  title={Second-Order Pooling for Graph Neural Networks},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  year={2020},
  publisher={IEEE}
}
```

## System requirement

#### Programming language
```
Python 3.6
```
#### Python Packages
```
PyTorch > 1.0.0, tqdm, networkx, numpy
```

## Run the code

We provide scripts to run the experiments. For bioinformatics datasets and the REDDIT datasets, run
```
chmod +x bio_opt.sh
./run_bio.sh [DATASET] [GPU_ID]
```

For the social network datasets, run
```
chmod +x social_opt.sh
./run_social.sh [DATASET] [GPU_ID]
```
