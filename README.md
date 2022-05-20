# Traffic Light Control Baselines



## Install

### Requirements

Before you begin, ensure you have met the following requirements:
- numpy==1.20.2
- keras==v2.3.1
- python==3.7.10
- tensorflow==1.14.0
- pandas==1.2.3
- scipy==1.6.2
- seaborn==0.11.1
- [CityFlow](https://github.com/cityflow-project/CityFlow)

Newer versions of the above items may not be fully compatible with our code.

### Conda Env
To make reproducibility easier, using a conda environment it is possible to load all dependencies.

**To create an environment from an environment file:**
```terminal
$conda env create -f conda_environment.yaml
```

<!---
**create requirements file**
```terminal
$conda list --export > requirements.txt
```
-->

## Usage
Just run any of the `run_*.py` scripts and pass the path of config file.

Example:
```Bash
python run_dqn.py ./envs/jinan_3_4/config.json
```



## How to cite this research

For citing this work, please use the following entries:

```bibtex
@InProceedings{Schreiber+2022ijcnn,
	author = {Schreiber, Lincoln and Alegre, Lucas N. and Bazzan, Ana L. C. and Ramos, Gabriel {\relax de} O.},
	title = {On the Explainability and Expressiveness of Function Approximation Methods in RL-Based Traffic Signal Control},
	booktitle = {2022 International Joint Conference on Neural Networks (IJCNN)},
	OPTpages = {},
	year = {2022},
	address = {Padova, Italy},
	month = {July},
	publisher = {IEEE},
	OPTdoi = {},
	OPTurl = {https://doi.org/},
	note = {Forthcoming}
}
```
## Publications

1. L. Schreiber, L. N. Alegre, A. L. C. Bazzan, and G. O. Ramos, “On the Explainability and Expressiveness of Function Approximation Methods in RL-Based Traffic Signal Control,” in 2022 International Joint Conference on Neural Networks (IJCNN), Padova, Italy, 2022. [LINK IN PROGRESS]

2. Schreiber, L. V., Ramos, G. de O. & Bazzan, A. L. C. (2021). Towards Explainable Deep Reinforcement Learning for Traffic Signal Control [Oral Presentation]. International Conference on Machine Learning Conference: LatinX in AI (LXAI) Research Workshop 2021, Virtual. [LINK](https://research.latinxinai.org/papers/icml/2021/pdf/paper_26.pdf) 

3. Alegre, L. N., Ziemke, T. & Bazzan, A. L. C. (2021). Using reinforcement learning to control traffic signals in a real-world scenario: an approach based on linear function approximation. IEEE Transactions on Intelligent Transportation Systems. [LINK](https://ieeexplore.ieee.org/document/9468362)

# License
This project uses the following license: [MIT](https://github.com/LincolnVS/tlc-baselines/blob/master/LICENSE).

> Created from a repo that provides a OpenAI Gym compatible environments for traffic light control scenario - [tlc-baselines](https://github.com/zhc134/tlc-baselines)
