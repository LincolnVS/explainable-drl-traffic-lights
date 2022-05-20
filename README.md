# Traffic Light Control Baselines

This repo provides OpenAI Gym compatible environments for traffic light control scenario and a bunch of baseline methods. 

Environments include single intersetion (single-agent) and multi intersections (multi-agents) with different road network and traffic flow settings.

Baselines include traditional TLC algorithms and reinforcement learning based methods.

## Publications

1. Schreiber, L. V., Ramos, G. de O. & Bazzan, A. L. C. (2021). Towards Explainable Deep Reinforcement Learning for Traffic Signal Control [Oral Presentation]. International Conference on Machine Learning Conference: LatinX in AI (LXAI) Research Workshop 2021, Virtual. [LINK](https://research.latinxinai.org/papers/icml/2021/pdf/paper_26.pdf) 
2 - Alegre, L. N., Ziemke, T. & Bazzan, A. L. C. (2021). Using reinforcement learning to control traffic signals in a real-world scenario: an approach based on linear function approximation. IEEE Transactions on Intelligent Transportation Systems. [LINK](https://ieeexplore.ieee.org/document/9468362)


## Install

### Requirements

Before you begin, ensure you have met the following requirements:
- numpy v1.20.2
- keras v2.3.1
- python v3.7.10
- tensorflow v1.14.0
- pandas v1.2.3
- scipy v1.6.2
- seaborn v0.11.1
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
## How to cite this research

For citing this work, please use the following entries.

```bibtex
@InProceedings{Schreiber+2021lxai,
	author = {Lincoln Schreiber and Gabriel {\relax de} O. Ramos and Ana L. C. Bazzan},
	title = {Towards Explainable Deep Reinforcement Learning for Traffic Signal Control},
	booktitle = {Proc. of LatinX in AI Workshop @ ICML 2021},
	year = {2021},
	month = {July},
	publisher = {LatinX in AI},
	url = {https://research.latinxinai.org/papers/icml/2021/pdf/paper_26.pdf}
}
```

```bibtex
@Article{Alegre+2021its, 
	author = {Lucas N. Alegre and Theresa Ziemke and Ana L. C. Bazzan},
	title = {Using reinforcement learning to control traffic signals in a real-world scenario: an approach based on linear function approximation},
	journal = {IEEE Transactions on Intelligent Transportation Systems},
	pages = {},
	doi = {10.1109/TITS.2021.3091014},
	year = {2021}
}
```

# License
This project uses the following license: [MIT](https://github.com/LincolnVS/tlc-baselines/blob/master/LICENSE).
