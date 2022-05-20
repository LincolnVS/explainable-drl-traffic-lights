# Traffic Light Control Baselines

This repo provides OpenAI Gym compatible environments for traffic light control scenario and a bunch of baseline methods. 

Environments include single intersetion (single-agent) and multi intersections (multi-agents) with different road network and traffic flow settings.

Baselines include traditional TLC algorithms and reinforcement learning based methods.

## Publications

1. Schreiber, L. V., Ramos, G. de O. & Bazzan, A. L. C. (2021). Towards Explainable Deep Reinforcement Learning for Traffic Signal Control [Oral Presentation]. International Conference on Machine Learning Conference: LatinX in AI (LXAI) Research Workshop 2021, Virtual. [LINK](https://research.latinxinai.org/papers/icml/2021/pdf/paper_26.pdf) 


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

# License
This project uses the following license: [MIT](https://github.com/LincolnVS/tlc-baselines/blob/master/LICENSE).
