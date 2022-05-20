# Traffic Light Control Baselines

This repo provides OpenAI Gym compatible environments for traffic light control scenario and a bunch of baseline methods. 

Environments include single intersetion (single-agent) and multi intersections (multi-agents) with different road network and traffic flow settings.

Baselines include traditional TLC algorithms and reinforcement learning based methods.


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
