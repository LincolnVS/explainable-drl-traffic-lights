# Traffic Light Control Baselines

This repo provides OpenAI Gym compatible environments for traffic light control scenario and a bunch of baseline methods. 

Environments include single intersetion (single-agent) and multi intersections (multi-agents) with different road network and traffic flow settings.

Baselines include traditional TLC algorithms and reinforcement learning based methods.


### Use

#### Requirements

**create requirements file**
```terminal
$conda list --export > requirements.txt
```

**use requirements file**
```terminal
$conda create --name <envname> --file requirements.txt
```

### Conda commands x Pip commands

| Task                                 | Conda package and environment manager command         | Pip and Virtualenv                                                    |
|--------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------|
| Install a package                    | `conda install $PACKAGE_NAME`                         | `pip install $PACKAGE_NAME`                                           |
| Update a package                     | `conda update --name $ENVIRONMENT_NAME $PACKAGE_NAME` | `pip install --upgrade $PACKAGE_NAME`                                 |
| Update package manager               | `conda update conda`                                  | Linux/macOS: `pip install -U pip` Win: `python -m pip install -U pip` |
| Uninstall a package                  | `conda remove --name $ENVIRONMENT_NAME $PACKAGE_NAME` | `pip uninstall $PACKAGE_NAME`                                         |
| Create an environment                | `conda create --name $ENVIRONMENT_NAME python`        | `cd $ENV_BASE_DIR; virtualenv $ENVIRONMENT_NAME`                      |
| Activate an environment              | `conda activate $ENVIRONMENT_NAME`                    | `source $ENV_BASE_DIR/$ENVIRONMENT_NAME/bin/activate`                 |
| Deactivate an environment            | `conda deactivate`                                    | `deactivate`                                                          |
| Search available packages            | `conda search $SEARCH_TERM`                           | `pip search $SEARCH_TERM`                                             |
| Install package from specific source | `conda install --channel $URL $PACKAGE_NAME`          | `pip install --index-url $URL $PACKAGE_NAME`                          |
| List installed packages              | `conda list --name $ENVIRONMENT_NAME`                 | `pip list`                                                            |
| Create requirements file             | `conda list --export`                                 | `pip freeze`                                                          |
| List all environments                | `conda info --envs`                                   | Install virtualenv wrapper, then `lsvirtualenv`                       |
| Install other package manager        | `conda install pip`                                   | `pip install conda`                                                   |
| Install Python                       | `conda install python=x.x`                            | X                                                                     |
| Update Python                        | `conda update python`                                 | X                                                                     |