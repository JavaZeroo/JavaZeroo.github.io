---
title: CloudComPy库安装 
date: 2022-08-30 11:16:22
mathjax: false
categories:
- 🖥️Python
tags:
- cv
- 点云配准
---

下载链接：[https://www.simulation.openfields.fr/index.php/download-binaries](https://www.simulation.openfields.fr/index.php/download-binaries)

官方教程：[https://github.com/CloudCompare/CloudComPy/blob/master/doc/UseLinuxCondaBinary.md](https://github.com/CloudCompare/CloudComPy/blob/master/doc/UseLinuxCondaBinary.md)

## Installing, testing and using a CloudComPy binary on Linux, with conda

先下载 *CloudComPy_Conda39_Linux64_-date-.tgz* [here](https://www.simulation.openfields.fr/index.php/download-binaries)


As CloudComPy is under development, these instructions and the link are subject to change from time to time...

**This binary works only on Linux 64, on recent distributions, and with a Conda environment as described below, not anywhere else!**

**Only tested un Ubuntu 20.04 (focal) and Debian 11 (bullseye), please report any problems on other distributions.**

GLIBC版本需要``2.29``以上。

检查GLIBC版本:

```
ldd --version
```

必须先更新conda
```
conda update -y -n base -c defaults conda
```

新建一个conda环境
```
conda create --name CloudComPy39 python=3.9
```

安装需要的包
```
conda activate CloudComPy39
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install "boost=1.72" "cgal=5.0" cmake ffmpeg "gdal=3.3" jupyterlab "matplotlib=3.5" "mysql=8.0" "numpy=1.22" "opencv=4.5.3" "openmp=8.0" "pcl=1.11" "pdal=2.3" "psutil=5.9" "qhull=2019.1" "qt=5.12" "scipy=1.8" sphinx_rtd_theme spyder tbb tbb-devel "xerces-c=3.2"
```

{% note info %}
如果遇到``Solving environment: failed with initial frozen solve. Retrying with flexible solve.``
可以尝试重新新建一个conda环境
{% endnote %}

where `<conda_dir>` is the installation directory of conda (often `~/anaconda3` or `~/miniconda3`)

Install the binary in the directory of your choice.

### Using CloudCompare and CloudComPy:

Before using CloudCompare or CloudComPy, you need to load the environment. 
The script `bin/condaCloud.sh` overrides the conda command for activation and deactivation, it must be sourced. 

From a new prompt (replace `<path install>` by its value): 

```
. <path install>/bin/condaCloud.sh activate CloudComPy39
```

if conda is unknown, execute the following instruction before:

```
. <conda_dir>/etc/profile.d/conda.sh
```
where `<conda_dir>` is the installation directory of conda (often `~/anaconda3` or `~/miniconda3`)

Have a look on the script usage:
```
. <path install>/bin/condaCloud.sh
```

To run CloudCompare:

```
CloudCompare
```

To execute a Python script using CloudComPy:

```
Python myscript.py
```
**Remark**: You may need to install libomp.so.5:
```
sudo apt-get install libomp5

```
The IDE [Spyder](https://www.spyder-ide.org/) and [Jupyter](https://jupyter.org/) can be launched in this environment:

```
spyder
```

```
jupyter notebook
```

An example of notebook is provided in ```doc/samples/histogramOnDistanceComputation.ipynb```.

### Execute all the Python tests:

```
. <path install>/bin/condaCloud.sh activate CloudComPy39
cd  <path install>/doc/PythonAPI_test
```

To execute all the tests (about two minutes):

```
ctest
```

The files created with the tests are in your user space: `${HOME}/CloudComPy/Data`

From the prompt, you can :

### In case of problem:

There may be differences in the versions of conda packages. When updating the conda configuration, the package versions may change slightly.This is usually not a problem, but since the CloudComPy binary is fixed, there may be a version difference on a package, which makes it incompatible with CloudComPy. For your information, here is the list of package versions when CloudComPy was built.

The result of ```conda list``` command is provided in the sources in [building directory](../building)
