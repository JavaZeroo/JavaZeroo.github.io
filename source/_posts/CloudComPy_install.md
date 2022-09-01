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

{% note danger %}
🚫 建议切换SHELL为bash
🚫 我在zsh上安装失败，切换bash成功
{% endnote %}

先下载 *CloudComPy_Conda39_Linux64_-date-.tgz* [here](https://www.simulation.openfields.fr/index.php/download-binaries)

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


### 如何使用:

不能用``conda activate <env>``需要使用 ``CloudComPy39/bin/condaCloud.sh`` 替代

将``<path install>``替换成你的CloudComPy39的目录
```
. <path install>/bin/condaCloud.sh activate CloudComPy39
```

**Remark**: 可能需要安装 libomp.so.5:
```
sudo apt-get install libomp5

```

测试所有项目（大约需要2分钟）:
```
. <path install>/bin/condaCloud.sh activate CloudComPy39
cd  <path install>/doc/PythonAPI_test
ctest
```

测试会输出到这里: ``${HOME}/CloudComPy/Data``

### 在vscode上编辑

如果直接打开在vscode的话就无法``import cloudComPy``

**正确的做法是**
```
. <path install>/bin/condaCloud.sh activate CloudComPy39
code -g <folder to edit>
```

### 官方文档

[https://www.simulation.openfields.fr/documentation/CloudComPy/html/index.html](https://www.simulation.openfields.fr/documentation/CloudComPy/html/index.html)

# 一些小细节

## 读取CloudComPy的点云数据

和open3d不太一样，读取点云数据有两种选择

1. 需要去修改点云的位置时用``toNpArray() ``

2. 复制出点云的点时用``toNpArrayCopy()``

## 给CloudComPy点云上色

当前版本(2.12)好像存在一些问题，建议转成open3d的数据类型后再添加颜色🤨