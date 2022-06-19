# wyb_graduation
## 温度伸缩实验
代码位于`my_code`文件夹，默认`IDE`为`pycharm`，主要包含了温度伸缩实验的代码。
### 环境配置
在你创建了环境`python 3.6`之后，下面的命令可以帮助你安装需要的包：
* pip install numpy
* pip install pandas
* pip install matplotlib
* pip install openpyxl

或者你也可以通过直接配置`pycharm`以配置上述包。
### 实验
判断生成`token`或者`word`是否正确采用`levenshtein`算法，具体实现见`levenshtein.py`。其中在文件末尾给出了使用样例。
#### 未排序`token`级别温度实验
`T_scaling.py`为未排序`token`级别温度实验的代码。通过改变读取`json`文件处的代码来改变读取的数据。功能模块通过`##`分块，通过`ctrl+enter`运行单个模块。各模块功能依次为初始化，计算各温度对应`ECE`，将`ECE-T`数据写入`excel`文件，绘制`ECE-T`曲线，将`accuracy-confidence`数据写入`excel`文件，绘制`accuracy-confidence`曲线。
#### 已排序`token`级别温度实验
`T_scaling_sort.py`为经过排序`token`级别温度实验的代码。通过改变读取`json`文件处的代码来改变读取的数据。功能模块通过`##`分块，通过`ctrl+enter`运行单个模块。各模块功能与`T_scaling.py`中相同。
#### 未排序`word`级别温度实验
`T_scaling_word.py`为未排序`word`级别温度实验的代码。通过改变读取`json`文件处的代码来改变读取的数据。功能模块通过`##`分块，通过`ctrl+enter`运行单个模块。各模块功能与`T_scaling.py`中相同。
#### 已排序`word`级别温度实验
`T_scaling_sort_word.py`为经过排序`word`级别温度实验的代码。通过改变读取`json`文件处的代码来改变读取的数据。功能模块通过`##`分块，通过`ctrl+enter`运行单个模块。各模块功能与`T_scaling.py`中相同。
#### 已排序分`domain`,`slot`,`value`温度实验
`T_scaling_divide.py`为经过排序分`domain`,`slot`,`value`温度实验的代码。通过改变读取`json`文件处的代码来改变读取的数据。功能模块通过`##`分块，通过`ctrl+enter`运行单个模块。各模块功能依次为初始化，计算各温度对应`ECE`并将`ECE-T`数据写入`excel`文件，绘制`ECE-T`曲线，将`accuracy-confidence`数据写入`excel`文件，绘制`accuracy-confidence`曲线。
## 能量模型实验
代码位于`SGA-JRUD-master`文件夹中，在服务器`13`上，在其中的位置为`/mnt/workspace/liuhong/wangyb/SGA-JRUD-master`。
### 环境配置
在你创建了环境`python 3.6`之后，下面的命令可以帮助你安装需要的包：
* pip install torch==1.5
* pip install transformers==3.5
* pip install spacy==3.1
* python -m spacy download en_core_web_sm
* pip install nltk
* pip install sklearn
* pip install tensorboard
* pip install future

此外，为了评估，你需要安装 [standard evaluation repository](https://github.com/Tomiinek/MultiWOZ_Evaluation) , 其中我们改变了 `mwzeval/utils.py/load_references()` 中的`references`为 `damd`, 由于我们采用了与 [DAMD](https://github.com/thu-spmi/damd-multiwoz)中相同的去词汇化。

### 数据预处理
处理的数据在目录`data/`中，通过以下代码解压，
```
unzip data/data.zip -d data/
```
如果要从头开始预处理，你需要下载[MultiWOZ2.1](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip)到`data/`中然后对其进行解压，然后执行下述命令：

```
python data_analysis.py
python preprocess.py
```
此外，数据库文件也是需要的：
```
unzip db.zip
```

### 训练

### `nce`方法训练
`train_nce.py`文件中为使用`nce`方法训练的代码，通过修改`train_nce.sh`来改变其参数。运行方式为：
```
bash train_nce.sh $GPU ${your_exp_name}
```
e.g.
```
bash train_nce.sh 0 DS-baseline
```

### `dnce`方法训练
`train_dnce.py`文件中为使用`dnce`方法训练的代码，通过修改`train_dnce.sh`来改变其参数。运行方式为：
```
bash train_dnce.sh $GPU ${your_exp_name}
```
e.g.
```
bash train_dnce.sh 0 DS-baseline
```

### `residual`方法训练
`train_residual.py`文件中为使用残差能量模型方法训练的代码，通过修改`train_residual.sh`来改变其参数。运行方式为：
```
bash train_residual.sh $GPU ${your_exp_name}
```
e.g.
```
bash train_residual.sh 0 DS-baseline
```


