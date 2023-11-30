# spark进行mapreduce

## apache spark介绍

Apache Spark is a multi-language engine for executing data engineering, data science, and machine learning on single-node machines or clusters.

Apache spark最重要的是提供了数据科学的集群化方案. 它本身就提供了一个pyspark shell进行交互, 当然也可以编写python脚本进行交互. pyspark提供了全套API与apache spark进行交互, 因此使用python是可以的.

为了使用spark, 这里需要注意, 使用spark在分布式集群上使用需要结合hadoop, k8s等集群管理系统使用. 单机使用spark也需要使用Hadoop的HDFS存储系统, 下载的bin已经集成了基础的Hadoop接口, 因此可以直接使用.

## 安装

用pip安装pyspark. 需要注意的是安装

```bash
pip install pyspark
```

会安装一个几百MB的库, 不知道为什么这么大. 这个库是JRE, Apache Spark本体的, 因此需要自行额外下载.

额外下载JRE, Apache Spark后将bin添加到PATH即可. 需要注意的是, 这对于使用pyspark还不够, 需要将下列3个环境变量添加

```bash
JAVA_HOME=path/to/java
SPARK_HOME=path/to/spark
HADOOP_HOME=path/to/hadoop
```

之前提到过, spark本身集成了基础的Hadoop, 因此这里spark和hadoop的home可以是一样的. 当然自己再下一个Hadoop添加也行.

## 使用

### 基础概念

- RDD: resilient distributed dataset (RDD) 为弹性分布式数据集. 可以将数据集分布在一个cluster上, 然后可以并行化处理. RDD开始是HDFS中的一个文件, 后经过转化获得.

- 我的

### 创建RDD

接口的使用是很简单的, 主要是需要注意一些提升性能的操作, 以及理解spark的核心思想.
