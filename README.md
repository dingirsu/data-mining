data.txt里合并了train和test的数据

models.py里包含三种聚类方法的实现以及最佳聚类参数的选择，生成的图片在fig/文件夹下面

run.py是在选择好最佳聚类参数后画图使用，运行时不需要输入参数的全名，如下输入即可

`python run.py -c 2 -e 0.5 -n 3 -nc 3`

缩写的含义在文件里已写明
