## 本代码用于***data_20230901_2***格式的多光谱茶叶、杂质数据的预标注
***
#### 代码的使用：
##### 1、安装相关的python包
##### 2、打开main.py文件，在文件底部修改path为自己想要预标注的图片所在路径，然 后运行main.py文件，即可在path目录下生成***00.json***文件
##### 3、打开labelme，打开path文件夹中的00.png图片，即可看到预标注数据
***
#### 补充说明：
##### 1、path应当是13通道的图片所在的文件夹、而不是具体某个图片的路径。譬如：path = r'D:\mycodes\RITH\puer\puer_baseline\data_20230901_2\data\test\3-test\combined_data'
![这是图片](file:///C:/Users/13635/Pictures/Screenshots/1.png)
##### 2、labelme是通过json文件中的键——"imagePath"来关联图片的。我在脚本中默认将键——"imagePath"的值设置为"00.png"。如果有需要的话可以在gray_to_labelme.py文件中修改相应的值
![这是图片](file:///C:/Users/13635/Pictures/Screenshots/2.png)
***
#### 标签修正注意事项
##### 1、预标注是不能把头发等比较细微的杂质标记出来的，因此修正标签时一定要加上这类杂质
##### 2、预标注经常会把环形茶叶的中间背景区域标注为茶叶，需要注意对这些问题的修正
##### 3、要注意对阴影区域做修正
![这是图片](file:///C:/Users/13635/Pictures/Screenshots/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-09-18%20140625.png)