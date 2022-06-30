## SJTU AU335 Computer Vision Course Project
# License Plate Detection and Recognition

## Files

- traditional_method/: 传统方法代码
    - images/: 要求识别的图片
        - easy/: 简单难度图片
        - medium/: 中等难度图片
        - difficult/: 困难难度图片
    - templates/: 用于模板匹配的模板
    - traditional_method.py: 基于传统方法的车牌检测识别算法代码
- deep_learning/: 深度学习代码
    - images/: 要求识别的图片
        - easy/: 简单难度图片
        - medium/: 中等难度图片
        - difficult/: 困难难度图片
    - resize.py: 用于处理训练集图片统一处理为512*512
    - get_train.py: 从labelme处理后的json文件中提取训练集和训练集标签
    - train.py: 训练
    - Unet.py: 包含训练和测试车牌定位的函数
    - CNN.py: 包含训练和测试车牌识别的函数
    - perspectiveTransform.py: 包含对车牌透视矫正的函数
    - main.py: 包含了可视化和测试调用的主函数
    - unet.h5, unet_green.h5, unet_green2.h5: 用于车牌定位的模型
    - cnn.h5: 用于车牌识别的模型
- CV_project_report.pdf: 报告

## Environment Requirements

**建议在Anaconda中配置不同环境**
- traditional_method:
    - opencv-python==4.5.5
    - numpy
- deep_learning:
    - python=3.6
    - opencv-python==4.1.0.25
    - tensorflow==1.15.2
    - h5py==2.10.0
    - keras==2.3.1 (keras相关会报warning, 但不影响运行)
    - numpy

## How to Run

- traditional method: 
    
    运行tradition_method.py，终端会提示"Please input the image name: "，输入一张测试图片的名字（如1-1.jpg），回车即可得到结果。

- deep learning:
    
    运行main.py，等待可视化框启动，待终端输出"already prepared"，可视化框跳出。在可视化框中点击选择文件，选择要测试的图片，再点击识别车牌即可得到结果。
