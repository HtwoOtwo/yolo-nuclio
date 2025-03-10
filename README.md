
## 启动nuclio服务
	```
    bash deploy.sh function_yaml_path
	```

## 文件结构

- `function.yaml`: 声明模型以及docker依赖

- `main.py`: 包含可被CVAT执行的handle函数

- `yolo11n-seg.pt`: yolov11模型

## 参考

1. https://opencv.github.io/cvat/docs/manual/advanced/serverless-tutorial/#adding-your-own-dl-models
2. https://stephencowchau.medium.com/journey-using-cvat-semi-automatic-annotation-with-a-partially-trained-model-to-tag-additional-8057c76bcee2
