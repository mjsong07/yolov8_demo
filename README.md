# 环境搭建
> 本人使用mac m1电脑安装

设置虚拟环境
```sh
brew install python@3.9 # 只能使用3.9的python
python3.9 -m venv yolo8_env # 使用3.9 创建
source yolo8_env/bin/activate # 激活
python -V  # 查看版本
```
安装依赖库
```sh
pip install torch torchvision torchaudio
```
**安装 YOLOv8**
```sh
pip install ultralytics 
yolo --version # 查看版本
```

# 验证
这里会自动拉取官方的模型yolov8s.pt，同时识别图片里面的人物和汽车是否识别
```sh
yolo detect predict model=yolov8s.pt source='https://ultralytics.com/images/bus.jpg'
```
看到正常使用模型运算
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/d8e25ec085b54649ad24c3e98cff54cb~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753262543&x-orig-sign=WN3KeC3q5KAClBhi0ErkjmYobEs%3D)

输出的文件
runs/detect/predict/bus.jpg

这里看到人和车都被识别出来了
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/ccd5c63fb10044aaaddfcd467dcba1a0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753264993&x-orig-sign=XpjQBuPh4JMn7jkmPHmDjMWbbBM%3D)

# 实现视频的微调
yolov8不仅能处理图片，视频也是杠杠的（其实视频也是一张张图片动态识别而已）

这是用自带`yolov8s.pt`的模型检查
```sh
yolo detect predict model=yolov8s.pt source='./mov_bbb.mp4'  show=True save=True
```

输出识别有问题，有时候把兔子识别成dog，有时候识别成sheep，我们来调教一下吧

![Jul-23-2025 09-35-13.gif](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/6b0380190bfe4c45941e3ff1a6ecb129~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753320972&x-orig-sign=sOgv8KWhVvLOgfQo2ze7LsemVGU%3D)
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c30aee2ee31a4db2906c8c963c55d906~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753267428&x-orig-sign=RLic4s%2FV5ovKEtWD%2FZ8v3hsgxdY%3D)


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4e5931e6155c45f3a4371b86f8fd5f21~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753267386&x-orig-sign=Q6v6w9HPN22DuxMwZ0TTRFiIJ%2Fs%3D)

微调目前主要是通过图片做坐标标记来实现的。

## 准备素材和整理文件夹

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/096c4e2b7b614ec998c17d837a9e4d83~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753265919&x-orig-sign=r%2Bn5GUn2eMXSOP20pLXJXVGcT%2Fk%3D)

在dataset下的images 
- train 代表验证的图片文件夹
- val 代表二次检验图片文件夹


在dataset下的labels
- train 代表验证的坐标x和y的信息
- val 代表二次检验的坐标x和y的信息

图片通过播放视频的时候分别截取5张，train 放3张，val放2张。


## 安装图像标记编辑工具 labelImg
安装
```js
pip install labelImg
```
直接运行
```js
labelImg
```

选择左边的工具栏 
- open Dir 选择上面的`./dataset/images/train`文件夹，作为输入路径

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c9349293fd284ab28060c7c75ddba972~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753267694&x-orig-sign=Ay2JZKUT8zwdyu9DTFG9V8duGOY%3D)



- change save Dir 选择上面的`./dataset/labels/train`文件夹，作为输出路径


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/45501c267df84527827df2c438353e99~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753266121&x-orig-sign=jy0lsXUxU3K2xigoaYpeiJG2L94%3D)

选择yolo模式

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/145c6cea34304c5096bed7cc931501db~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753266143&x-orig-sign=A%2FcjVGqWQz5cvGhPh7ms7pSFu5A%3D)


开始标记 ，点击左边的createRect box，

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/3740af97238e4bbbb295d5dc5901db0d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753267846&x-orig-sign=jAKwkufxdlg%2FZQjA88x5QCXlbe8%3D)

选中动物的范围，然后输入标记名称为 rabbit 
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/eb0053a0b1a941e7a5665b65d998646e~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753266169&x-orig-sign=gVpSZTitXX5%2Bv4VZYbDvbajor8c%3D)


每次标记完后，点击ok，再点击save，然后再点击 next image，快速编辑下一张

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/df7460b353c04f85951e5b686c0c18a0~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753267933&x-orig-sign=q995SU06nlD7qER5DnvVmbIIjSo%3D)


相同的操作 处理 `./dataset/images/val` 和 `./dataset/labels/val`

- open Dir 选择上面的`./dataset/images/val`文件夹，作为输入文件夹
- change save Dir 选择上面的`./dataset/labels/val`文件夹，作为输出文件夹


处理完最后再 labels下面的 train和val都会生成对应的坐标信息 
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/fba8eb55f2ff4c0d97c9cc4d44afd517~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753266296&x-orig-sign=M9xKxDKS4ELx1TwZj2hAX%2FM%2B74U%3D)


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/262ad3b01ca74e748a9696dfde3a886d~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753268106&x-orig-sign=cwbwlTPTfvBDdz7h2kjpGYjnEUk%3D)


## 创建配置 code.yaml 文件
设置一下信息
```yaml
path: ./dataset  # 相对于你运行脚本的位置
train: images/train # 训练的图片地址
val: images/val # 训练的图片地址2
nc: 1
names: ['rabbit']
```

##  训练命令（根据图片+标记训练）

```sh
yolo detect train \
  model=yolov8s.pt \
  data=code.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=code_project2 \
  name=exp2
```

这时候就要考验你的硬件能力了，我是漫长的等待

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c404eaa8267940d295cfd6958443a9e4~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753266661&x-orig-sign=ERBGUnXToSmAKKlA0lUFRMOGc%2Bk%3D)

结束后最重要的是 这个模型文件 
code_project2/exp2/weights/best.pt
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/4a5e7c1c7b37427c9f8ba7caa3caf316~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753266748&x-orig-sign=9zEJLPsyjNwuJmb5uDZy3EURBpQ%3D)

我们这时候就可以使用上面这个微调过的新模型进行识别


## 运行命令
code_project2/exp2/weights/best.pt 这个就是上面生成的新模型
```sh
 yolo detect predict \
  model=code_project2/exp2/weights/best.pt \
  source=./mov_bbb.mp4 \
  conf=0.25 \
  save=True
```



![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/cf035ae5b4f2453a9adea05723fd6ea3~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753266877&x-orig-sign=XXgDbBhnh0FtUOa4bEI1xc3E7jw%3D)
# 最后输出结果 
rabbit ， nice ～


![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/c1b6dc4a429545fd8f9cf07b210816b8~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753267293&x-orig-sign=YN4CpIujPIW9GhH2SfWd8yQma1c%3D)


![Jul-23-2025 09-40-36.gif](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5272b37e8b2741b9abd507781416d952~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAgamFzb25feWFuZw==:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMjk3MjcwNDc5NTgwMjY1MyJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1753321268&x-orig-sign=wBxXN3Yv6RqNHPTA8J4wHtAug2Q%3D)

当然还有优化的空间，蝴蝶也被标记成rabbit ，可以继续使用labelImg标记 butterfly

# 参考代码
https://github.com/mjsong07/yolov8_demo
