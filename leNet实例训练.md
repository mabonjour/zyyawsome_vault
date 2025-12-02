 model.py
**train.py**
**test.py**
流程是相似的
# 前向传播
![](assets/leNet实例训练/file-20251201193741833.png)![](assets/leNet实例训练/file-20251201200235698.png)
# 数据集
transforms 加载数据集，转换为张量形式
![](assets/leNet实例训练/file-20251202094218580.png)
# 模型训练代码
```
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):  
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    # 使用Adam优化器，学习率为0.001  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    # 损失函数为交叉熵函数  
    criterion = nn.CrossEntropyLoss()  
    # 将模型放入到训练设备中  
    model = model.to(device)  
    # 复制当前模型的参数  
    best_model_wts = copy.deepcopy(model.state_dict())
```
# 训练模型 反向传播的过程
# 验证过程
