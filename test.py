import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 模拟数据集
class DummyDataset:
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 返回一个随机的输入和目标张量
        return torch.randn(3, 224, 224), torch.tensor([1.0])

# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 224 * 224, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# 训练函数
def train_model(dataset, model, optimizer, criterion, first_iter, total_iters):
    # 创建进度条
    progress_bar = tqdm(range(first_iter, total_iters), desc="Training progress")
    
    # 获取数据加载器
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    for epoch in range(first_iter, total_iters):
        running_loss = 0.0
        for inputs, targets in data_loader:
            # 将数据移动到GPU（如果可用）
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计损失值
            running_loss += loss.item()
        
        # 更新进度条并显示当前损失值
        avg_loss = running_loss / len(data_loader)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
        progress_bar.update(1)  # 更新进度条
        
    progress_bar.close()

# 设置设备 (CPU 或 GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化数据集、模型、优化器和损失函数
dataset = DummyDataset(size=100)
model = SimpleModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()

# 设置训练参数
first_iter = 0
total_iters = 10

# 开始训练
train_model(dataset, model, optimizer, criterion, first_iter, total_iters)