import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

class Worker:
    def __init__(self, model, worker_list):
        self.model = model
        self.worker_list = worker_list

    def act(self, model, action, worker_list):
        # 假设 action 是一个索引，用于选择 worker_list 中的一个模型
        selected_model = worker_list[action]
        return selected_model

    def update_model_with_state_dict(self, action):
        # 使用 load_state_dict 方法更新模型
        self.model.load_state_dict(self.act(self.model, action, self.worker_list).state_dict())

    def update_model_with_direct_assignment(self, action):
        # 直接赋值模型
        self.model = self.act(self.model, action, self.worker_list)

def compare_state_dicts(state_dict1, state_dict2):
    # 检查两个字典的键是否相同
    if state_dict1.keys() != state_dict2.keys():
        return False
    
    # 逐个比较字典中的键值对
    for key in state_dict1:
        if not torch.allclose(state_dict1[key], state_dict2[key]):
            return False
    
    return True

# 示例调用
model1 = SimpleModel()
model2 = SimpleModel()
worker_list = [model1, model2]

worker = Worker(model1, worker_list)

# 假设 action 是一个索引，用于选择 worker_list 中的一个模型
action = 1

# 使用 load_state_dict 方法更新模型
worker.update_model_with_state_dict(action)
print("Model after update_model_with_state_dict:")
print(worker.model)

# 使用直接赋值方法更新模型
worker.update_model_with_direct_assignment(action)
print("Model after update_model_with_direct_assignment:")
print(worker.model)

# 比较两种方法得到的模型是否相同
print("Are the models the same after both updates? (state_dict comparison):",
      compare_state_dicts(worker.model.state_dict(), model2.state_dict()))
print("Are the models the same after both updates? (instance comparison):",
      worker.model is model2)