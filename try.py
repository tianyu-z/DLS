import json
import matplotlib.pyplot as plt
import re
# 假设你的JSON文件名为data.json
def draw(jsonpath, sourpath):
    workers = ['worker0', 'worker1', 'worker2', 'worker3', 'worker4']
    worker_actions = {worker: [] for worker in workers}
    with open(jsonpath, 'r') as file:
    
        for line in file:
 
            # 初始化每个worker的action列表
          
            
            
            match = re.match(r'"worker(\d+)": (\d+)', line.strip())
            if match:
                worker_number = int(match.group(1))
                action_value = int(match.group(2))
                worker_actions[workers[worker_number]].append(int(action_value))
                # print(worker_number, action_value)
        # 提取时间步和每个worker的action
        # timesteps = list(data.keys())
        # workers = ['worker0', 'worker1', 'worker2', 'worker3', 'worker4']

        # 初始化每个worker的action列表
        # worker_actions = {worker: [] for worker in workers}

        # 填充每个worker的action列表
        # for timestep in timesteps:
            # for worker in workers:
                # worker_actions[worker].append(data[timestep][worker])

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    timesteps = list(range(len(worker_actions[workers[0]])))
    
    for worker in workers:
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, worker_actions[worker], label=worker)
        # print(worker, worker_actions[worker])
        # print(timesteps)\
        
        
        tar = sourpath + 'mode'+ jsonpath[-6] + worker + '.jpg'
        plt.savefig(tar)
        plt.close()
    '''
    plt.xlabel('Time Step')
    plt.ylabel('Action')
    plt.title('Actions of Workers over Time Steps')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # 旋转x轴标签以便更好地显示
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    # plt.show()
    savepath=jsonpath.replace('json','jpg')
    plt.savefig(savepath)
    # plt.close()
    '''
pathlist = ['/mnt/csp/mmvision/home/lwh/DLS/heuristic2_record_choose0.json',
            '/mnt/csp/mmvision/home/lwh/DLS/heuristic2_record_choose1.json',
            '/mnt/csp/mmvision/home/lwh/DLS/heuristic_record_choose2.json',
            '/mnt/csp/mmvision/home/lwh/DLS/heuristic_record_choose3.json',
            '/mnt/csp/mmvision/home/lwh/DLS/heuristic_record_choose4.json',
            
            ]

def count(jsonpath):
    print(jsonpath[-6])
    workers = ['worker0', 'worker1', 'worker2', 'worker3', 'worker4']
    worker_actions = {worker: [] for worker in workers}
    with open(jsonpath, 'r') as file:
    
        for line in file:
 
            # 初始化每个worker的action列表
          
            
            
            match = re.match(r'"worker(\d+)": (\d+)', line.strip())
            if match:
                worker_number = int(match.group(1))
                action_value = int(match.group(2))
                worker_actions[workers[worker_number]].append(int(action_value))
    for worker in workers:
        data = worker_actions[worker]
        count_dict = {i: 0 for i in range(5)}

        # 遍历列表并计数
        for num in data:
            count_dict[num] += 1

        # 计算总数量
        total_count = len(data)

        # 计算每个数字的比例
        proportion_dict = {num: count / total_count for num, count in count_dict.items()}

        # 打印结果
        for num, proportion in proportion_dict.items():
            print(f"{worker} action {num} 占的比例: {proportion:.2%}")
        print('/n')
spath = "/mnt/csp/mmvision/home/lwh/DLS/draw/"
for path in pathlist:
    count(path)