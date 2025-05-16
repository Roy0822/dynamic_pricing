class Config:
    # 环境参数
    STATE_DIM = 6    # 简化后的状态维度
    ACTION_DIM = 20   # 5个折扣级别：0%,5%,10%,15%,20%
    MAX_STEPS = 24   # 模拟24小时
    
    # 训练参数
    BATCH_SIZE = 32
    BUFFER_SIZE = 10000
    GAMMA = 0.95
    LR = 1e-3
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.995
    TARGET_UPDATE = 100