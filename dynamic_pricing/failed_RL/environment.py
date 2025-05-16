import numpy as np
import gym
from gym import spaces
from config import Config

class PricingEnv(gym.Env):
    """
    动态定价环境（强化版离峰时段刺激机制）
    状态空间: [标准化小时, 标准化评分, 标准化价格, 标准化需求, 原始评分, 价格趋势]
    动作空间: 20个折扣级别 (0%~95%)
    """

    def __init__(self):
        super(PricingEnv, self).__init__()
        
        # 初始化动作和状态空间
        self.action_space = spaces.Discrete(Config.ACTION_DIM)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 3.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 基础定价参数
        self.base_price = 100.0       # 基准价格
        self.base_cost = 60.0         # 成本价（base_price的60%）
        self.operating_hours = {      # 运营时段定义
            'peak': [(12, 14), (18, 20)],
            'offpeak': [(10, 12), (14, 18), (20, 22)],
            'closed': [(0, 10), (22, 24)]
        }
        self.reset()

    def reset(self):
        """重置环境状态"""
        self.time_period = np.random.randint(0, 24)  # 随机初始小时
        self.rating = 4.5                            # 初始评分
        self.current_price = self.base_price         # 当前价格
        self.price_history = [self.current_price]    # 价格历史记录
        self.no_discount_streak = 0                  # 连续无折扣计数
        return self._get_state()

    def _get_state(self):
        """构建状态向量"""
        return np.array([
            self.time_period / 23.0,                # 标准化小时 [0-1]
            self.rating / 5.0,                      # 标准化评分 [0-1]
            self.current_price / 150.0,             # 标准化价格 [0-1]
            self._calculate_base_demand() / 100.0,  # 标准化需求 [0-1]
            self.rating,                            # 原始评分 [3-5]
            self._calculate_price_trend()           # 价格趋势 [-1,1]
        ], dtype=np.float32)

    def step(self, action):
        """执行一个时间步"""
        # 1. 解析动作
        discount = action * 0.05  # 转换为折扣率
        
        # 2. 价格计算
        new_price = self.base_price * (1 - discount)
        
        # 3. 需求计算（区分时段）
        if self._is_operating('peak'):
            # 高峰时段：基础需求 + 轻度价格弹性
            demand = self._peak_demand(new_price)
        elif self._is_operating('offpeak'):
            # 离峰时段：强化折扣刺激机制
            demand = self._offpeak_demand(new_price, discount)
        else:
            # 非营业时段无销售
            demand = 0.0
        
        # 4. 评分更新逻辑
        self._update_rating(discount)
        
        # 5. 连续无折扣惩罚
        self._update_discount_streak(action)
        
        # 6. 计算奖励
        reward = self._calculate_reward(new_price, demand, discount)
        
        # 7. 状态更新
        self._update_state(new_price)
        
        done = (self.time_period == 0)  # 以24小时为周期
        return self._get_state(), reward, done, {}

    def _calculate_base_demand(self):
        """基础需求计算（不考虑价格因素）"""
        if self._is_operating('peak'):
            return 100.0 * (self.rating / 5.0)
        elif self._is_operating('offpeak'):
            return 60.0 * (self.rating / 5.0)
        return 0.0

    def _peak_demand(self, price):
        """高峰时段需求计算"""
        price_elasticity = 2.0  # 较高弹性
        return max(20.0, self._calculate_base_demand() - price_elasticity*(price - self.base_price))

    def _offpeak_demand(self, price, discount):
        """离峰时段需求计算（强化折扣刺激）"""
        # 基础需求
        base_demand = self._calculate_base_demand()
        
        # 折扣刺激因子（指数增长）
        discount_boost = np.exp(4.0 * discount)  # 折扣越大增长越快
        
        # 价格弹性（离峰时段弹性较低）
        price_elasticity = 1.2 - discount*2.0    # 折扣越大弹性越低
        
        demand = base_demand * discount_boost - price_elasticity*(price - self.base_price)
        return np.clip(demand, 10.0, 200.0)

    def _update_rating(self, discount):
        """评分动态更新"""
        # 折扣越大评分提升越快
        rating_change = np.random.uniform(-0.05, 0.1 + 0.2*discount)
        self.rating = np.clip(self.rating + rating_change, 3.0, 5.0)

    def _calculate_reward(self, price, demand, discount):
        """奖励函数设计"""
        profit = (price - self.base_cost) * demand
        streak_penalty = 0.5 * self.no_discount_streak**2
        
        # 离峰时段额外奖励
        if self._is_operating('offpeak'):
            offpeak_bonus = 25.0 * discount * demand  # 折扣销量加成
        else:
            offpeak_bonus = 0.0
            
        return 0.5*profit + offpeak_bonus - streak_penalty

    def _update_state(self, new_price):
        """更新环境状态"""
        self.time_period = (self.time_period + 1) % 24
        self.price_history.append(new_price)
        self.current_price = new_price

    def _calculate_price_trend(self):
        """价格趋势计算（最近3小时变化）"""
        if len(self.price_history) >= 4:
            recent = self.price_history[-4:-1]
            return np.tanh((recent[-1] - np.mean(recent)) / 10.0)
        return 0.0

    def _update_discount_streak(self, action):
        """更新连续无折扣计数"""
        if action == 0:
            self.no_discount_streak += 1
        else:
            self.no_discount_streak = 0

    def _is_operating(self, period_type):
        """检查当前是否在指定时段"""
        current_hour = self.time_period
        for start, end in self.operating_hours[period_type]:
            if start <= current_hour < end:
                return True
        return False