#!/usr/bin/env python3
"""
2048 AI自动玩家 - 机器学习版

支持两种运行模式:
1. 🧘 冥想模式 - 纯AI模拟训练
   - 在虚拟环境中快速训练，无需真实游戏界面
   - 速度极快（可达数十局/秒）
   - 适合快速生成训练数据集
   - 使用完整的游戏模拟器

2. ⚔️ 实战模式 - 基于图像识别
   - 通过颜色识别和模板匹配控制真实游戏
   - 自动检测游戏区域、识别方块、执行操作
   - 支持自动重启（Try Again按钮识别）
   - 适合实际游戏测试和演示
"""

import time
import random
import sys
import copy
import math
import numpy as np
import pyautogui
import keyboard
import os
import json
from datetime import datetime

# 配置参数
ANIMATION_DELAY = 0.3  # 等待滑块动画完成，避免识别错误
SAVE_SCREENSHOTS = False  # 关闭以提升速度
SCREENSHOT_DIR = "screenshots"
DEBUG_MODE = False  # 关闭调试以提升速度

# Try Again 按钮检测优化配置
BUTTON_CHECK_INTERVAL = 3  # 每隔N步检查一次按钮（减少不必要的检测）
LAST_BUTTON_CHECK_TIME = 0  # 上次检查按钮的时间

# Try Again 按钮相对位置配置（基准：500x500游戏区域）
TRY_AGAIN_RELATIVE_X = 250  # 相对于游戏区域左上角的X偏移（基准500宽）
TRY_AGAIN_RELATIVE_Y = 360  # 相对于游戏区域左上角的Y偏移（基准500高）
TRY_AGAIN_BASE_SIZE = 500   # 基准游戏区域大小（500x500）
USE_RELATIVE_CLICK = True   # 优先使用相对位置点击（更快更准）

# 多尺度匹配说明：
# 为了解决不同屏幕分辨率导致的按钮大小不一致问题，
# 系统会先尝试相对位置点击（最快），
# 如果失败，则尝试 PyAutoGUI 的标准匹配，
# 最后启用 OpenCV 多尺度模板匹配（0.5x-2.0x）

# 机器学习配置
ML_ENABLED = True  # 启用机器学习功能
ML_DATA_FILE = "game_history.json"  # 历史数据文件
ML_AUTO_OPTIMIZE = True  # 自动优化策略
AUTO_RESTART = True  # 自动重启游戏（在训练模式下）
TRY_AGAIN_BUTTON = "try_again.png"  # Try again 按钮图片
SUMMARY_FREQUENCY = 5  # 每N局总结一次经验（1=每局，5=每5局）

# 键盘映射
KEY_MAP = {
    "down": "↓ DOWN",
    "left": "← LEFT",
    "right": "→ RIGHT",
    "up": "↑ UP"
}
ALL_MOVES = ["down", "left", "right", "up"]
MOVE_PREFERENCE = {"down": 0, "left": 1, "right": 2, "up": 3}

# 固定左下角策略（最稳定，参考GitHub项目的简洁思路）
# 复杂的自适应策略容易导致决策不一致
ADAPTIVE_PRIORITY = False  # 禁用自适应，使用固定策略

# 策略参数类
class StrategyParams:
    """策略参数配置"""
    def __init__(self, name="default", 
                 priority_down=2000, priority_left=2000, 
                 priority_right=50, priority_up=1,
                 empty_weight=500000, merge_weight=5000,
                 adjacent_weight=1500, corner="左下"):
        self.name = name
        self.priority = {
            "down": priority_down,
            "left": priority_left,
            "right": priority_right,
            "up": priority_up
        }
        self.empty_weight = empty_weight
        self.merge_weight = merge_weight
        self.adjacent_weight = adjacent_weight
        self.corner = corner  # "左下", "右下", "左上", "右上"
    
    def to_dict(self):
        """转换为字典"""
        return {
            "name": self.name,
            "priority": self.priority,
            "empty_weight": self.empty_weight,
            "merge_weight": self.merge_weight,
            "adjacent_weight": self.adjacent_weight,
            "corner": self.corner
        }
    
    @staticmethod
    def from_dict(data):
        """从字典创建"""
        return StrategyParams(
            name=data.get("name", "custom"),
            priority_down=data["priority"]["down"],
            priority_left=data["priority"]["left"],
            priority_right=data["priority"]["right"],
            priority_up=data["priority"]["up"],
            empty_weight=data.get("empty_weight", 500000),
            merge_weight=data.get("merge_weight", 5000),
            adjacent_weight=data.get("adjacent_weight", 1500),
            corner=data.get("corner", "左下")
        )

# 当前使用的策略
CURRENT_STRATEGY = StrategyParams(name="左下角策略")

# 固定左下角优先级：强化下和左（用户强调）
FIXED_PRIORITY = CURRENT_STRATEGY.priority

# AI算法参数 - 自适应角落策略权重矩阵
WEIGHT_MATRICES = {
    "左下": [  # 左下角策略
        [2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12],
        [2 ** 8,  2 ** 9,  2 ** 10, 2 ** 11],
        [2 ** 7,  2 ** 6,  2 ** 5,  2 ** 4],
        [2 ** 17, 2 ** 16, 2 ** 2,  2 ** 0]
    ],
    "右下": [  # 右下角策略
        [2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15],
        [2 ** 11, 2 ** 10, 2 ** 9,  2 ** 8],
        [2 ** 4,  2 ** 5,  2 ** 6,  2 ** 7],
        [2 ** 0,  2 ** 2,  2 ** 16, 2 ** 17]
    ],
    "左上": [  # 左上角策略
        [2 ** 17, 2 ** 16, 2 ** 2,  2 ** 0],
        [2 ** 7,  2 ** 6,  2 ** 5,  2 ** 4],
        [2 ** 8,  2 ** 9,  2 ** 10, 2 ** 11],
        [2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12]
    ],
    "右上": [  # 右上角策略
        [2 ** 0,  2 ** 2,  2 ** 16, 2 ** 17],
        [2 ** 4,  2 ** 5,  2 ** 6,  2 ** 7],
        [2 ** 11, 2 ** 10, 2 ** 9,  2 ** 8],
        [2 ** 12, 2 ** 13, 2 ** 14, 2 ** 15]
    ]
}

# 默认使用左下角策略（固定策略，参考GitHub项目）
WEIGHT_MATRIX = [  # 左下角权重，指数级递增
    [2 ** 15, 2 ** 14, 2 ** 13, 2 ** 12],
    [2 ** 8,  2 ** 9,  2 ** 10, 2 ** 11],
    [2 ** 7,  2 ** 6,  2 ** 5,  2 ** 4],
    [2 ** 17, 2 ** 16, 2 ** 2,  2 ** 0]
]
# 权重参数（用户反馈：相同数字合并最重要）- 从策略中动态获取
def get_empty_cell_weight():
    return CURRENT_STRATEGY.empty_weight

def get_merge_potential_weight():
    return CURRENT_STRATEGY.merge_weight

def get_adjacent_similar_weight():
    return CURRENT_STRATEGY.adjacent_weight

EMPTY_CELL_WEIGHT_FACTOR = 500000   # 默认值
MERGE_POTENTIAL_WEIGHT = 5000       # 默认值
ADJACENT_SIMILAR_WEIGHT = 1500      # 默认值
MONOTONICITY_WEIGHT = 1.0           
SMOOTHNESS_WEIGHT = 0.05

CACHE = {}

# 游戏区域检测参数
GAME_REGION = None

# 2048游戏标准颜色方案（RGB值）
NUMBER_COLORS = {
    0: (205, 193, 180),  # 空格 #CDC1B4
    2: (238, 228, 218),  # 2 #EEE4DA
    4: (237, 224, 200),  # 4 #EDE0C8
    8: (242, 177, 121),  # 8 #F2B179
    16: (245, 149, 99),  # 16 #F59563
    32: (246, 124, 95),  # 32 #F67C5F
    64: (246, 94, 59),  # 64 #F65E3B
    128: (237, 207, 114),  # 128 #EDCF72
    256: (237, 204, 97),  # 256 #EDCC61     237,204,97
    512: (237, 200, 80),  # 512 #EDC850     237,200,80
    1024: (237, 197, 63),  # 1024 #EDC53F
    2048: (237, 194, 46),  # 2048 #EDC22E
}


# ==================== 机器学习模块 ====================
class GameRecorder:
    """游戏数据记录器"""
    def __init__(self, data_file=ML_DATA_FILE):
        self.data_file = data_file
        self.current_game = None
        self.history = self.load_history()
    
    def load_history(self):
        """加载历史数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载历史数据失败: {e}")
                return {"games": [], "best_strategy": None}
        return {"games": [], "best_strategy": None}
    
    def save_history(self):
        """保存历史数据"""
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            print(f"✓ 游戏数据已保存到 {self.data_file}")
        except Exception as e:
            print(f"保存历史数据失败: {e}")
    
    def start_game(self, strategy: StrategyParams):
        """开始记录新游戏"""
        self.current_game = {
            "start_time": datetime.now().isoformat(),
            "strategy": strategy.to_dict(),
            "max_tile": 0,
            "final_score": 0,
            "moves_count": 0,
            "duration": 0,
            "success": False  # 是否达到2048
        }
    
    def update_game(self, matrix, score, moves):
        """更新游戏状态"""
        if self.current_game:
            max_tile = max(max(row) for row in matrix)
            self.current_game["max_tile"] = max(max_tile, self.current_game["max_tile"])
            self.current_game["final_score"] = score
            self.current_game["moves_count"] = moves
    
    def end_game(self, matrix, score, moves):
        """结束游戏记录"""
        if not self.current_game:
            return
        
        max_tile = max(max(row) for row in matrix)
        self.current_game["max_tile"] = max(max_tile, self.current_game["max_tile"])
        self.current_game["final_score"] = score
        self.current_game["moves_count"] = moves
        self.current_game["success"] = (max_tile >= 2048)
        
        # 计算游戏时长
        start_time = datetime.fromisoformat(self.current_game["start_time"])
        duration = (datetime.now() - start_time).total_seconds()
        self.current_game["duration"] = round(duration, 2)
        self.current_game["end_time"] = datetime.now().isoformat()
        
        # 添加到历史记录
        self.history["games"].append(self.current_game)
        
        # 打印游戏总结
        self.print_game_summary()
        
        # 更新学习器
        if ADAPTIVE_LEARNER:
            ADAPTIVE_LEARNER.update_all_learners(self.current_game)
        
        # 判断是否需要阶段性总结
        total_games = len(self.history["games"])
        if total_games % SUMMARY_FREQUENCY == 0:
            self.print_phase_summary()
        
        # 保存数据
        self.save_history()
        
        # 更新最佳策略
        if ML_AUTO_OPTIMIZE:
            self.update_best_strategy()
        
        self.current_game = None
    
    def print_phase_summary(self):
        """打印阶段性总结"""
        total = len(self.history["games"])
        recent_games = self.history["games"][-SUMMARY_FREQUENCY:]
        
        print("\n" + "="*60)
        print(f"📊 阶段性总结 (最近 {len(recent_games)} 局)")
        print("="*60)
        
        # 统计最近表现
        recent_max_tiles = [g["max_tile"] for g in recent_games]
        recent_scores = [g["final_score"] for g in recent_games]
        recent_success = sum(1 for g in recent_games if g["success"])
        
        print(f"最近成功率: {recent_success}/{len(recent_games)} ({recent_success/len(recent_games)*100:.1f}%)")
        print(f"平均最大方块: {sum(recent_max_tiles)/len(recent_max_tiles):.0f}")
        print(f"平均分数: {sum(recent_scores)/len(recent_scores):.0f}")
        print(f"最高方块: {max(recent_max_tiles)}")
        
        # 进化趋势
        if total >= SUMMARY_FREQUENCY * 2:
            prev_games = self.history["games"][-SUMMARY_FREQUENCY*2:-SUMMARY_FREQUENCY]
            prev_avg_tile = sum(g["max_tile"] for g in prev_games) / len(prev_games)
            curr_avg_tile = sum(recent_max_tiles) / len(recent_max_tiles)
            
            improvement = curr_avg_tile - prev_avg_tile
            print(f"\n进化趋势: {improvement:+.0f} (相比前 {SUMMARY_FREQUENCY} 局)")
            
            if improvement > 0:
                print("🎉 性能提升中！")
            elif improvement < -100:
                print("⚠️  性能下降，建议调整策略")
        
        print("="*60)
    
    def print_game_summary(self):
        """打印游戏总结"""
        print("\n" + "="*60)
        print("📊 游戏总结")
        print("="*60)
        print(f"策略名称: {self.current_game['strategy']['name']}")
        print(f"最大方块: {self.current_game['max_tile']}")
        print(f"最终分数: {self.current_game['final_score']}")
        print(f"移动步数: {self.current_game['moves_count']}")
        print(f"游戏时长: {self.current_game['duration']}秒")
        print(f"达到2048: {'✓ 是' if self.current_game['success'] else '✗ 否'}")
        print("="*60)
        
        # 显示策略参数
        strategy = self.current_game['strategy']
        print(f"\n策略参数:")
        print(f"  优先级: {strategy['priority']}")
        print(f"  空格权重: {strategy['empty_weight']}")
        print(f"  合并权重: {strategy['merge_weight']}")
        print(f"  相邻权重: {strategy['adjacent_weight']}")
        print(f"  角落位置: {strategy['corner']}")
        print("="*60)
    
    def update_best_strategy(self):
        """更新最佳策略"""
        if not self.history["games"]:
            return
        
        # 按最大方块排序，然后按分数排序
        sorted_games = sorted(
            self.history["games"],
            key=lambda x: (x["max_tile"], x["final_score"]),
            reverse=True
        )
        
        best_game = sorted_games[0]
        self.history["best_strategy"] = best_game["strategy"]
        
        print(f"\n🏆 最佳策略更新:")
        print(f"   策略名称: {best_game['strategy']['name']}")
        print(f"   最大方块: {best_game['max_tile']}")
        print(f"   最终分数: {best_game['final_score']}")
    
    def get_statistics(self):
        """获取统计信息"""
        if not self.history["games"]:
            return None
        
        games = self.history["games"]
        total_games = len(games)
        success_count = sum(1 for g in games if g["success"])
        
        max_tiles = [g["max_tile"] for g in games]
        scores = [g["final_score"] for g in games]
        
        stats = {
            "total_games": total_games,
            "success_count": success_count,
            "success_rate": round(success_count / total_games * 100, 2),
            "avg_max_tile": round(sum(max_tiles) / total_games, 2),
            "best_max_tile": max(max_tiles),
            "avg_score": round(sum(scores) / total_games, 2),
            "best_score": max(scores)
        }
        
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        if not stats:
            print("暂无游戏数据")
            return
        
        print("\n" + "="*60)
        print("📈 历史统计")
        print("="*60)
        print(f"总游戏次数: {stats['total_games']}")
        print(f"达到2048次数: {stats['success_count']}")
        print(f"成功率: {stats['success_rate']}%")
        print(f"平均最大方块: {stats['avg_max_tile']}")
        print(f"历史最大方块: {stats['best_max_tile']}")
        print(f"平均分数: {stats['avg_score']}")
        print(f"历史最高分: {stats['best_score']}")
        print("="*60)
    
    def get_best_strategy(self):
        """获取最佳策略"""
        if self.history["best_strategy"]:
            return StrategyParams.from_dict(self.history["best_strategy"])
        return None


class StrategyGenerator:
    """策略生成器 - 生成不同的策略变体用于测试"""
    
    @staticmethod
    def generate_corner_strategies():
        """生成四个角落的策略"""
        strategies = []
        
        corners = {
            "左下": {"down": 2000, "left": 2000, "right": 50, "up": 1},
            "右下": {"down": 2000, "right": 2000, "left": 50, "up": 1},
            "左上": {"up": 2000, "left": 2000, "right": 50, "down": 1},
            "右上": {"up": 2000, "right": 2000, "left": 50, "down": 1}
        }
        
        for corner_name, priority in corners.items():
            strategy = StrategyParams(
                name=f"{corner_name}角策略",
                priority_down=priority["down"],
                priority_left=priority["left"],
                priority_right=priority["right"],
                priority_up=priority["up"],
                corner=corner_name
            )
            strategies.append(strategy)
        
        return strategies
    
    @staticmethod
    def generate_weight_variants(base_strategy: StrategyParams, count=5):
        """基于基础策略生成权重变体"""
        strategies = []
        
        # 生成不同的权重组合
        weight_configs = [
            (500000, 5000, 1500),   # 原始
            (600000, 6000, 1800),   # 增强版
            (400000, 4000, 1200),   # 保守版
            (500000, 8000, 2000),   # 激进合并
            (700000, 3000, 1000),   # 保守空格
        ]
        
        for i, (empty_w, merge_w, adjacent_w) in enumerate(weight_configs[:count]):
            strategy = StrategyParams(
                name=f"{base_strategy.name}_变体{i+1}",
                priority_down=base_strategy.priority["down"],
                priority_left=base_strategy.priority["left"],
                priority_right=base_strategy.priority["right"],
                priority_up=base_strategy.priority["up"],
                empty_weight=empty_w,
                merge_weight=merge_w,
                adjacent_weight=adjacent_w,
                corner=base_strategy.corner
            )
            strategies.append(strategy)
        
        return strategies


# 全局记录器
GAME_RECORDER = GameRecorder() if ML_ENABLED else None


# ==================== 高级机器学习算法 ====================
class BayesianOptimizer:
    """贝叶斯优化器 - 智能参数搜索"""
    
    def __init__(self, recorder: GameRecorder):
        self.recorder = recorder
        self.param_ranges = {
            'priority_down': (1000, 3000),
            'priority_left': (1000, 3000),
            'priority_right': (10, 200),
            'priority_up': (1, 100),
            'empty_weight': (300000, 700000),
            'merge_weight': (3000, 10000),
            'adjacent_weight': (1000, 3000)
        }
    
    def suggest_next_params(self):
        """基于历史数据建议下一组参数"""
        if not self.recorder.history["games"] or len(self.recorder.history["games"]) < 3:
            # 初期随机探索
            return self._random_params()
        
        # 分析表现最好的策略
        top_games = sorted(
            self.recorder.history["games"],
            key=lambda x: (x["max_tile"], x["final_score"]),
            reverse=True
        )[:3]
        
        # 基于最佳策略进行小幅度变异
        best_strategy = top_games[0]["strategy"]
        new_params = self._mutate_params(best_strategy)
        
        return new_params
    
    def _random_params(self):
        """随机生成参数"""
        import random
        params = {}
        for key, (min_val, max_val) in self.param_ranges.items():
            params[key] = random.randint(min_val, max_val)
        return params
    
    def _mutate_params(self, base_strategy, mutation_rate=0.2):
        """对参数进行变异"""
        import random
        
        new_params = {}
        priority = base_strategy['priority']
        
        # 变异优先级参数
        for key in ['down', 'left', 'right', 'up']:
            base_val = priority[key]
            if random.random() < mutation_rate:
                delta = int(base_val * random.uniform(-0.3, 0.3))
                new_val = max(1, base_val + delta)
                new_params[f'priority_{key}'] = new_val
            else:
                new_params[f'priority_{key}'] = base_val
        
        # 变异权重参数
        for key in ['empty_weight', 'merge_weight', 'adjacent_weight']:
            base_val = base_strategy[key]
            if random.random() < mutation_rate:
                delta = int(base_val * random.uniform(-0.3, 0.3))
                min_val, max_val = self.param_ranges[key]
                new_val = max(min_val, min(max_val, base_val + delta))
                new_params[key] = new_val
            else:
                new_params[key] = base_val
        
        return new_params


class GeneticAlgorithm:
    """遗传算法 - 种群进化优化"""
    
    def __init__(self, recorder: GameRecorder, population_size=10):
        self.recorder = recorder
        self.population_size = population_size
        self.generation = 0
    
    def evolve_population(self):
        """进化一代种群"""
        games = self.recorder.history["games"]
        if len(games) < self.population_size:
            # 种群不足，生成新个体
            return self._generate_initial_population()
        
        # 选择最优个体
        sorted_games = sorted(
            games[-self.population_size:],
            key=lambda x: self._fitness(x),
            reverse=True
        )
        
        # 精英保留
        elite_count = max(2, self.population_size // 5)
        elites = sorted_games[:elite_count]
        
        # 交叉和变异生成新一代
        new_generation = []
        for elite in elites:
            new_generation.append(elite["strategy"])
        
        while len(new_generation) < self.population_size:
            # 选择两个父代
            parent1 = random.choice(sorted_games[:self.population_size//2])
            parent2 = random.choice(sorted_games[:self.population_size//2])
            
            # 交叉
            child = self._crossover(parent1["strategy"], parent2["strategy"])
            
            # 变异
            if random.random() < 0.3:
                child = self._mutate(child)
            
            new_generation.append(child)
        
        self.generation += 1
        return new_generation
    
    def _fitness(self, game):
        """计算适应度"""
        max_tile = game["max_tile"]
        score = game["final_score"]
        success = 1000 if game["success"] else 0
        return max_tile * 100 + score + success
    
    def _crossover(self, parent1, parent2):
        """交叉操作"""
        child = {
            "name": f"Gen{self.generation}_Cross",
            "priority": {},
            "corner": random.choice([parent1["corner"], parent2["corner"]])
        }
        
        # 优先级交叉
        for key in ['down', 'left', 'right', 'up']:
            if random.random() < 0.5:
                child["priority"][key] = parent1["priority"][key]
            else:
                child["priority"][key] = parent2["priority"][key]
        
        # 权重交叉
        for key in ['empty_weight', 'merge_weight', 'adjacent_weight']:
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return child
    
    def _mutate(self, strategy):
        """变异操作"""
        mutated = copy.deepcopy(strategy)
        
        # 随机变异一个参数
        param_type = random.choice(['priority', 'weight'])
        
        if param_type == 'priority':
            key = random.choice(['down', 'left', 'right', 'up'])
            base_val = mutated["priority"][key]
            mutated["priority"][key] = max(1, int(base_val * random.uniform(0.7, 1.3)))
        else:
            key = random.choice(['empty_weight', 'merge_weight', 'adjacent_weight'])
            base_val = mutated[key]
            mutated[key] = max(1000, int(base_val * random.uniform(0.7, 1.3)))
        
        return mutated
    
    def _generate_initial_population(self):
        """生成初始种群"""
        population = []
        
        # 基础策略
        base_strategies = StrategyGenerator.generate_corner_strategies()
        population.extend([s.to_dict() for s in base_strategies])
        
        # 随机变体
        while len(population) < self.population_size:
            base = random.choice(base_strategies)
            variant = self._mutate(base.to_dict())
            population.append(variant)
        
        return population


class ReinforcementLearner:
    """强化学习 - Q学习简化版"""
    
    def __init__(self, recorder: GameRecorder):
        self.recorder = recorder
        self.q_table = {}  # 状态-动作值表
        self.learning_rate = 0.1
        self.discount_factor = 0.95
    
    def update_q_values(self, game_data):
        """根据游戏结果更新Q值"""
        max_tile = game_data["max_tile"]
        moves = game_data["moves_count"]
        strategy = game_data["strategy"]
        
        # 计算奖励
        reward = self._calculate_reward(max_tile, moves)
        
        # 简化的状态表示
        state = self._get_state_key(strategy)
        
        # 更新Q值
        if state not in self.q_table:
            self.q_table[state] = reward
        else:
            old_value = self.q_table[state]
            self.q_table[state] = old_value + self.learning_rate * (reward - old_value)
    
    def _calculate_reward(self, max_tile, moves):
        """计算奖励"""
        tile_reward = math.log2(max_tile) if max_tile > 0 else 0
        efficiency_reward = tile_reward / (moves + 1) * 100
        return tile_reward * 100 + efficiency_reward
    
    def _get_state_key(self, strategy):
        """获取状态键"""
        return f"{strategy['corner']}_{strategy['priority']['down']}_{strategy['priority']['left']}"
    
    def get_best_strategy_by_q(self):
        """根据Q值获取最佳策略"""
        if not self.q_table:
            return None
        
        best_state = max(self.q_table.items(), key=lambda x: x[1])[0]
        
        # 从历史中找到对应的策略
        for game in self.recorder.history["games"]:
            state_key = self._get_state_key(game["strategy"])
            if state_key == best_state:
                return StrategyParams.from_dict(game["strategy"])
        
        return None


class AdaptiveLearner:
    """自适应学习器 - 综合多种算法"""
    
    def __init__(self, recorder: GameRecorder):
        self.recorder = recorder
        self.bayesian = BayesianOptimizer(recorder)
        self.genetic = GeneticAlgorithm(recorder)
        self.rl = ReinforcementLearner(recorder)
    
    def suggest_next_strategy(self, mode='auto'):
        """建议下一个策略"""
        games_count = len(self.recorder.history["games"])
        
        if mode == 'auto':
            # 自动选择算法
            if games_count < 5:
                # 初期：随机探索
                mode = 'random'
            elif games_count < 20:
                # 中期：贝叶斯优化
                mode = 'bayesian'
            else:
                # 后期：遗传算法
                mode = 'genetic'
        
        if mode == 'bayesian':
            params = self.bayesian.suggest_next_params()
            return self._params_to_strategy(params, f"Bayesian_{games_count}")
        
        elif mode == 'genetic':
            population = self.genetic.evolve_population()
            best = population[0]
            return StrategyParams.from_dict(best)
        
        elif mode == 'reinforcement':
            strategy = self.rl.get_best_strategy_by_q()
            return strategy if strategy else self._random_strategy()
        
        else:  # random
            return self._random_strategy()
    
    def _params_to_strategy(self, params, name):
        """将参数转换为策略对象"""
        return StrategyParams(
            name=name,
            priority_down=params['priority_down'],
            priority_left=params['priority_left'],
            priority_right=params['priority_right'],
            priority_up=params['priority_up'],
            empty_weight=params['empty_weight'],
            merge_weight=params['merge_weight'],
            adjacent_weight=params['adjacent_weight'],
            corner="左下"
        )
    
    def _random_strategy(self):
        """生成随机策略"""
        corners = ["左下", "右下", "左上", "右上"]
        return StrategyParams(
            name=f"Random_{len(self.recorder.history['games'])}",
            priority_down=random.randint(1000, 3000),
            priority_left=random.randint(1000, 3000),
            priority_right=random.randint(10, 200),
            priority_up=random.randint(1, 100),
            empty_weight=random.randint(300000, 700000),
            merge_weight=random.randint(3000, 10000),
            adjacent_weight=random.randint(1000, 3000),
            corner=random.choice(corners)
        )
    
    def update_all_learners(self, game_data):
        """更新所有学习器"""
        self.rl.update_q_values(game_data)


# 全局自适应学习器
ADAPTIVE_LEARNER = AdaptiveLearner(GAME_RECORDER) if ML_ENABLED and GAME_RECORDER else None


def create_screenshot_dir():
    """创建截图目录"""
    if SAVE_SCREENSHOTS and not os.path.exists(SCREENSHOT_DIR):
        os.makedirs(SCREENSHOT_DIR)


def take_screenshot(region=None, save_name=None):
    """截取屏幕截图"""
    try:
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()

        if SAVE_SCREENSHOTS and save_name:
            screenshot.save(os.path.join(SCREENSHOT_DIR, f"{save_name}.png"))
            if DEBUG_MODE:
                print(f"已保存截图: {save_name}.png")

        return screenshot
    except Exception as e:
        print(f"截图失败: {e}")
        return None


def detect_game_region_manual():
    """手动检测游戏区域"""
    global GAME_REGION

    print("请将鼠标移动到游戏区域的左上角，然后按空格键...")
    keyboard.wait('space')
    start_pos = pyautogui.position()
    print(f"左上角位置: {start_pos}")

    print("请将鼠标移动到游戏区域的右下角，然后按空格键...")
    keyboard.wait('space')
    end_pos = pyautogui.position()
    print(f"右下角位置: {end_pos}")

    # 计算区域
    x = min(start_pos.x, end_pos.x)
    y = min(start_pos.y, end_pos.y)
    w = abs(end_pos.x - start_pos.x)
    h = abs(end_pos.y - start_pos.y)

    GAME_REGION = (x, y, w, h)
    print(f"游戏区域: {GAME_REGION}")
    return True


def detect_game_region():
    """检测游戏区域"""
    return detect_game_region_manual()


def extract_cell_image(screenshot, row, col):
    """从截图中提取指定位置的单元格图像"""
    if not GAME_REGION:
        return None

    x, y, width, height = GAME_REGION
    cell_width = width // 4
    cell_height = height // 4

    # 计算单元格在游戏区域内的相对位置
    cell_x = col * cell_width
    cell_y = row * cell_height

    # 使用更大的边距以避免边框，约8%的单元格大小
    margin = int(min(cell_width, cell_height) * 0.08)

    # 计算边界，相对于游戏区域截图
    left = max(0, cell_x + margin)
    top = max(0, cell_y + margin)
    right = min(screenshot.width, cell_x + cell_width - margin)
    bottom = min(screenshot.height, cell_y + cell_height - margin)

    # 确保边界有效
    if left >= right or top >= bottom:
        return None

    cell_region = (left, top, right, bottom)
    cell_image = screenshot.crop(cell_region)

    return cell_image


def color_distance(c1, c2):
    """计算两个颜色的距离"""
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)


def get_dominant_color_exact(cell_image):
    """获取精确的主要颜色 - 使用固定网格采样"""
    if not cell_image:
        return None, None
    
    # 转换为numpy数组
    img_array = np.array(cell_image)
    height, width = img_array.shape[:2]
    
    # 使用固定网格采样（稳定可靠）
    # 在中心区域按网格采样
    samples = []
    step_y = max(1, height // 10)  # 纵向10个点
    step_x = max(1, width // 10)   # 横向10个点
    
    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            if y < height and x < width:
                pixel = tuple(int(v) for v in img_array[y, x][:3])
                samples.append(pixel)
    
    # 统计所有颜色
    color_count = {}
    for pixel in samples:
        color_count[pixel] = color_count.get(pixel, 0) + 1
    
    if not color_count:
        return (205, 193, 180), {'total_samples': len(samples), 'color_stats': {}}
    
    # 按出现次数排序
    sorted_colors = sorted(color_count.items(), key=lambda x: x[1], reverse=True)
    
    debug_info = {
        'total_samples': len(samples),
        'color_stats': dict(sorted_colors[:10])
    }
    
    # 返回最常见的颜色
    dominant_color = sorted_colors[0][0]
    return dominant_color, debug_info


def recognize_number_by_color(cell_image):
    """通过精确颜色识别数字"""
    if not cell_image:
        return 0
    
    # 获取主要颜色（精确）
    dominant_color, debug_info = get_dominant_color_exact(cell_image)
    if not dominant_color:
        return 0
    
    r, g, b = dominant_color
    
    if DEBUG_MODE:
        print(f"  主要颜色: RGB({r}, {g}, {b})")
        print(f"  采样: {debug_info['total_samples']}个点")
        print(f"  颜色统计: ", end="")
        # 显示前5个最常见的颜色
        sorted_colors = sorted(debug_info['color_stats'].items(), key=lambda x: x[1], reverse=True)
        for i, (color, count) in enumerate(sorted_colors[:5]):
            print(f"{color}({count}次)", end=" ")
        print()
    
    # 先尝试精确匹配
    if dominant_color in NUMBER_COLORS.values():
        # 找到对应的数字
        for number, color in NUMBER_COLORS.items():
            if color == dominant_color:
                if DEBUG_MODE:
                    print(f"  ✓ 精确匹配: {number}")
                return number
    
    # 如果精确匹配失败，找最接近的
    min_distance = float('inf')
    best_match = 0
    distances = {}
    
    for number, standard_color in NUMBER_COLORS.items():
        distance = color_distance(dominant_color, standard_color)
        distances[number] = distance
        if distance < min_distance:
            min_distance = distance
            best_match = number
    
    if DEBUG_MODE:
        # 显示前5个最接近的
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])[:5]
        print(f"  最接近: ", end="")
        for num, dist in sorted_distances:
            print(f"{num}({dist:.1f}) ", end="")
        print()
        print(f"  ✓ 近似匹配: {best_match} (距离{min_distance:.1f})")
    
    return best_match


def get_board_matrix_from_screenshot():
    """从截图中解析游戏棋盘 - 优化版"""
    global GAME_REGION

    if not GAME_REGION:
        print("❌ 游戏区域未设置")
        return None

    try:
        # 截取游戏区域
        save_name = f"board_{int(time.time())}" if SAVE_SCREENSHOTS else None
        screenshot = take_screenshot(region=GAME_REGION, save_name=save_name)
        if not screenshot:
            print("❌ 无法截取游戏区域")
            return None

        matrix = [[0] * 4 for _ in range(4)]

        if DEBUG_MODE:
            print("\n" + "="*70)
            print("开始识别棋盘")
            print("="*70)

        # 快速识别所有单元格
        failed_cells = []
        for row in range(4):
            for col in range(4):
                if DEBUG_MODE:
                    print(f"\n【格子 ({row}, {col})】")
                try:
                    cell_image = extract_cell_image(screenshot, row, col)
                    if cell_image is None:
                        if DEBUG_MODE:
                            print("  ❌ 无法提取单元格")
                        failed_cells.append((row, col))
                        continue

                    # 保存单元格用于调试
                    if SAVE_SCREENSHOTS:
                        cell_filename = f"debug_cell_{row}_{col}.png"
                        cell_image.save(os.path.join(SCREENSHOT_DIR, cell_filename))
                        if DEBUG_MODE:  
                            print(f"  已保存: {cell_filename}")

                    number = recognize_number_by_color(cell_image)
                    matrix[row][col] = number

                except Exception as e:
                    if DEBUG_MODE:
                        print(f"  ❌ 处理失败: {e}")
                        import traceback
                        traceback.print_exc()
                    failed_cells.append((row, col))
                    continue

        # 检查识别结果质量
        if len(failed_cells) > 4:
            print(f"⚠️  警告: {len(failed_cells)} 个单元格识别失败")

        if DEBUG_MODE:
            print("\n" + "="*70)
            print("最终识别结果:")
            print("="*70)
            for i, row in enumerate(matrix):
                print(f"第{i}行: {row}")
            print()
            print("可视化:")
            for row in matrix:
                line = ""
                for cell in row:
                    if cell == 0:
                        line += "   .   "
                    else:
                        line += f" {cell:>4}  "
                print(line)
            print("="*70)

        return matrix
        
    except Exception as e:
        print(f"❌ 识别棋盘时发生错误: {e}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        return None


def get_score_from_screenshot():
    """从截图中识别分数"""
    # 实战模式会通过 ScoreTracker 计算分数
    return 0


# ==================== 分数追踪器 ====================
class ScoreTracker:
    """分数追踪器 - 通过棋盘变化计算分数"""
    
    def __init__(self):
        self.total_score = 0
        self.last_matrix = None
    
    def reset(self):
        """重置分数"""
        self.total_score = 0
        self.last_matrix = None
    
    def update(self, old_matrix, new_matrix):
        """根据棋盘变化计算并累加分数
        
        Args:
            old_matrix: 移动前的棋盘
            new_matrix: 移动后的棋盘
        
        Returns:
            本次移动获得的分数
        """
        if old_matrix is None or new_matrix is None:
            return 0
        
        # 计算所有方块的总和差异
        old_sum = sum(sum(row) for row in old_matrix)
        new_sum = sum(sum(row) for row in new_matrix)
        
        # 新生成的方块（2或4）
        # 计算新增的方块值（通常是2或4）
        new_tiles_sum = 0
        for r in range(4):
            for c in range(4):
                if old_matrix[r][c] == 0 and new_matrix[r][c] != 0:
                    new_tiles_sum += new_matrix[r][c]
        
        # 得分 = 总和增加 - 新生成的方块
        # 例如：64+64=128，总和增加128-64-64=0，但合并得128分
        # 实际计算：(new_sum - old_sum) - new_tiles = 合并得分
        score_gain = (new_sum - old_sum) - new_tiles_sum
        
        if score_gain > 0:
            self.total_score += score_gain
            if DEBUG_MODE:
                print(f"  💰 本次得分: +{score_gain} | 总分: {self.total_score}")
        
        self.last_matrix = [row[:] for row in new_matrix]
        return score_gain
    
    def get_total_score(self):
        """获取总分"""
        return self.total_score


# 全局分数追踪器
SCORE_TRACKER = ScoreTracker()


def detect_try_again_button():
    """检测 Try again 按钮是否出现 - 多尺度匹配优化版"""
    try:
        if not os.path.exists(TRY_AGAIN_BUTTON):
            if DEBUG_MODE:
                print(f"Try again 按钮图片不存在: {TRY_AGAIN_BUTTON}")
            return None
        
        # 尝试不同的置信度阈值（适应不同分辨率）
        confidence_levels = [0.8, 0.75, 0.7, 0.65, 0.6]
        
        for confidence in confidence_levels:
            try:
                # 在屏幕上查找 Try again 按钮（不使用灰度匹配，保持精准）
                button_location = pyautogui.locateOnScreen(
                    TRY_AGAIN_BUTTON, 
                    confidence=confidence
                )
                
                if button_location:
                    if DEBUG_MODE:
                        print(f"✓ 检测到 Try again 按钮 (confidence={confidence}): {button_location}")
                    return button_location
            except pyautogui.ImageNotFoundException:
                # 这个特定异常表示未找到，继续尝试更低的阈值
                continue
            except Exception as e:
                # 其他错误（如缺少 opencv）
                if "confidence" in str(e).lower():
                    # 如果不支持 confidence 参数，回退到基础方法
                    if DEBUG_MODE:
                        print("警告: 不支持 confidence 参数，使用基础匹配")
                    button_location = pyautogui.locateOnScreen(TRY_AGAIN_BUTTON)
                    if button_location:
                        if DEBUG_MODE:
                            print(f"✓ 检测到 Try again 按钮 (基础匹配): {button_location}")
                        return button_location
                    break
                else:
                    raise
        
        # 如果都失败了，尝试使用 OpenCV 多尺度匹配（适应不同分辨率）
        try:
            return detect_try_again_button_multiscale()
        except:
            pass
        
        return None
    except Exception as e:
        if DEBUG_MODE:
            print(f"检测 Try again 按钮失败: {e}")
        return None


def detect_try_again_button_multiscale():
    """使用 OpenCV 多尺度模板匹配检测按钮（解决分辨率问题）"""
    try:
        import cv2
        from PIL import ImageGrab
        
        # 截取整个屏幕
        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        screen_gray = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
        
        # 读取模板
        template = cv2.imread(TRY_AGAIN_BUTTON)
        if template is None:
            return None
        
        template_gray = template
        h, w = template_gray.shape[:2]
        
        # 尝试不同的缩放比例（0.5x 到 2x）
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
        best_match = None
        best_val = 0
        
        for scale in scales:
            # 缩放模板
            resized_template = cv2.resize(template_gray, 
                                         (int(w * scale), int(h * scale)))
            th, tw = resized_template.shape[:2]
            
            # 如果模板比屏幕大，跳过
            if th > screen_gray.shape[0] or tw > screen_gray.shape[1]:
                continue
            
            # 模板匹配
            result = cv2.matchTemplate(screen_gray, resized_template, 
                                      cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # 保存最佳匹配
            if max_val > best_val:
                best_val = max_val
                best_match = {
                    'left': max_loc[0],
                    'top': max_loc[1],
                    'width': tw,
                    'height': th,
                    'scale': scale,
                    'confidence': max_val
                }
        
        # 如果找到了较好的匹配（阈值 0.5）
        if best_match and best_val > 0.5:
            if DEBUG_MODE:
                print(f"✓ OpenCV多尺度检测到按钮 (scale={best_match['scale']:.1f}, conf={best_val:.2f})")
            
            # 转换为 pyautogui 的格式
            class ButtonLocation:
                def __init__(self, left, top, width, height):
                    self.left = left
                    self.top = top
                    self.width = width
                    self.height = height
            
            return ButtonLocation(best_match['left'], best_match['top'], 
                                 best_match['width'], best_match['height'])
        
        return None
        
    except Exception as e:
        if DEBUG_MODE:
            print(f"OpenCV多尺度匹配失败: {e}")
        return None


def calculate_try_again_position():
    """根据游戏区域大小计算 Try Again 按钮的相对位置"""
    global GAME_REGION
    
    if not GAME_REGION:
        return None
    
    x, y, width, height = GAME_REGION
    
    # 计算缩放比例（相对于500x500基准）
    scale_x = width / TRY_AGAIN_BASE_SIZE
    scale_y = height / TRY_AGAIN_BASE_SIZE
    
    # 计算实际位置（游戏区域左上角 + 按比例缩放的偏移）
    actual_x = x + int(TRY_AGAIN_RELATIVE_X * scale_x)
    actual_y = y + int(TRY_AGAIN_RELATIVE_Y * scale_y)
    
    if DEBUG_MODE:
        print(f"Try Again 相对位置计算:")
        print(f"  游戏区域: {width}x{height}")
        print(f"  缩放比例: X={scale_x:.2f}, Y={scale_y:.2f}")
        print(f"  计算位置: ({actual_x}, {actual_y})")
    
    return (actual_x, actual_y)


def click_try_again_by_position(retry_count=3):
    """使用相对位置点击 Try Again 按钮（快速准确）"""
    for attempt in range(retry_count):
        try:
            position = calculate_try_again_position()
            
            if position is None:
                print("⚠️  无法计算按钮位置（游戏区域未设置）")
                return False
            
            click_x, click_y = position
            
            print(f"🖱️  点击 Try again 按钮（相对位置）: ({click_x}, {click_y})")
            
            # 移动鼠标到按钮位置
            pyautogui.moveTo(click_x, click_y, duration=0.2)
            time.sleep(0.1)
            
            # 点击
            pyautogui.click()
            time.sleep(2)  # 等待游戏重启
            
            # 简单验证：尝试识别新游戏的棋盘
            time.sleep(0.5)
            test_board = get_board_matrix_from_screenshot()
            if test_board:
                # 检查是否是新游戏（通常只有2-3个方块）
                non_zero_count = sum(1 for row in test_board for cell in row if cell > 0)
                if non_zero_count <= 3:
                    print("✓ 游戏成功重启（相对位置点击）")
                    return True
            
            print(f"⚠️  点击可能失败，重试 {attempt + 1}/{retry_count}")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"相对位置点击失败 (尝试 {attempt + 1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(0.5)
                continue
            else:
                return False
    
    return False


def click_try_again_button(button_location=None, retry_count=3):
    """点击 Try again 按钮 - 优化版（优先使用相对位置）
    
    Args:
        button_location: 按钮位置（None 则自动检测）
        retry_count: 重试次数
    """
    # 策略1：优先使用相对位置点击（最快最准）
    if USE_RELATIVE_CLICK:
        if DEBUG_MODE:
            print("尝试使用相对位置点击...")
        if click_try_again_by_position(retry_count=2):
            return True
        else:
            print("⚠️  相对位置点击失败，尝试图像识别...")
    
    # 策略2：图像识别点击（fallback）
    for attempt in range(retry_count):
        try:
            if button_location is None:
                button_location = detect_try_again_button()
            
            if button_location:
                # 计算按钮中心位置
                center_x = button_location.left + button_location.width // 2
                center_y = button_location.top + button_location.height // 2
                
                print(f"🖱️  点击 Try again 按钮（图像识别）: ({center_x}, {center_y})")
                
                # 移动鼠标到按钮位置
                pyautogui.moveTo(center_x, center_y, duration=0.2)
                time.sleep(0.1)
                
                # 点击
                pyautogui.click()
                time.sleep(1.5)  # 等待游戏重启动画
                
                # 验证点击是否成功
                time.sleep(0.5)
                if not detect_try_again_button():
                    print("✓ 游戏成功重启（图像识别）")
                    return True
                else:
                    print(f"⚠️  按钮仍然存在，重试 {attempt + 1}/{retry_count}")
                    button_location = None
                    continue
            else:
                if attempt < retry_count - 1:
                    print(f"未找到 Try again 按钮，重试 {attempt + 1}/{retry_count}...")
                    time.sleep(0.5)
                    button_location = None
                    continue
                else:
                    print("❌ 未找到 Try again 按钮")
                    return False
                    
        except Exception as e:
            print(f"点击 Try again 按钮失败 (尝试 {attempt + 1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(0.5)
                button_location = None
                continue
            else:
                return False
    
    return False


def is_game_over_from_screenshot():
    """从截图中检测游戏是否结束 - 优化版"""
    # 检测 Try again 按钮（使用缓存避免重复检测）
    button = detect_try_again_button()
    return button is not None


# 完全复制原版的AI算法函数
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def reverse_rows(matrix):
    return [row[::-1] for row in matrix]


def compress(row):
    new_row = [i for i in row if i != 0]
    new_row.extend([0] * (4 - len(new_row)))
    return new_row


def merge(row):
    new_row = list(row)
    for i in range(3):
        if new_row[i] > 0 and new_row[i] == new_row[i + 1]:
            new_row[i] *= 2
            new_row[i + 1] = 0
    return new_row


def move_left(matrix):
    new_matrix = []
    for row in matrix:
        new_matrix.append(compress(merge(compress(row))))
    return new_matrix


def move_right(matrix):
    return reverse_rows(move_left(reverse_rows(matrix)))


def move_up(matrix):
    return transpose(move_left(transpose(matrix)))


def move_down(matrix):
    return transpose(move_right(transpose(matrix)))


SIMULATE_MAP = {
    "left": move_left,
    "right": move_right,
    "up": move_up,
    "down": move_down
}


def simulate_move(matrix, move_key):
    if move_key not in SIMULATE_MAP:
        return matrix, False
    board_copy = copy.deepcopy(matrix)
    new_matrix = SIMULATE_MAP[move_key](board_copy)
    return new_matrix, (new_matrix != matrix)


def get_empty_cells(matrix):
    """返回所有空格的 (row, col) 列表"""
    return [(r, c) for r in range(4) for c in range(4) if matrix[r][c] == 0]


def calculate_monotonicity(matrix):
    """
    增强的单调性计算（参考ovolve项目）
    检查每行每列是否保持单调递增或递减
    """
    monotonicity = 0.0
    
    # 检查所有行
    for r in range(4):
        # 尝试递增和递减两个方向
        increasing = 0
        decreasing = 0
        for c in range(3):
            if matrix[r][c] > 0 and matrix[r][c + 1] > 0:
                if matrix[r][c] < matrix[r][c + 1]:
                    increasing += 1
                elif matrix[r][c] > matrix[r][c + 1]:
                    decreasing += 1
        # 取更好的方向
        monotonicity += max(increasing, decreasing)
    
    # 检查所有列
    for c in range(4):
        increasing = 0
        decreasing = 0
        for r in range(3):
            if matrix[r][c] > 0 and matrix[r + 1][c] > 0:
                if matrix[r][c] < matrix[r + 1][c]:
                    increasing += 1
                elif matrix[r][c] > matrix[r + 1][c]:
                    decreasing += 1
        monotonicity += max(increasing, decreasing)
    
    return monotonicity


def calculate_merge_potential(matrix):
    """
    合并潜力（用户强调：相同数字必须合并！）
    两个16或32在同一行/列，优先合并
    """
    merge_score = 0
    
    # 水平相邻相同（用户说：应该左或下合成）
    for r in range(4):
        for c in range(3):
            if matrix[r][c] > 0 and matrix[r][c] == matrix[r][c + 1]:
                value = matrix[r][c]
                # 相同值越大，奖励越高（指数级）
                if value >= 128:
                    merge_score += value * 5  # 大数合并极其重要
                elif value >= 32:
                    merge_score += value * 4  # 中等数合并很重要
                else:
                    merge_score += value * 2  # 小数合并也重要
    
    # 垂直相邻相同
    for r in range(3):
        for c in range(4):
            if matrix[r][c] > 0 and matrix[r][c] == matrix[r + 1][c]:
                value = matrix[r][c]
                if value >= 128:
                    merge_score += value * 5
                elif value >= 32:
                    merge_score += value * 4
                else:
                    merge_score += value * 2
    
    return merge_score


def calculate_adjacent_similar(matrix):
    """
    相邻相似值奖励（用户需求：64旁边应该是32，128旁边是64）
    鼓励相似数字靠近，方便未来合并
    """
    similar_score = 0
    
    # 检查水平相邻
    for r in range(4):
        for c in range(3):
            v1, v2 = matrix[r][c], matrix[r][c + 1]
            if v1 > 0 and v2 > 0 and v1 != v2:
                # 检查是否是2倍关系（64和32，128和64）
                if v1 == v2 * 2 or v2 == v1 * 2:
                    similar_score += min(v1, v2)  # 奖励较小的那个值
    
    # 检查垂直相邻
    for r in range(3):
        for c in range(4):
            v1, v2 = matrix[r][c], matrix[r + 1][c]
            if v1 > 0 and v2 > 0 and v1 != v2:
                if v1 == v2 * 2 or v2 == v1 * 2:
                    similar_score += min(v1, v2)
    
    return similar_score


def evaluate_board(matrix, empty_cells_count):
    """ 
    评估函数（用户优先级）
    1. 相同数字合并（最高优先级！）
    2. 空格最大化
    3. 相似数字靠近（64旁32）
    4. 左下角策略
    """
    if empty_cells_count == 0:
        can_move = False
        for move_key in ALL_MOVES:
            _, changed = simulate_move(matrix, move_key)
            if changed:
                can_move = True
                break
        if not can_move:
            return -float('inf')

    # 使用当前策略的权重
    empty_weight = get_empty_cell_weight()
    merge_weight = get_merge_potential_weight()
    adjacent_weight = get_adjacent_similar_weight()

    # 1. 合并潜力（用户强调：两个16/32要合并！）
    merge_potential = calculate_merge_potential(matrix)
    
    # 2. 空格得分（生存基础）
    empty_score = empty_cells_count ** 2 * empty_weight

    # 3. 相邻相似值（用户需求：64旁32，128旁64）
    adjacent_similar = calculate_adjacent_similar(matrix)
    
    # 4. 位置权重（强化角落策略）
    # 根据当前策略选择权重矩阵
    weight_matrix = WEIGHT_MATRICES.get(CURRENT_STRATEGY.corner, WEIGHT_MATRIX)
    grid_score = 0.0
    for r in range(4):
        for c in range(4):
            value = matrix[r][c]
            if value > 0:
                grid_score += value * weight_matrix[r][c]

    # 5. 最大值必须在指定角落
    max_value = max(max(row) for row in matrix)
    corner_bonus = 0
    corner_pos = {"左下": (3, 0), "右下": (3, 3), "左上": (0, 0), "右上": (0, 3)}
    target_pos = corner_pos.get(CURRENT_STRATEGY.corner, (3, 0))
    
    if matrix[target_pos[0]][target_pos[1]] == max_value:
        corner_bonus = max_value * 1500  # 提高奖励
    else:
        corner_bonus = -max_value * 800  # 加重惩罚

    # 6. 单调性（简化）
    monotonicity_score = calculate_monotonicity(matrix)

    # 最终得分（突出合并和相似靠近）
    final_score = (
        merge_potential * merge_weight +                 # 动态权重
        empty_score +                                    # 动态权重
        adjacent_similar * adjacent_weight +             # 动态权重
        grid_score +                                     # 位置权重
        corner_bonus +                                   # 角落奖励
        monotonicity_score * MONOTONICITY_WEIGHT * 100   # 单调性
    )
    
    return final_score


def get_search_depth(empty_count):
    """
    优化的搜索深度（平衡速度和质量）
    nneonneo能在1秒内响应，说明深度不需要太深
    关键是评估函数的质量，不是搜索深度
    """
    if empty_count >= 10:
        return 2    # 初期：极浅，快速决策
    elif empty_count >= 6:
        return 3    # 早中期：浅搜索
    elif empty_count >= 3:
        return 3    # 中后期：标准深度
    else:
        return 3    # 危险期：深搜索


def search(matrix, depth, is_player_turn):
    """ 递归搜索: Expectimax """
    board_tuple = tuple(map(tuple, matrix))
    cache_key = (board_tuple, depth, is_player_turn)
    if cache_key in CACHE:
        return CACHE[cache_key]

    empty_cells = get_empty_cells(matrix)

    if depth == 0:
        score = evaluate_board(matrix, len(empty_cells))
        CACHE[cache_key] = score
        return score

    if is_player_turn:
        best_score = -float('inf')
        has_valid_move = False
        for move_key in ALL_MOVES:
            new_matrix, changed = simulate_move(matrix, move_key)
            if changed:
                has_valid_move = True
                score = search(new_matrix, depth, False)
                best_score = max(best_score, score)

        if not has_valid_move:
            best_score = -float('inf')

        CACHE[cache_key] = best_score
        return best_score

    else:
        # 随机节点：计算期望值
        num_empty = len(empty_cells)
        if num_empty == 0:
            score = search(matrix, depth - 1, True)
            CACHE[cache_key] = score
            return score

        expected_score = 0.0
        
        # 优化的采样策略（速度优先，参考nneonneo）
        # 空格多时大胆采样，空格少时才全检查
        if num_empty > 8:
            # 空格很多：只采样3-4个最重要位置（速度优先）
            weighted_cells = [(WEIGHT_MATRIX[r][c], r, c) for r, c in empty_cells]
            weighted_cells.sort(reverse=True)
            sample_cells = [(r, c) for _, r, c in weighted_cells[:3]]
        elif num_empty > 4:
            # 空格中等：采样一半位置
            weighted_cells = [(WEIGHT_MATRIX[r][c], r, c) for r, c in empty_cells]
            weighted_cells.sort(reverse=True)
            sample_size = max(4, num_empty // 2)
            sample_cells = [(r, c) for _, r, c in weighted_cells[:sample_size]]
        else:
            # 空格少时：检查所有位置（精确）
            sample_cells = empty_cells

        # 正确的概率分配
        total_cells = len(sample_cells)
        prob_2 = 0.9 / total_cells  # 90%概率出现2
        prob_4 = 0.1 / total_cells  # 10%概率出现4

        # 计算期望值
        for r, c in sample_cells:
            # 尝试放置2
            matrix[r][c] = 2
            expected_score += prob_2 * search(matrix, depth - 1, True)

            # 尝试放置4
            matrix[r][c] = 4
            expected_score += prob_4 * search(matrix, depth - 1, True)

            # 恢复空格
            matrix[r][c] = 0

        CACHE[cache_key] = expected_score
        return expected_score


def find_best_move_ai(matrix, return_thinking=False):
    """
    AI决策（用户优先级：左下角策略，优先合并）
    
    Args:
        matrix: 当前棋盘
        return_thinking: 是否返回思考过程
    
    Returns:
        如果 return_thinking=False: 返回最佳移动
        如果 return_thinking=True: 返回 (最佳移动, 思考信息字典)
    """
    global CACHE
    CACHE.clear()

    empty_count = len(get_empty_cells(matrix))
    current_depth = get_search_depth(empty_count)
    
    thinking_info = {
        'candidates': [],
        'best_move': None,
        'reason': '',
        'board_analysis': {
            'empty_cells': empty_count,
            'merge_potential': calculate_merge_potential(matrix),
            'monotonicity': calculate_monotonicity(matrix)
        }
    }

    # 初期：快速决策，强制左下优先
    if empty_count > 10:
        # down和left优先级最高
        for move in ["down", "left", "right", "up"]:
            _, changed = simulate_move(matrix, move)
            if changed:
                thinking_info['best_move'] = move
                thinking_info['reason'] = f"初期快速决策，空格充足({empty_count}个)"
                if return_thinking:
                    return move, thinking_info
                return move
        if return_thinking:
            return None, thinking_info
        return None

    # 中后期：Expectimax + 左下角优先级加成
    move_scores = {}
    
    for move_key in ALL_MOVES:
        new_matrix, changed = simulate_move(matrix, move_key)
        if changed:
            base_score = search(new_matrix, current_depth, False)
            # 强化左下优先级
            priority_bonus = FIXED_PRIORITY.get(move_key, 0)
            final_score = base_score + priority_bonus
            move_scores[move_key] = final_score
            thinking_info['candidates'].append((move_key, final_score))

    if not move_scores:
        thinking_info['reason'] = "无可用移动"
        if return_thinking:
            return None, thinking_info
        return None

    best_move = max(move_scores.items(), key=lambda x: x[1])[0]
    thinking_info['best_move'] = best_move
    
    # 分析选择理由
    best_score = move_scores[best_move]
    sorted_moves = sorted(move_scores.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_moves) > 1:
        second_best = sorted_moves[1]
        score_diff = best_score - second_best[1]
        if score_diff > 100000:
            thinking_info['reason'] = f"明显优于其他选择（优势{score_diff:.0f}分）"
        elif empty_count < 3:
            thinking_info['reason'] = f"危险局面（仅{empty_count}个空格），谨慎选择"
        else:
            thinking_info['reason'] = f"综合评估最优（深度{current_depth}）"
    else:
        thinking_info['reason'] = "唯一可行移动"
    
    if DEBUG_MODE:
        print(f"评分: {move_scores}, 选择: {best_move}")

    if return_thinking:
        return best_move, thinking_info
    return best_move


def send_key(key):
    """发送键盘按键"""
    try:
        keyboard.press_and_release(key)
        time.sleep(ANIMATION_DELAY)
        return True
    except Exception as e:
        print(f"发送按键失败: {e}")
        return False


def print_matrix(matrix, score, move_str="", show_thinking=False, thinking_info=None):
    """打印游戏棋盘
    
    Args:
        matrix: 游戏棋盘
        score: 当前分数
        move_str: 移动方向
        show_thinking: 是否显示AI思考过程
        thinking_info: AI思考信息字典
    """
    print("\033c", end="")
    header = f" SCORE: {score:<8} | MOVE: {move_str:<6} "
    width = 29
    print("=" * width)
    print(f"|{header:^{width - 2}}|")
    print("=" * width)
    max_val = 0
    for row in matrix:
        print("|", end="")
        for cell in row:
            if cell > max_val:
                max_val = cell
            cell_str = str(cell) if cell > 0 else '.'
            print(f"{cell_str:^6}|", end="")
        print("\n" + "-" * width)
    print(f"| MAX TILE: {max_val:<15} |")
    print("=" * width)
    
    # 显示AI思考过程
    if show_thinking and thinking_info:
        print("\n" + "="*50)
        print("🤔 AI 思考过程")
        print("="*50)
        
        if 'candidates' in thinking_info and thinking_info['candidates']:
            print("\n候选移动评分:")
            
            # 安全计算最大分数
            max_score = max((abs(s) for _, s in thinking_info['candidates']), default=1)
            if max_score == 0 or not math.isfinite(max_score):
                max_score = 1
            
            for move, score_val in thinking_info['candidates']:
                # 安全计算进度条长度
                if math.isfinite(score_val) and max_score > 0:
                    bar_length = int(abs(score_val) / max_score * 20)
                    bar_length = max(0, min(20, bar_length))  # 限制在0-20之间
                else:
                    bar_length = 0
                
                bar = "█" * bar_length
                
                # 安全显示分数
                if math.isfinite(score_val):
                    print(f"  {move:>5}: {score_val:>12.0f} {bar}")
                else:
                    print(f"  {move:>5}: {'N/A':>12} {bar}")
        
        if 'best_move' in thinking_info:
            print(f"\n✓ 选择: {thinking_info['best_move']}")
        
        if 'reason' in thinking_info:
            print(f"💭 理由: {thinking_info['reason']}")
        
        if 'board_analysis' in thinking_info:
            analysis = thinking_info['board_analysis']
            print(f"\n📊 棋盘分析:")
            print(f"  空格数: {analysis.get('empty_cells', 0)}")
            print(f"  可合并: {analysis.get('merge_potential', 0)}")
            print(f"  单调性: {analysis.get('monotonicity', 0):.1f}")
        
        print("="*50)


def play_game(strategy: StrategyParams = None, auto_restart=False, skip_region_detect=False):
    """主游戏循环
    
    Args:
        strategy: 使用的策略
        auto_restart: 是否自动重启游戏
        skip_region_detect: 是否跳过区域检测（训练模式下使用）
    """
    global CACHE, CURRENT_STRATEGY, FIXED_PRIORITY, GAME_REGION, SCORE_TRACKER

    # 设置当前策略
    if strategy:
        CURRENT_STRATEGY = strategy
        FIXED_PRIORITY = strategy.priority
    
    print("2048 视觉识别版本启动（无OCR依赖）...")
    print(f"当前策略: {CURRENT_STRATEGY.name}")
    print(f"自动重启: {'启用' if auto_restart else '禁用'}")
    print("请确保2048游戏窗口可见且未被遮挡")
    
    # 检查 Try Again 按钮配置
    if auto_restart:
        if USE_RELATIVE_CLICK:
            print(f"✓ Try Again 点击模式: 相对位置（基准 {TRY_AGAIN_BASE_SIZE}x{TRY_AGAIN_BASE_SIZE}）")
            print(f"  偏移: X={TRY_AGAIN_RELATIVE_X}, Y={TRY_AGAIN_RELATIVE_Y}")
        elif not os.path.exists(TRY_AGAIN_BUTTON):
            print(f"\n⚠️  警告: 未找到 Try Again 按钮图片: {TRY_AGAIN_BUTTON}")
            print("自动重启功能可能无法正常工作")

    # 创建截图目录
    create_screenshot_dir()

    # 检测游戏区域（首次或手动模式）
    if not skip_region_detect or GAME_REGION is None:
        if not detect_game_region():
            print("无法检测游戏区域，退出")
            return False
    else:
        print(f"使用已保存的游戏区域: {GAME_REGION}")

    print("游戏区域检测完成，AI 开始运行...")
    time.sleep(0.5)

    move_count = 0
    last_board_tuple = None
    last_board_matrix = None
    stuck_count = 0
    game_completed = False
    
    # 重置分数追踪器
    SCORE_TRACKER.reset()
    
    # 开始记录游戏
    if GAME_RECORDER:
        GAME_RECORDER.start_game(CURRENT_STRATEGY)

    try:
        while True:
            # 获取当前棋盘状态
            current_board = get_board_matrix_from_screenshot()
            if current_board is None:
                print("无法获取棋盘状态，等待...")
                time.sleep(1)
                continue

            current_board_tuple = tuple(map(tuple, current_board))
            if current_board_tuple == last_board_tuple:
                stuck_count += 1
            else:
                stuck_count = 0
                last_board_tuple = current_board_tuple

            # 优化：只在必要时检查游戏结束状态
            if stuck_count > 6:
                if DEBUG_MODE:
                    print("⚠️  检测到棋盘长时间无变化，强制检查Game Over状态...")
                time.sleep(0.5)

            # 检测游戏结束（优化：智能间隔检测）
            try_again_btn = None
            should_check_button = (
                stuck_count > 3 or  # 棋盘无变化
                move_count % BUTTON_CHECK_INTERVAL == 0 or  # 定期检查
                move_count < 5  # 初期更频繁检查（防止错误重启）
            )
            
            if should_check_button:
                try_again_btn = detect_try_again_button()
            
            if try_again_btn or stuck_count > 10:
                final_score = SCORE_TRACKER.get_total_score()
                print_matrix(current_board, final_score, "END")
                print(f"\n>>> GAME OVER! <<< | 最终分数: {final_score} | 总步数: {move_count}")
                
                # 记录游戏结束
                if GAME_RECORDER:
                    GAME_RECORDER.end_game(current_board, final_score, move_count)
                    # 只在非自动重启模式或达到总结频率时显示统计
                    if not auto_restart or len(GAME_RECORDER.history["games"]) % SUMMARY_FREQUENCY == 0:
                        GAME_RECORDER.print_statistics()
                
                game_completed = True
                
                # 处理自动重启
                if auto_restart and try_again_btn:
                    print("\n🔄 自动重启游戏...")
                    time.sleep(1)
                    if click_try_again_button(try_again_btn):
                        print("✓ 游戏已重启，等待新游戏加载...")
                        time.sleep(2)
                        return True  # 返回True表示需要继续训练
                    else:
                        print("❌ 重启失败")
                        return False
                else:
                    # 非自动重启模式，结束游戏
                    return False

            # AI计算最佳移动
            move = find_best_move_ai(current_board)
            move_str = KEY_MAP.get(move, 'WAIT')
            
            # 显示当前状态（使用追踪的分数）
            current_score = SCORE_TRACKER.get_total_score()
            print_matrix(current_board, current_score, move_str)
            print(f"步数: {move_count + 1:<4} | 策略: {CURRENT_STRATEGY.name}")

            if move is None:
                print("AI 认为无路可走, 等待游戏结束判定...")
                time.sleep(1)
                stuck_count += 2
                continue

            # 发送按键
            if send_key(move):
                move_count += 1
                
                # 等待游戏动画完成后获取新棋盘
                time.sleep(ANIMATION_DELAY)
                new_board = get_board_matrix_from_screenshot()
                
                if new_board and last_board_matrix:
                    # 更新分数追踪器
                    score_gain = SCORE_TRACKER.update(last_board_matrix, new_board)
                    if score_gain > 0 and not DEBUG_MODE:
                        print(f"💰 +{score_gain} 分")
                
                # 更新游戏记录
                current_score = SCORE_TRACKER.get_total_score()
                if GAME_RECORDER:
                    GAME_RECORDER.update_game(current_board, current_score, move_count)
                
                # 保存当前棋盘用于下次计算
                last_board_matrix = [row[:] for row in current_board]
            else:
                print("按键发送失败，重试...")
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n用户中断脚本。")
        if GAME_RECORDER and not game_completed:
            try:
                current_board = get_board_matrix_from_screenshot()
                final_score = get_score_from_screenshot()
                GAME_RECORDER.end_game(current_board, final_score, move_count)
            except:
                pass
        return False
    except Exception as e:
        import traceback
        print(f"\n运行中发生未知错误: {e}")
        traceback.print_exc()
        if GAME_RECORDER and not game_completed:
            try:
                current_board = get_board_matrix_from_screenshot()
                final_score = get_score_from_screenshot()
                GAME_RECORDER.end_game(current_board, final_score, move_count)
            except:
                pass
        return False
    finally:
        if not game_completed:
            final_score = get_score_from_screenshot()
            print(f"\n脚本结束。当前/最终分数: {final_score}")


def main():
    """主函数"""
    # 检查依赖
    try:
        import cv2
        import pyautogui
        import keyboard
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装以下依赖:")
        print("pip install opencv-python pyautogui keyboard pillow numpy")
        sys.exit(1)
    
    # 显示欢迎信息
    print("\n" + "="*60)
    print("🎮 2048 AI 自动玩家 - 机器学习版")
    print("="*60)
    
    # 首先选择运行模式
    print("\n请选择运行模式:")
    print("  1. 🧘 冥想模式 - 纯AI模拟训练（不需要游戏界面，速度极快）")
    print("  2. ⚔️  实战模式 - 基于图像识别控制真实游戏")
    print("  3. 📊 查看训练数据统计")
    
    mode_choice = input("\n请选择模式 (1-3) [默认: 2]: ").strip()
    
    if mode_choice == "1":
        # 冥想模式 - 纯模拟训练
        meditation_mode()
        return
    elif mode_choice == "3":
        # 查看统计
        if GAME_RECORDER and GAME_RECORDER.history["games"]:
            GAME_RECORDER.print_statistics()
            print_strategy_details()
        else:
            print("\n暂无训练数据")
        return
    
    # 实战模式（原有逻辑）
    print("\n" + "="*60)
    print("⚔️  实战模式 - 图像识别控制")
    print("="*60)
    
    # 询问是否需要AI自动训练
    print("\n🤖 是否需要AI自动训练？")
    print("   自动训练模式会在游戏结束后自动点击 Try Again 继续训练")
    auto_train = input("\n请输入 Y(是) 或 N(否) [默认: N]: ").strip().upper()
    
    if auto_train == 'Y':
        # 进入自动训练模式
        print("\n请输入训练次数:")
        print("  - 输入具体数字 (如: 10, 30, 100)")
        print("  - 输入 0 表示无限循环训练（按 Ctrl+C 停止）")
        
        loop_count_input = input("\n训练次数 [默认: 10]: ").strip()
        
        if loop_count_input == '':
            loop_count = 10
        elif loop_count_input == '0':
            loop_count = None  # 无限循环
        else:
            try:
                loop_count = int(loop_count_input)
                if loop_count < 0:
                    print("⚠️  无效输入，使用默认值 10")
                    loop_count = 10
            except ValueError:
                print("⚠️  无效输入，使用默认值 10")
                loop_count = 10
        
        # 选择学习算法
        print("\n选择学习算法:")
        print("  a. 自动选择（推荐）")
        print("  b. 贝叶斯优化")
        print("  c. 遗传算法")
        print("  d. 随机探索")
        
        algo_choice = input("\n请选择 [默认: a]: ").strip().lower()
        algo_map = {"a": "auto", "b": "bayesian", "c": "genetic", "d": "random", "": "auto"}
        algorithm = algo_map.get(algo_choice, "auto")
        
        # 显示训练配置
        print("\n" + "="*60)
        print("📋 训练配置")
        print("="*60)
        if loop_count is None:
            print(f"训练次数: 无限循环 ♾️")
        else:
            print(f"训练次数: {loop_count} 局")
        print(f"学习算法: {algorithm}")
        print(f"总结频率: 每 {SUMMARY_FREQUENCY} 局")
        print("="*60)
        
        input("\n按回车键开始训练...")
        
        # 启动训练模式
        train_mode(max_games=loop_count, algorithm=algorithm)
        return
    
    # 非自动训练模式 - 原有逻辑
    # 如果有历史数据，显示统计信息
    if GAME_RECORDER and GAME_RECORDER.history["games"]:
        GAME_RECORDER.print_statistics()
        
        # 询问是否使用最佳策略
        if ML_AUTO_OPTIMIZE and GAME_RECORDER.get_best_strategy():
            print("\n发现历史最佳策略！")
            best_strategy = GAME_RECORDER.get_best_strategy()
            print(f"策略名称: {best_strategy.name}")
            print("\n选项:")
            print("1. 使用历史最佳策略（单局）")
            print("2. 使用默认策略（单局）")
            print("3. 测试多个策略（手动重启）")
            print("4. 查看策略详情")
            
            choice = input("\n请选择 (1-4, 直接回车使用最佳策略): ").strip()
            
            if choice == "3":
                # 多策略测试模式
                test_multiple_strategies()
                return
            elif choice == "4":
                # 查看策略详情
                print_strategy_details()
                return
            elif choice == "2":
                # 使用默认策略
                play_game()
            else:
                # 使用最佳策略（默认选项）
                play_game(best_strategy)
        else:
            play_game()
    else:
        # 首次运行
        print("\n这是首次运行，将使用默认策略")
        print("游戏结束后，数据将保存到 game_history.json")
        print("\n提示: 可以运行多局游戏来让AI自动学习最佳策略！\n")
        play_game()
    
    print("\n--- ALL DONE ---")


def train_mode(max_games=None, algorithm='auto'):
    """智能训练模式 - 自动运行多局游戏并学习
    
    Args:
        max_games: 最大训练局数（None表示无限）
        algorithm: 使用的学习算法 ('auto', 'bayesian', 'genetic', 'random')
    """
    print("\n" + "="*60)
    print("🤖 智能训练模式")
    print("="*60)
    print(f"算法: {algorithm}")
    print(f"总结频率: 每 {SUMMARY_FREQUENCY} 局")
    if max_games:
        print(f"目标局数: {max_games}")
    else:
        print("目标局数: 无限（按 Ctrl+C 停止）")
    print("="*60)
    
    # 首次设置游戏区域
    print("\n首次设置游戏区域（之后会自动重用）...")
    if not detect_game_region():
        print("无法检测游戏区域，退出")
        return
    
    game_count = 0
    start_time = time.time()
    
    try:
        while True:
            game_count += 1
            
            # 使用自适应学习器选择策略
            if ADAPTIVE_LEARNER and game_count > 1:
                strategy = ADAPTIVE_LEARNER.suggest_next_strategy(mode=algorithm)
                print(f"\n🎯 第 {game_count} 局 - 策略: {strategy.name}")
            else:
                strategy = CURRENT_STRATEGY
                print(f"\n🎯 第 {game_count} 局 - 使用默认策略")
            
            # 运行游戏（自动重启）
            should_continue = play_game(
                strategy=strategy,
                auto_restart=True,
                skip_region_detect=True
            )
            
            # 检查是否继续
            if not should_continue:
                print("\n❌ 游戏无法继续，训练终止")
                break
            
            # 检查是否达到目标局数
            if max_games and game_count >= max_games:
                print(f"\n✅ 已完成 {max_games} 局训练")
                break
            
            # 短暂延迟
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断训练")
    finally:
        # 训练总结
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("🏁 训练完成")
        print("="*60)
        print(f"完成局数: {game_count}")
        print(f"总用时: {elapsed_time/60:.1f} 分钟")
        print(f"平均每局: {elapsed_time/game_count:.1f} 秒")
        
        if GAME_RECORDER:
            print("\n最终统计:")
            GAME_RECORDER.print_statistics()
            
            # 显示最佳策略
            if GAME_RECORDER.history["best_strategy"]:
                print("\n🏆 最佳策略:")
                best = GAME_RECORDER.history["best_strategy"]
                print(f"  名称: {best['name']}")
                print(f"  角落: {best['corner']}")
                print(f"  优先级: {best['priority']}")


def test_multiple_strategies():
    """测试多个策略"""
    print("\n" + "="*60)
    print("🧪 多策略测试模式")
    print("="*60)
    
    # 生成测试策略
    strategies = []
    
    print("\n选择测试模式:")
    print("1. 测试四个角落策略（4局）")
    print("2. 测试权重变体（5局）")
    print("3. 全面测试（9局）")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == "1":
        strategies = StrategyGenerator.generate_corner_strategies()
    elif choice == "2":
        strategies = StrategyGenerator.generate_weight_variants(CURRENT_STRATEGY)
    elif choice == "3":
        strategies.extend(StrategyGenerator.generate_corner_strategies())
        strategies.extend(StrategyGenerator.generate_weight_variants(CURRENT_STRATEGY, count=3))
    else:
        print("无效选择，返回主菜单")
        return
    
    print(f"\n将测试 {len(strategies)} 个策略")
    print("每局游戏结束后请手动重启游戏，然后按任意键继续...")
    
    for i, strategy in enumerate(strategies, 1):
        print("\n" + "="*60)
        print(f"🎯 测试策略 {i}/{len(strategies)}: {strategy.name}")
        print("="*60)
        input("准备好后按回车开始...")
        
        play_game(strategy)
        
        if i < len(strategies):
            print("\n请重启游戏，准备下一局测试...")
            input("按回车继续...")
    
    # 显示最终统计
    print("\n" + "="*60)
    print("🏆 测试完成！最终统计:")
    print("="*60)
    if GAME_RECORDER:
        GAME_RECORDER.print_statistics()


def print_strategy_details():
    """打印所有策略详情"""
    if not GAME_RECORDER or not GAME_RECORDER.history["games"]:
        print("暂无历史数据")
        return
    
    print("\n" + "="*60)
    print("📋 所有游戏记录")
    print("="*60)
    
    # 按最大方块排序
    games = sorted(
        GAME_RECORDER.history["games"],
        key=lambda x: (x["max_tile"], x["final_score"]),
        reverse=True
    )
    
    for i, game in enumerate(games, 1):
        print(f"\n游戏 #{i}")
        print(f"  策略: {game['strategy']['name']}")
        print(f"  最大方块: {game['max_tile']}")
        print(f"  分数: {game['final_score']}")
        print(f"  步数: {game['moves_count']}")
        print(f"  时长: {game['duration']}秒")
        print(f"  达到2048: {'✓' if game['success'] else '✗'}")
        print(f"  时间: {game.get('end_time', 'N/A')}")


# ==================== 冥想模式（纯模拟训练） ====================
class Game2048Simulator:
    """2048游戏模拟器 - 用于冥想模式"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置游戏"""
        self.matrix = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.move_count = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.matrix
    
    def add_random_tile(self):
        """添加随机方块（90% 概率是2，10%概率是4）"""
        empty_cells = [(r, c) for r in range(4) for c in range(4) if self.matrix[r][c] == 0]
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.matrix[r][c] = 2 if random.random() < 0.9 else 4
            return True
        return False
    
    def move(self, direction):
        """执行移动"""
        old_matrix = [row[:] for row in self.matrix]
        old_score = self.score
        
        # 执行移动
        if direction == "left":
            self.matrix, score_gain = self._move_left(self.matrix)
        elif direction == "right":
            self.matrix, score_gain = self._move_right(self.matrix)
        elif direction == "up":
            self.matrix, score_gain = self._move_up(self.matrix)
        elif direction == "down":
            self.matrix, score_gain = self._move_down(self.matrix)
        else:
            return False
        
        # 检查是否有变化
        if self.matrix == old_matrix:
            return False
        
        self.score += score_gain
        self.move_count += 1
        self.add_random_tile()
        return True
    
    def _move_left(self, matrix):
        """向左移动"""
        new_matrix = []
        score_gain = 0
        for row in matrix:
            new_row, gain = self._compress_and_merge(row)
            new_matrix.append(new_row)
            score_gain += gain
        return new_matrix, score_gain
    
    def _move_right(self, matrix):
        """向右移动"""
        new_matrix = []
        score_gain = 0
        for row in matrix:
            reversed_row = row[::-1]
            new_row, gain = self._compress_and_merge(reversed_row)
            new_matrix.append(new_row[::-1])
            score_gain += gain
        return new_matrix, score_gain
    
    def _move_up(self, matrix):
        """向上移动"""
        transposed = transpose(matrix)
        moved, score_gain = self._move_left(transposed)
        return transpose(moved), score_gain
    
    def _move_down(self, matrix):
        """向下移动"""
        transposed = transpose(matrix)
        moved, score_gain = self._move_right(transposed)
        return transpose(moved), score_gain
    
    def _compress_and_merge(self, row):
        """压缩并合并一行"""
        # 移除0
        new_row = [x for x in row if x != 0]
        score_gain = 0
        
        # 合并相同的
        i = 0
        while i < len(new_row) - 1:
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                score_gain += new_row[i]
                new_row.pop(i + 1)
            i += 1
        
        # 补0
        new_row.extend([0] * (4 - len(new_row)))
        return new_row, score_gain
    
    def is_game_over(self):
        """检查游戏是否结束"""
        # 检查是否有空格
        for row in self.matrix:
            if 0 in row:
                return False
        
        # 检查是否可以合并
        for r in range(4):
            for c in range(4):
                val = self.matrix[r][c]
                # 检查右边
                if c < 3 and self.matrix[r][c + 1] == val:
                    return False
                # 检查下边
                if r < 3 and self.matrix[r + 1][c] == val:
                    return False
        
        return True
    
    def get_max_tile(self):
        """获取最大方块"""
        return max(max(row) for row in self.matrix)
    
    def get_state(self):
        """获取当前状态"""
        return {
            'matrix': [row[:] for row in self.matrix],
            'score': self.score,
            'move_count': self.move_count,
            'max_tile': self.get_max_tile()
        }


def meditation_mode():
    """冥想模式 - 纯AI模拟训练"""
    print("\n" + "="*60)
    print("🧘 冥想模式 - 纯AI模拟训练")
    print("="*60)
    print("在此模式下，AI将在虚拟环境中快速训练")
    print("无需真实游戏界面，速度极快！")
    print("="*60)
    
    # 询问训练次数
    print("\n请输入训练次数:")
    print("  - 输入具体数字 (如: 100, 500, 1000)")
    print("  - 输入 0 表示无限循环训练（按 Ctrl+C 停止）")
    
    loop_count_input = input("\n训练次数 [默认: 100]: ").strip()
    
    if loop_count_input == '':
        loop_count = 100
    elif loop_count_input == '0':
        loop_count = None  # 无限循环
    else:
        try:
            loop_count = int(loop_count_input)
            if loop_count < 0:
                print("⚠️  无效输入，使用默认值 100")
                loop_count = 100
        except ValueError:
            print("⚠️  无效输入，使用默认值 100")
            loop_count = 100
    
    # 选择显示模式
    print("\n选择显示模式:")
    print("  1. 静默训练 - 只显示统计（最快）")
    print("  2. 实时战况 - 显示棋盘变化")
    print("  3. 思考模式 - 显示AI决策过程（最详细）")
    
    display_choice = input("\n请选择 [默认: 1]: ").strip()
    show_realtime = display_choice in ['2', '3']
    thinking_mode = display_choice == '3'
    
    if show_realtime:
        print("\n💡 提示: 实时显示会降低训练速度")
        if thinking_mode:
            print("   思考模式将展示AI的决策分析过程")
    
    # 选择学习算法
    print("\n选择学习算法:")
    print("  a. 自动选择（推荐）")
    print("  b. 贝叶斯优化")
    print("  c. 遗传算法")
    print("  d. 随机探索")
    print("  e. 固定策略（左下角）")
    
    algo_choice = input("\n请选择 [默认: a]: ").strip().lower()
    algo_map = {"a": "auto", "b": "bayesian", "c": "genetic", "d": "random", "e": "fixed", "": "auto"}
    algorithm = algo_map.get(algo_choice, "auto")
    
    # 显示训练配置
    print("\n" + "="*60)
    print("📋 冥想训练配置")
    print("="*60)
    if loop_count is None:
        print(f"训练次数: 无限循环 ♾️")
    else:
        print(f"训练次数: {loop_count} 局")
    print(f"显示模式: {'静默训练' if not show_realtime else ('思考模式' if thinking_mode else '实时战况')}")
    print(f"学习算法: {algorithm}")
    print(f"总结频率: 每 {SUMMARY_FREQUENCY} 局")
    print("="*60)
    
    input("\n按回车键开始冥想训练...")
    
    # 开始训练
    game_count = 0
    start_time = time.time()
    
    try:
        while True:
            game_count += 1
            
            # 选择策略
            if algorithm == "fixed":
                strategy = CURRENT_STRATEGY
            elif ADAPTIVE_LEARNER and game_count > 1:
                strategy = ADAPTIVE_LEARNER.suggest_next_strategy(mode=algorithm)
            else:
                strategy = CURRENT_STRATEGY
            
            # 显示当前局数
            if show_realtime:
                print(f"\n{'='*60}")
                print(f"🎮 第 {game_count} 局开始 | 策略: {strategy.name}")
                print(f"{'='*60}")
            
            # 运行一局模拟游戏
            result = simulate_one_game(strategy, show_realtime=show_realtime, thinking_mode=thinking_mode)
            
            # 记录结果
            if GAME_RECORDER:
                GAME_RECORDER.start_game(strategy)
                GAME_RECORDER.end_game(
                    result['final_matrix'],
                    result['final_score'],
                    result['move_count']
                )
            
            # 显示进度（静默模式下每10局显示一次）
            if not show_realtime and game_count % 10 == 0:
                elapsed = time.time() - start_time
                speed = game_count / elapsed
                print(f"进度: {game_count} 局 | 速度: {speed:.1f} 局/秒 | "
                      f"最大方块: {result['max_tile']} | 分数: {result['final_score']}")
            
            # 检查是否完成
            if loop_count and game_count >= loop_count:
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断训练")
    finally:
        # 训练总结
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("🏁 冥想训练完成")
        print("="*60)
        print(f"完成局数: {game_count}")
        print(f"总用时: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
        print(f"平均每局: {elapsed_time/game_count:.2f} 秒")
        print(f"训练速度: {game_count/elapsed_time:.1f} 局/秒")
        
        if GAME_RECORDER:
            print("\n最终统计:")
            GAME_RECORDER.print_statistics()
            
            # 显示最佳策略
            if GAME_RECORDER.history["best_strategy"]:
                print("\n🏆 最佳策略:")
                best = GAME_RECORDER.history["best_strategy"]
                print(f"  名称: {best['name']}")
                print(f"  角落: {best['corner']}")
                print(f"  优先级: {best['priority']}")


def simulate_one_game(strategy: StrategyParams, show_realtime=False, thinking_mode=False):
    """模拟一局游戏（冥想模式）
    
    Args:
        strategy: 使用的策略
        show_realtime: 是否显示实时战况
        thinking_mode: 是否显示AI思考过程
    """
    global CACHE, CURRENT_STRATEGY, FIXED_PRIORITY
    
    # 设置当前策略
    CURRENT_STRATEGY = strategy
    FIXED_PRIORITY = strategy.priority
    
    # 创建模拟器
    simulator = Game2048Simulator()
    
    move_count = 0
    
    while not simulator.is_game_over():
        # AI决策
        if thinking_mode:
            best_move, thinking_info = find_best_move_ai(simulator.matrix, return_thinking=True)
        else:
            best_move = find_best_move_ai(simulator.matrix)
            thinking_info = None
        
        if best_move is None:
            break
        
        # 显示实时战况
        if show_realtime:
            move_str = KEY_MAP.get(best_move, 'WAIT')
            print_matrix(simulator.matrix, simulator.score, move_str, 
                        show_thinking=thinking_mode, thinking_info=thinking_info)
            print(f"步数: {move_count + 1} | 空格: {len(get_empty_cells(simulator.matrix))}")
            
            # 控制速度，让用户能看清
            time.sleep(0.05)  # 快速但可见
        
        # 执行移动
        success = simulator.move(best_move)
        if not success:
            break
        
        move_count += 1
    
    # 显示最终状态
    if show_realtime:
        print_matrix(simulator.matrix, simulator.score, "END")
        print(f"游戏结束 | 最大方块: {simulator.get_max_tile()} | 总分: {simulator.score}")
        time.sleep(1)  # 暂停1秒显示结果
    
    # 返回结果
    state = simulator.get_state()
    return {
        'final_matrix': state['matrix'],
        'final_score': state['score'],
        'move_count': state['move_count'],
        'max_tile': state['max_tile']
    }


if __name__ == "__main__":
    main()
