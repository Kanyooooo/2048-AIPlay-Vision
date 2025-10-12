#!/usr/bin/env python3
"""
机器学习系统模块

包含：
- 游戏数据记录器
- 贝叶斯优化器
- 遗传算法
- 强化学习
- 自适应学习器
"""

import os
import json
import copy
import random
import math
from datetime import datetime
from ai_strategy import StrategyParams, StrategyGenerator


# ==================== 配置参数 ====================

ML_DATA_FILE = "game_history.json"
SUMMARY_FREQUENCY = 5  # 每N局总结一次


# ==================== 游戏数据记录器 ====================

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
            "success": False
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
        
        # 更新学习器（在外部处理，避免循环依赖）
        
        # 判断是否需要阶段性总结
        total_games = len(self.history["games"])
        if total_games % SUMMARY_FREQUENCY == 0:
            self.print_phase_summary()
        
        # 保存数据
        self.save_history()
        
        # 更新最佳策略
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


# ==================== 贝叶斯优化器 ====================

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
        params = {}
        for key, (min_val, max_val) in self.param_ranges.items():
            params[key] = random.randint(min_val, max_val)
        return params
    
    def _mutate_params(self, base_strategy, mutation_rate=0.2):
        """对参数进行变异"""
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


# ==================== 遗传算法 ====================

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


# ==================== 强化学习 ====================

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


# ==================== 自适应学习器 ====================

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
                mode = 'random'
            elif games_count < 20:
                mode = 'bayesian'
            else:
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

