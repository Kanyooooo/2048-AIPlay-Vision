#!/usr/bin/env python3
"""
AI策略模块

包含：
- 策略参数类
- Expectimax 搜索算法
- 评估函数
- AI 决策引擎
"""

import copy
import math
from game_logic import simulate_move, get_empty_cells


# ==================== 全局配置 ====================

# 键盘映射
KEY_MAP = {
    "down": "↓ DOWN",
    "left": "← LEFT",
    "right": "→ RIGHT",
    "up": "↑ UP"
}

ALL_MOVES = ["down", "left", "right", "up"]

# 搜索缓存
CACHE = {}

# 评估函数权重
MONOTONICITY_WEIGHT = 1.0
SMOOTHNESS_WEIGHT = 0.05


# ==================== 策略参数类 ====================

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


# 当前使用的策略（全局变量）
CURRENT_STRATEGY = StrategyParams(name="左下角策略")
FIXED_PRIORITY = CURRENT_STRATEGY.priority


# ==================== 权重矩阵 ====================

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

# 默认使用左下角策略
WEIGHT_MATRIX = WEIGHT_MATRICES["左下"]


# ==================== 评估函数 ====================

def calculate_monotonicity(matrix):
    """增强的单调性计算 - 检查每行每列是否保持单调递增或递减"""
    monotonicity = 0.0
    
    # 检查所有行
    for r in range(4):
        increasing = 0
        decreasing = 0
        for c in range(3):
            if matrix[r][c] > 0 and matrix[r][c + 1] > 0:
                if matrix[r][c] < matrix[r][c + 1]:
                    increasing += 1
                elif matrix[r][c] > matrix[r][c + 1]:
                    decreasing += 1
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
    """合并潜力 - 相同数字必须合并"""
    merge_score = 0
    
    # 水平相邻相同
    for r in range(4):
        for c in range(3):
            if matrix[r][c] > 0 and matrix[r][c] == matrix[r][c + 1]:
                value = matrix[r][c]
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
    """相邻相似值奖励 - 鼓励相似数字靠近"""
    similar_score = 0
    
    # 检查水平相邻
    for r in range(4):
        for c in range(3):
            v1, v2 = matrix[r][c], matrix[r][c + 1]
            if v1 > 0 and v2 > 0 and v1 != v2:
                if v1 == v2 * 2 or v2 == v1 * 2:
                    similar_score += min(v1, v2)
    
    # 检查垂直相邻
    for r in range(3):
        for c in range(4):
            v1, v2 = matrix[r][c], matrix[r + 1][c]
            if v1 > 0 and v2 > 0 and v1 != v2:
                if v1 == v2 * 2 or v2 == v1 * 2:
                    similar_score += min(v1, v2)
    
    return similar_score


def evaluate_board(matrix, empty_cells_count):
    """评估函数 - 综合评价棋盘状态"""
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
    empty_weight = CURRENT_STRATEGY.empty_weight
    merge_weight = CURRENT_STRATEGY.merge_weight
    adjacent_weight = CURRENT_STRATEGY.adjacent_weight

    # 1. 合并潜力
    merge_potential = calculate_merge_potential(matrix)
    
    # 2. 空格得分
    empty_score = empty_cells_count ** 2 * empty_weight

    # 3. 相邻相似值
    adjacent_similar = calculate_adjacent_similar(matrix)
    
    # 4. 位置权重
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
        corner_bonus = max_value * 1500
    else:
        corner_bonus = -max_value * 800

    # 6. 单调性
    monotonicity_score = calculate_monotonicity(matrix)

    # 最终得分
    final_score = (
        merge_potential * merge_weight +
        empty_score +
        adjacent_similar * adjacent_weight +
        grid_score +
        corner_bonus +
        monotonicity_score * MONOTONICITY_WEIGHT * 100
    )
    
    return final_score


# ==================== 搜索算法 ====================

def get_search_depth(empty_count):
    """优化的搜索深度"""
    if empty_count >= 10:
        return 2
    elif empty_count >= 6:
        return 3
    elif empty_count >= 3:
        return 3
    else:
        return 3


def search(matrix, depth, is_player_turn):
    """Expectimax 搜索"""
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
        
        # 优化的采样策略
        if num_empty > 8:
            weighted_cells = [(WEIGHT_MATRIX[r][c], r, c) for r, c in empty_cells]
            weighted_cells.sort(reverse=True)
            sample_cells = [(r, c) for _, r, c in weighted_cells[:3]]
        elif num_empty > 4:
            weighted_cells = [(WEIGHT_MATRIX[r][c], r, c) for r, c in empty_cells]
            weighted_cells.sort(reverse=True)
            sample_size = max(4, num_empty // 2)
            sample_cells = [(r, c) for _, r, c in weighted_cells[:sample_size]]
        else:
            sample_cells = empty_cells

        # 正确的概率分配
        total_cells = len(sample_cells)
        prob_2 = 0.9 / total_cells
        prob_4 = 0.1 / total_cells

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


# ==================== AI 决策引擎 ====================

def find_best_move_ai(matrix, return_thinking=False):
    """AI决策引擎"""
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

    if return_thinking:
        return best_move, thinking_info
    return best_move


# ==================== 策略生成器 ====================

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


# ==================== 工具函数 ====================

def set_current_strategy(strategy: StrategyParams):
    """设置当前策略"""
    global CURRENT_STRATEGY, FIXED_PRIORITY
    CURRENT_STRATEGY = strategy
    FIXED_PRIORITY = strategy.priority


def print_matrix(matrix, score, move_str="", show_thinking=False, thinking_info=None):
    """打印游戏棋盘"""
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
                    bar_length = max(0, min(20, bar_length))
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

