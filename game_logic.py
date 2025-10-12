#!/usr/bin/env python3
"""
游戏逻辑模块

包含：
- 2048 游戏核心逻辑（移动、合并）
- 游戏模拟器（用于冥想模式）
- 基础工具函数
"""

import random
import copy


# ==================== 游戏核心逻辑 ====================

def transpose(matrix):
    """矩阵转置"""
    return [list(row) for row in zip(*matrix)]


def reverse_rows(matrix):
    """反转每一行"""
    return [row[::-1] for row in matrix]


def compress(row):
    """压缩行（移除0）"""
    new_row = [i for i in row if i != 0]
    new_row.extend([0] * (4 - len(new_row)))
    return new_row


def merge(row):
    """合并相同数字"""
    new_row = list(row)
    for i in range(3):
        if new_row[i] > 0 and new_row[i] == new_row[i + 1]:
            new_row[i] *= 2
            new_row[i + 1] = 0
    return new_row


def move_left(matrix):
    """向左移动"""
    new_matrix = []
    for row in matrix:
        new_matrix.append(compress(merge(compress(row))))
    return new_matrix


def move_right(matrix):
    """向右移动"""
    return reverse_rows(move_left(reverse_rows(matrix)))


def move_up(matrix):
    """向上移动"""
    return transpose(move_left(transpose(matrix)))


def move_down(matrix):
    """向下移动"""
    return transpose(move_right(transpose(matrix)))


SIMULATE_MAP = {
    "left": move_left,
    "right": move_right,
    "up": move_up,
    "down": move_down
}


def simulate_move(matrix, move_key):
    """模拟移动并返回新棋盘和是否发生变化"""
    if move_key not in SIMULATE_MAP:
        return matrix, False
    board_copy = copy.deepcopy(matrix)
    new_matrix = SIMULATE_MAP[move_key](board_copy)
    return new_matrix, (new_matrix != matrix)


def get_empty_cells(matrix):
    """返回所有空格的 (row, col) 列表"""
    return [(r, c) for r in range(4) for c in range(4) if matrix[r][c] == 0]


# ==================== 游戏模拟器 ====================

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

