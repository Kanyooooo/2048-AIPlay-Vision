#!/usr/bin/env python3
"""
视觉识别系统模块

包含：
- 游戏区域检测
- 颜色识别和数字识别
- Try Again 按钮检测
- 分数追踪器
"""

import time
import math
import os
import numpy as np
import pyautogui
import keyboard


# ==================== 配置参数 ====================

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
    256: (237, 204, 97),  # 256 #EDCC61
    512: (237, 200, 80),  # 512 #EDC850
    1024: (237, 197, 63),  # 1024 #EDC53F
    2048: (237, 194, 46),  # 2048 #EDC22E
}

# Try Again 按钮相对位置配置（基准：500x500游戏区域）
TRY_AGAIN_RELATIVE_X = 250  # 相对于游戏区域左上角的X偏移（基准500宽）
TRY_AGAIN_RELATIVE_Y = 360  # 相对于游戏区域左上角的Y偏移（基准500高）
TRY_AGAIN_BASE_SIZE = 500   # 基准游戏区域大小（500x500）
USE_RELATIVE_CLICK = True   # 优先使用相对位置点击（更快更准）

TRY_AGAIN_BUTTON = "try_again.png"  # Try again 按钮图片

# 全局变量
GAME_REGION = None
SAVE_SCREENSHOTS = False
SCREENSHOT_DIR = "screenshots"
DEBUG_MODE = False


# ==================== 游戏区域检测 ====================

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


# ==================== 颜色识别系统 ====================

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


# ==================== Try Again 按钮检测 ====================

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
    """点击 Try again 按钮 - 优化版（优先使用相对位置）"""
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
        """根据棋盘变化计算并累加分数"""
        if old_matrix is None or new_matrix is None:
            return 0
        
        # 计算所有方块的总和差异
        old_sum = sum(sum(row) for row in old_matrix)
        new_sum = sum(sum(row) for row in new_matrix)
        
        # 计算新增的方块值（通常是2或4）
        new_tiles_sum = 0
        for r in range(4):
            for c in range(4):
                if old_matrix[r][c] == 0 and new_matrix[r][c] != 0:
                    new_tiles_sum += new_matrix[r][c]
        
        # 得分 = 总和增加 - 新生成的方块
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

