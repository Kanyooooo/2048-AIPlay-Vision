#!/usr/bin/env python3
"""
2048 AI自动玩家 - 机器学习版

支持两种运行模式:
1. 🧘 冥想模式 - 纯AI模拟训练
2. ⚔️ 实战模式 - 基于图像识别

使用方法:
    python AI2048.py

作者: AI Assistant + Human Collaboration
版本: 2.0 (模块化重构版)
"""

import time
import sys
import keyboard

# 导入各模块
from game_logic import Game2048Simulator, get_empty_cells
from vision_system import (
    GAME_REGION, SAVE_SCREENSHOTS, DEBUG_MODE,
    detect_game_region, create_screenshot_dir,
    get_board_matrix_from_screenshot,
    detect_try_again_button, click_try_again_button,
    ScoreTracker
)
from ai_strategy import (
    KEY_MAP, ALL_MOVES,
    StrategyParams, StrategyGenerator,
    find_best_move_ai, print_matrix,
    set_current_strategy
)
from ml_system import (
    GameRecorder, AdaptiveLearner
)

# 导入 vision_system 的全局变量模块
import vision_system


# ==================== 全局配置 ====================

ANIMATION_DELAY = 0.3  # 等待滑块动画完成
ML_ENABLED = True
ML_AUTO_OPTIMIZE = True
AUTO_RESTART = True
SUMMARY_FREQUENCY = 5
BUTTON_CHECK_INTERVAL = 3

# 全局对象
GAME_RECORDER = GameRecorder() if ML_ENABLED else None
ADAPTIVE_LEARNER = AdaptiveLearner(GAME_RECORDER) if ML_ENABLED and GAME_RECORDER else None
SCORE_TRACKER = ScoreTracker()


# ==================== 实战模式 ====================

def send_key(key):
    """发送键盘按键"""
    try:
        keyboard.press_and_release(key)
        time.sleep(ANIMATION_DELAY)
        return True
    except Exception as e:
        print(f"发送按键失败: {e}")
        return False


def play_game(strategy: StrategyParams = None, auto_restart=False, skip_region_detect=False):
    """主游戏循环 - 实战模式"""
    global SCORE_TRACKER
    
    # 设置当前策略
    if strategy:
        set_current_strategy(strategy)
    
    print("2048 视觉识别版本启动（无OCR依赖）...")
    print(f"当前策略: {strategy.name if strategy else '默认策略'}")
    print(f"自动重启: {'启用' if auto_restart else '禁用'}")
    print("请确保2048游戏窗口可见且未被遮挡")
    
    # 创建截图目录
    create_screenshot_dir()
    
    # 检测游戏区域
    if not skip_region_detect or vision_system.GAME_REGION is None:
        if not detect_game_region():
            print("无法检测游戏区域，退出")
            return False
    else:
        print(f"使用已保存的游戏区域: {vision_system.GAME_REGION}")
    
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
        GAME_RECORDER.start_game(strategy if strategy else StrategyParams())
    
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
            
            # 检测游戏结束
            try_again_btn = None
            should_check_button = (
                stuck_count > 3 or
                move_count % BUTTON_CHECK_INTERVAL == 0 or
                move_count < 5
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
                    # 更新学习器
                    if ADAPTIVE_LEARNER:
                        ADAPTIVE_LEARNER.update_all_learners(GAME_RECORDER.current_game or {})
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
                    return False
            
            # AI计算最佳移动
            move = find_best_move_ai(current_board)
            move_str = KEY_MAP.get(move, 'WAIT')
            
            # 显示当前状态
            current_score = SCORE_TRACKER.get_total_score()
            print_matrix(current_board, current_score, move_str)
            print(f"步数: {move_count + 1:<4} | 策略: {strategy.name if strategy else '默认'}")
            
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
                
                # 保存当前棋盘
                last_board_matrix = [row[:] for row in current_board]
            else:
                print("按键发送失败，重试...")
                time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n用户中断脚本。")
        if GAME_RECORDER and not game_completed:
            try:
                current_board = get_board_matrix_from_screenshot()
                final_score = SCORE_TRACKER.get_total_score()
                GAME_RECORDER.end_game(current_board, final_score, move_count)
            except:
                pass
        return False
    except Exception as e:
        import traceback
        print(f"\n运行中发生未知错误: {e}")
        traceback.print_exc()
        return False


def train_mode(max_games=None, algorithm='auto'):
    """智能训练模式 - 自动运行多局游戏并学习"""
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
                strategy = StrategyParams()
                print(f"\n🎯 第 {game_count} 局 - 使用默认策略")
            
            # 运行游戏
            should_continue = play_game(
                strategy=strategy,
                auto_restart=True,
                skip_region_detect=True
            )
            
            if not should_continue:
                print("\n❌ 游戏无法继续，训练终止")
                break
            
            if max_games and game_count >= max_games:
                print(f"\n✅ 已完成 {max_games} 局训练")
                break
            
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
        if game_count > 0:
            print(f"平均每局: {elapsed_time/game_count:.1f} 秒")
        
        if GAME_RECORDER:
            print("\n最终统计:")
            GAME_RECORDER.print_statistics()


# ==================== 冥想模式 ====================

def simulate_one_game(strategy: StrategyParams, show_realtime=False, thinking_mode=False):
    """模拟一局游戏（冥想模式）"""
    # 设置当前策略
    set_current_strategy(strategy)
    
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
            time.sleep(0.05)
        
        # 执行移动
        success = simulator.move(best_move)
        if not success:
            break
        
        move_count += 1
    
    # 显示最终状态
    if show_realtime:
        print_matrix(simulator.matrix, simulator.score, "END")
        print(f"游戏结束 | 最大方块: {simulator.get_max_tile()} | 总分: {simulator.score}")
        time.sleep(1)
    
    # 返回结果
    state = simulator.get_state()
    return {
        'final_matrix': state['matrix'],
        'final_score': state['score'],
        'move_count': state['move_count'],
        'max_tile': state['max_tile']
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
        loop_count = None
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
                strategy = StrategyParams()
            elif ADAPTIVE_LEARNER and game_count > 1:
                strategy = ADAPTIVE_LEARNER.suggest_next_strategy(mode=algorithm)
            else:
                strategy = StrategyParams()
            
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
                # 更新学习器
                if ADAPTIVE_LEARNER and GAME_RECORDER.current_game:
                    ADAPTIVE_LEARNER.update_all_learners(GAME_RECORDER.current_game)
            
            # 显示进度
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
        if game_count > 0:
            print(f"平均每局: {elapsed_time/game_count:.2f} 秒")
            print(f"训练速度: {game_count/elapsed_time:.1f} 局/秒")
        
        if GAME_RECORDER:
            print("\n最终统计:")
            GAME_RECORDER.print_statistics()
            
            if GAME_RECORDER.history["best_strategy"]:
                print("\n🏆 最佳策略:")
                best = GAME_RECORDER.history["best_strategy"]
                print(f"  名称: {best['name']}")
                print(f"  角落: {best['corner']}")
                print(f"  优先级: {best['priority']}")


# ==================== 策略测试和查看 ====================

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
        strategies = StrategyGenerator.generate_weight_variants(StrategyParams())
    elif choice == "3":
        strategies.extend(StrategyGenerator.generate_corner_strategies())
        strategies.extend(StrategyGenerator.generate_weight_variants(StrategyParams(), count=3))
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


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 检查依赖
    try:
        import cv2
        import pyautogui
        import keyboard
        import numpy as np
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装以下依赖:")
        print("pip install opencv-python pyautogui keyboard pillow numpy")
        sys.exit(1)
    
    # 显示欢迎信息
    print("\n" + "="*60)
    print("🎮 2048 AI 自动玩家 - 机器学习版")
    print("="*60)
    
    # 选择运行模式
    print("\n请选择运行模式:")
    print("  1. 🧘 冥想模式 - 纯AI模拟训练（不需要游戏界面，速度极快）")
    print("  2. ⚔️  实战模式 - 基于图像识别控制真实游戏")
    print("  3. 📊 查看训练数据统计")
    
    mode_choice = input("\n请选择模式 (1-3) [默认: 2]: ").strip()
    
    if mode_choice == "1":
        # 冥想模式
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
    
    # 实战模式
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
            loop_count = None
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
    
    # 非自动训练模式 - 单局游戏
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
                test_multiple_strategies()
                return
            elif choice == "4":
                print_strategy_details()
                return
            elif choice == "2":
                play_game()
            else:
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


if __name__ == "__main__":
    main()

