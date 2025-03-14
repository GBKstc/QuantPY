# test_keylogger.py
from pynput import keyboard

def on_press(key):
    try:
        print(f'按下按键: {key.char}')
    except AttributeError:
        print(f'按下特殊按键: {key}')

def on_release(key):
    print(f'释放按键: {key}')
    if key == keyboard.Key.esc:
        # 按下 ESC 键停止监听
        return False

# 创建监听器
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    print("开始监听键盘输入（按 ESC 键退出）...")
    listener.join()