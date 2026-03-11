import pyautogui
import time

def reboot():
    pyautogui.hotkey('ctrl', 'r')

    time.sleep(2)

    target_x = 59
    target_y = 171

    pyautogui.click(x=target_x, y=target_y)

    print(f"Rebooted")

if __name__ == "__main__":
    print(pyautogui.position())
    print("Starting in 60 seconds...")
    time.sleep(60)
    reboot()