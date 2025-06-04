import pyautogui
import time

def move_cursor_forever(interval=30, distance=10):
    print("Starting cursor movement. Press Ctrl+C to stop.")
    try:
        while True:
            x, y = pyautogui.position()
            pyautogui.moveTo(x + distance, y + distance, duration=0.2)
            time.sleep(1)
            pyautogui.moveTo(x, y, duration=0.2)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped cursor movement.")

if __name__ == "__main__":
    move_cursor_forever()
