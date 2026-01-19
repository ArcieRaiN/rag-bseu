import subprocess
import sys
import os
import venv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(PROJECT_DIR, ".venv")

# 1️⃣ Создаём виртуальное окружение
if not os.path.exists(ENV_DIR):
    print("Создаём виртуальное окружение...")
    venv.create(ENV_DIR, with_pip=True)
else:
    print("Виртуальное окружение уже существует.")

# 2️⃣ Формируем путь к pip внутри venv
if os.name == "nt":  # Windows
    pip_path = os.path.join(ENV_DIR, "Scripts", "pip.exe")
    python_path = os.path.join(ENV_DIR, "Scripts", "python.exe")
else:  # Linux / Mac
    pip_path = os.path.join(ENV_DIR, "bin", "pip")
    python_path = os.path.join(ENV_DIR, "bin", "python")

# 3️⃣ Устанавливаем зависимости
if os.path.exists(os.path.join(PROJECT_DIR, "requirements.txt")):
    print("Устанавливаем зависимости из requirements.txt...")
    subprocess.check_call([python_path, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([pip_path, "install", "-r", os.path.join(PROJECT_DIR, "requirements.txt")])
else:
    print("requirements.txt не найден, пропускаем установку зависимостей.")

print(f"Виртуальное окружение готово! Python: {python_path}")
print("В PyCharm можно выбрать этот интерпретатор:", python_path)
