import subprocess
import sys
import os
import venv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(PROJECT_DIR, ".venv")

# Создаём виртуальное окружение, если ещё нет
if not os.path.exists(ENV_DIR):
    print("Создаём виртуальное окружение...")
    venv.create(ENV_DIR, with_pip=True)
else:
    print("Виртуальное окружение уже существует.")

# Пути внутри venv
if os.name == "nt":
    pip_path = os.path.join(ENV_DIR, "Scripts", "pip.exe")
    python_path = os.path.join(ENV_DIR, "Scripts", "python.exe")
else:
    pip_path = os.path.join(ENV_DIR, "bin", "pip")
    python_path = os.path.join(ENV_DIR, "bin", "python")

# Установка зависимостей
requirements_file = os.path.join(PROJECT_DIR, "requirements.txt")
if os.path.exists(requirements_file):
    print("Устанавливаем зависимости из requirements.txt...")
    subprocess.check_call([python_path, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([pip_path, "install", "-r", requirements_file])
else:
    print("requirements.txt не найден.")

print(f"Виртуальное окружение готово! Python: {python_path}")
print("В PyCharm можно выбрать этот интерпретатор:", python_path)
