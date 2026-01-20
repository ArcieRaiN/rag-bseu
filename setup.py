import os
import sys
import subprocess
import venv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(PROJECT_DIR, ".venv")

def in_venv():
    return sys.prefix != sys.base_prefix

# Пути внутри venv
if os.name == "nt":
    PYTHON_VENV = os.path.join(ENV_DIR, "Scripts", "python.exe")
    PIP_VENV = os.path.join(ENV_DIR, "Scripts", "pip.exe")
else:
    PYTHON_VENV = os.path.join(ENV_DIR, "bin", "python")
    PIP_VENV = os.path.join(ENV_DIR, "bin", "pip")

# 1️⃣ Если не в venv — создаём и перезапускаемся
if not in_venv():
    if not os.path.exists(ENV_DIR):
        print("Создаём виртуальное окружение...")
        venv.create(ENV_DIR, with_pip=True)
    else:
        print("Виртуальное окружение уже существует.")

    print("Перезапуск скрипта через виртуальное окружение...")
    subprocess.check_call([PYTHON_VENV, __file__])
    sys.exit(0)

# 2️⃣ Мы уже внутри venv
print("Работаем внутри виртуального окружения")
print("Python:", sys.executable)

# Установка зависимостей
requirements_file = os.path.join(PROJECT_DIR, "requirements.txt")
if os.path.exists(requirements_file):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
else:
    print("requirements.txt не найден")

print("Готово ✅")
