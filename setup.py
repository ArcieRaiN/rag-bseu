import os
import sys
import subprocess
import venv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ENV_DIR = PROJECT_DIR / "..venv"


def in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def get_venv_paths():
    if os.name == "nt":
        return (
            ENV_DIR / "Scripts" / "python.exe",
            ENV_DIR / "Scripts" / "pip.exe",
        )
    else:
        return (
            ENV_DIR / "bin" / "python",
            ENV_DIR / "bin" / "pip",
        )


def main():
    python_venv, _ = get_venv_paths()

    if not in_venv():
        if not ENV_DIR.exists():
            print("Создаём виртуальное окружение...")
            venv.create(ENV_DIR, with_pip=True)
        else:
            print("ℹВиртуальное окружение уже существует.")

        print("Перезапуск через .venv...")
        subprocess.check_call([str(python_venv), __file__])
        sys.exit(0)

    print("Работаем внутри .venv")
    print("Python:", sys.executable)

    req = PROJECT_DIR / "requirements.txt"
    if req.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
    else:
        print("requirements.txt не найден")

    print("Готово")


if __name__ == "__main__":
    main()
