from __future__ import annotations

import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.belstat.gov.by"
COMPILATION_PATH = "/ofitsialnaya-statistika/publications/izdania/public_compilation/"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]


class SiteParser:
    """Парсер статистических сборников (compilation) с сайта Белстата.

    Скачивает только PDF-файлы, пропускает zip и прочие форматы.
    Имитирует поведение человека: случайные задержки, ротация User-Agent.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        max_pages: int = 2,
        delay_range: tuple[float, float] = (1.5, 4.0),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.max_pages = max_pages
        self.delay_range = delay_range
        self._session = requests.Session()

    def _random_headers(self) -> dict[str, str]:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.5,en;q=0.3",
            "Referer": BASE_URL + COMPILATION_PATH,
        }

    def _sleep(self) -> None:
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

    def _get_soup(self, url: str) -> BeautifulSoup:
        self._sleep()
        resp = self._session.get(url, headers=self._random_headers(), timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")

    def _detect_last_page(self, soup: BeautifulSoup) -> int:
        pagination = soup.find(class_="pagination")
        if not pagination:
            return 1
        nums = re.findall(r"PAGEN_1=(\d+)", str(pagination))
        return max(map(int, nums)) if nums else 1

    def _sanitize_filename(self, name: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', "", name).strip()

    def _download_file(self, url: str, dest: Path) -> None:
        self._sleep()
        resp = self._session.get(url, headers=self._random_headers(), timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)

    def parse(self) -> list[Path]:
        """Запускает скачивание. Возвращает список путей к скачанным PDF."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        listing_url = BASE_URL + COMPILATION_PATH + "?PAGEN_1=1"
        first_soup = self._get_soup(listing_url)
        last_available = self._detect_last_page(first_soup)
        pages_to_fetch = min(self.max_pages, last_available)

        print(f"Белстат: доступно страниц — {last_available}, "
              f"будет обработано — {pages_to_fetch}")

        downloaded: list[Path] = []

        for page_num in range(1, pages_to_fetch + 1):
            print(f"\n[{page_num}/{pages_to_fetch}] Обрабатываю страницу…")

            if page_num == 1:
                soup = first_soup
            else:
                page_url = f"{BASE_URL}{COMPILATION_PATH}?PAGEN_1={page_num}"
                soup = self._get_soup(page_url)

            table = soup.find("table")
            if table is None:
                print(f"  Таблица не найдена на странице {page_num}, пропуск.")
                continue

            rows = table.find_all("tr")[1:]  # без заголовка
            for row in rows:
                link_tag = row.find("a")
                if link_tag is None:
                    continue

                pub_url = urljoin(BASE_URL, link_tag["href"])
                pub_soup = self._get_soup(pub_url)

                download_tag = pub_soup.find(class_="link-download")
                if download_tag is None:
                    continue

                file_url = urljoin(BASE_URL, download_tag["href"])

                if not file_url.lower().endswith(".pdf"):
                    title = pub_soup.find("h1")
                    name_hint = title.text.strip() if title else file_url
                    print(f"  [Пропущено] Файл не в .pdf: {name_hint}")
                    continue

                title = pub_soup.find("h1")
                raw_name = title.text.strip() if title else "document"
                file_name = self._sanitize_filename(raw_name) + ".pdf"
                dest_path = self.output_dir / file_name

                if dest_path.exists():
                    print(f"  [Пропущено] Уже скачан: {file_name}")
                    continue

                try:
                    self._download_file(file_url, dest_path)
                    downloaded.append(dest_path)
                    print(f"  Скачано: {file_name}")
                except requests.RequestException as exc:
                    print(f"  [Ошибка] Не удалось скачать {file_name}: {exc}")

        print(f"\nГотово. Скачано файлов: {len(downloaded)}")
        return downloaded
