"""여러 모듈에서 같이 쓰는 문자열 유틸."""
import re

# 캐릭터 이름이 프롬프트/키워드에 새어 들어갔는지 검사할 때 쓰는 표기 목록.
# 여기에 없는 캐릭터는 이름 그대로 검사한다.
NAME_ALIASES = {
    "Pac-Man":               ["pac man", "pacman"],
    "Wall-E":                ["wall e", "walle"],
    "Lightning McQueen":     ["lightning mcqueen", "mcqueen"],
    "Mike Wazowski":         ["mike wazowski", "wazowski"],
    "Puss in Boots":         ["puss in boots"],
    "Monkey D. Luffy":       ["monkey d luffy", "luffy"],
    "Chun-Li":               ["chun li", "chunli"],
    "Buzz Lightyear":        ["buzz lightyear", "lightyear"],
    "Astro Boy":             ["astro boy", "astroboy"],
    "Mr. Incredible":        ["mr incredible", "mister incredible"],
    "SpongeBob SquarePants": ["spongebob", "squarepants"],
    "Donald Duck":           ["donald duck", "donald"],
}


def slugify(name: str) -> str:
    """'Spider-Man' -> 'spider_man' (폴더/파일 이름용)"""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s.lower())).strip()


def has_name(text: str, name: str) -> bool:
    """text 안에 캐릭터 이름(또는 NAME_ALIASES의 표기)이 단어 단위로 들어 있으면 True"""
    t = f" {_norm(text)} "
    return any(f" {_norm(a)} " in t for a in NAME_ALIASES.get(name, [name]))


def build_negative_prompt(name: str, keywords: list) -> str:
    """생성 제외 키워드(네거티브 프롬프트) 문자열: '캐릭터명, 키워드1, ..., 키워드5'"""
    kws = [k for k in dict.fromkeys(keywords) if k.lower() != name.lower()]
    return f"{name}, " + ", ".join(kws)
