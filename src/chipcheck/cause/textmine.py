"""로그 텍스트용 경량 검색. 외부 의존성 없이 BM25까지만.

LLM을 쓰지 않는 이유가 여기 있다. 원인 후보는 고정된 문서 집합이고,
거기서 질의에 맞는 것을 순위 매기는 일은 역색인으로 끝난다. 문서 1,000건
기준 질의 한 번이 밀리초 아래다.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# 한글·영문·숫자 덩어리만 남긴다. 장비 로그의 코드(E-1234, RF_PWR)도 살린다.
_TOKEN = re.compile(r"[0-9A-Za-z_\-]+|[가-힣]+")


def tokenize(text: str, bigram_hangul: bool = True) -> list[str]:
    """공백 분리 + 한글은 글자 2-gram까지.

    한국어 조사·어미 때문에 '스크래치가'와 '스크래치는'이 다른 토큰이 된다.
    형태소 분석기를 끌어오는 대신 글자 2-gram을 같이 넣어 그 차이를 흡수한다.
    """
    out: list[str] = []
    for m in _TOKEN.findall(text.lower()):
        out.append(m)
        if bigram_hangul and len(m) >= 2 and "가" <= m[0] <= "힣":
            out.extend(m[i : i + 2] for i in range(len(m) - 1))
    return out


@dataclass
class BM25:
    """표준 BM25. k1·b는 널리 쓰이는 기본값."""

    k1: float = 1.5
    b: float = 0.75
    docs: list[list[str]] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _tf: list[Counter] = field(default_factory=list)
    _len: list[int] = field(default_factory=list)
    _avg: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        toks = tokenize(text)
        self.doc_ids.append(doc_id)
        self.docs.append(toks)
        tf = Counter(toks)
        self._tf.append(tf)
        self._len.append(len(toks))
        for t in tf:
            self._df[t] += 1

    def build(self) -> "BM25":
        self._avg = sum(self._len) / max(len(self._len), 1)
        return self

    def idf(self, term: str) -> float:
        n, df = len(self.docs), self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> dict[str, float]:
        """문서별 점수와, 어떤 토큰이 걸렸는지 함께 쓰도록 분리해 둔다."""
        q = tokenize(query)
        out: dict[str, float] = {}
        for i, doc_id in enumerate(self.doc_ids):
            tf, dl = self._tf[i], self._len[i]
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self._avg, 1e-9))
                s += self.idf(term) * f * (self.k1 + 1) / denom
            if s:
                out[doc_id] = s
        return out

    def matched_terms(self, doc_id: str, query: str, top: int = 6) -> list[str]:
        """콘솔에 '왜 이게 올라왔는지' 보여줄 근거 토큰."""
        i = self.doc_ids.index(doc_id)
        tf = self._tf[i]
        hits = {t: self.idf(t) * tf[t] for t in set(tokenize(query)) if tf.get(t)}
        ranked = sorted(hits, key=lambda t: -hits[t])
        # 2-gram 조각보다 온전한 단어를 먼저 보여준다
        ranked.sort(key=lambda t: (len(t) < 3, -hits[t]))
        return ranked[:top]
