"""회사·장비마다 갈아끼우는 층.

새 팹에 넣을 때 작성하는 코드는 여기 하나뿐이다. 원본 로그가 무엇이든
`DefectEvent` 목록으로 바꿔 주기만 하면 엔진은 그대로 돈다.
"""

from .base import REGISTRY, LogAdapter, register
from .klarf_like import KlarfLikeAdapter
from .flat_csv import FlatCsvAdapter

__all__ = ["LogAdapter", "REGISTRY", "register", "KlarfLikeAdapter", "FlatCsvAdapter"]
