from abc import ABC, abstractmethod
from typing import Dict, Any

import babel.numbers


class ProductFormatter(ABC):
    locale_dict = {
        'DE': ('EUR', 'de_DE'),
        'JP': ('JPY', 'ja_JP'),
        'UK': ('GBP', 'en_GB'),
        'ES': ('EUR', 'es_ES'),
        'FR': ('EUR', 'fr_FR'),
        'IT': ('EUR', 'it_IT'),
    }

    @staticmethod
    def format_price(price: float, locale: str) -> str:
        currency, locale = ProductFormatter.locale_dict[locale]
        return babel.numbers.format_currency(price, currency, locale=locale)

    @abstractmethod
    def format(self, product: Dict[str, Any]) -> str:
        pass
