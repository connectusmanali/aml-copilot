"""Mask personal data before any text leaves the bank; restore it in the response.

Account and customer IDs become stable placeholders (ACCT_1, CUST_1, ...); emails, phone numbers
and card-like numbers become typed tokens. The mapping stays in memory on our side: the model
reasons about ACCT_1 and ACCT_2, and the investigator reads the real IDs.

Masking happens in the gateway, the single choke point every model call passes through, so no
individual prompt can forget it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ACCOUNT = re.compile(r"\b(?:T[0-9a-f]{6}-)?\d{3}-[0-9A-F]{8,9}\b")
CUSTOMER = re.compile(r"\bC[0-9A-F]{12}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass
class Masker:
    forward: dict[str, str] = field(default_factory=dict)
    backward: dict[str, str] = field(default_factory=dict)

    def _token(self, value: str, kind: str) -> str:
        if value not in self.forward:
            n = sum(1 for t in self.backward if t.startswith(kind)) + 1
            token = f"{kind}_{n}"
            self.forward[value] = token
            self.backward[token] = value
        return self.forward[value]

    def mask(self, text: str) -> str:
        text = ACCOUNT.sub(lambda m: self._token(m.group(0), "ACCT"), text)
        text = CUSTOMER.sub(lambda m: self._token(m.group(0), "CUST"), text)
        text = EMAIL.sub("[EMAIL]", text)
        text = CARD.sub("[CARD]", text)
        return PHONE.sub("[PHONE]", text)

    def unmask(self, text: str) -> str:
        # Longest tokens first, so ACCT_12 is not replaced as ACCT_1 + "2".
        for token in sorted(self.backward, key=len, reverse=True):
            text = text.replace(token, self.backward[token])
        return text


def contains_pii(text: str) -> bool:
    return any(p.search(text) for p in (ACCOUNT, CUSTOMER, EMAIL, CARD))
