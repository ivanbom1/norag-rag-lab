import pytest

from app.redaction import Redactor


@pytest.mark.parametrize("value, kind", [
    ("marie.dupont@example.com", "EMAIL"),
    ("01 99 00 12 34", "PHONE"),
    ("06.99.00.56.78", "PHONE"),
    ("+33 6 99 00 56 78", "PHONE"),
    ("+44 20 7946 0958", "PHONE"),
    ("(555) 010-4477", "PHONE"),
    ("555-010-9921", "PHONE"),
    ("FR76 3000 6000 0112 3456 7890 189", "IBAN"),
    ("4111 1111 1111 1111", "CARD"),
    ("203.0.113.42", "IP"),
])
def test_personal_data_is_replaced(value, kind):
    redactor = Redactor()
    text = redactor.redact(f"Contact: {value}, thanks.")
    assert text == f"Contact: [{kind}_1], thanks."
    assert redactor.restore(text) == f"Contact: {value}, thanks."


@pytest.mark.parametrize("text", [
    "Effective 2026-11-14, or 14/11/2026 at 08:00–20:00.",
    "Error KST-2044 on product-guide.md#3, see section 4.2.1.",
    "The Business plan costs 1,200.50 EUR for 25 seats in 2025 and 2026.",
    "Order 1234 5678 9012 3456 failed.",  # 16 digits, not a valid card number
    "FR76 3000 6000 0112 3456 7890 180 is mistyped.",  # wrong IBAN checksum
    "Version 3.5 of gemini-3.5-flash-lite.",
])
def test_lookalikes_are_left_alone(text):
    assert Redactor().redact(text) == text


def test_one_placeholder_per_value_in_a_request():
    redactor = Redactor()
    text = redactor.redact("a@example.com, b@example.com, a@example.com, 01 99 00 12 34")
    assert text == "[EMAIL_1], [EMAIL_2], [EMAIL_1], [PHONE_1]"
    assert dict(redactor.counts) == {"EMAIL": 2, "PHONE": 1}
    assert redactor.restore("[EMAIL_2] and [EMAIL_9]") == "b@example.com and [EMAIL_9]"
