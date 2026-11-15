from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("page", ["Home.py", *sorted(f"pages/{p.name}" for p in ROOT.glob("pages/*.py"))])
def test_page_runs(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=240).run()
    assert not at.exception, at.exception
