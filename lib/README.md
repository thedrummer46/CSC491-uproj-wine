# lib/

`ezr.py` is the lecturer's ezr library, copied **unchanged** from
[timm/ezr](https://github.com/timm/ezr/blob/e2d95ae354e586c6bfc60b6b6ab47aff2468d31e/ezr.py)
at commit `e2d95ae`. It's MIT-licensed, © 2026 Tim Menzies (see the file's header). It needs Python 3.12+ and only the standard library.

It's used two ways:
- **Background library for T5/T6.** It's the "ezr" the LLM is given.
- **The harness's stats and scoring:** `same`, `wins`, `ydist`.

Don't edit it. If a newer version is needed, re-copy it and update the commit above.
