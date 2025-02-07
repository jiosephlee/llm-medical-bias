# How handbook was cleaned

1) Manually selected relevant pages
2) sed pymupdf4llm to convert pdf to md
3) Manually deleted references
4) Manually deleted Tables
5) Automatically deleted any statements that started with ** and ended with **, which signifies words that are in figures
6) Automatically replaced pdf-style paragraphs with actual paragraphs i.e. sentences in pdfs are separated by newlines for formatting sake. 
7) Automatically deleted "-------" which signify page breaks
8) Manually deleted newlines that broke up paragraphs in the middle
9) Manully deleted unnecessary phrases (e.g. trademarks)

Time: 1 hour