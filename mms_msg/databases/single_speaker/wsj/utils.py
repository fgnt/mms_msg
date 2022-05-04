PUNCTUATION_SYMBOLS = set('''
    &AMPERSAND
    ,COMMA
    ;SEMI-COLON
    :COLON
    !EXCLAMATION-POINT
    ...ELLIPSIS
    -HYPHEN
    .PERIOD
    .DOT
    ?QUESTION-MARK

    .DECIMAL
    .PERCENT
    /SLASH

    'SINGLE-QUOTE
    "DOUBLE-QUOTE
    "QUOTE
    "UNQUOTE
    "END-OF-QUOTE
    "END-QUOTE
    "CLOSE-QUOTE
    "IN-QUOTES

    (PAREN
    (PARENTHESES
    (IN-PARENTHESIS
    (BRACE
    (LEFT-PAREN
    (PARENTHETICALLY
    (BEGIN-PARENS
    )CLOSE-PAREN
    )CLOSE_PAREN
    )END-THE-PAREN
    )END-OF-PAREN
    )END-PARENS
    )CLOSE-BRACE
    )RIGHT-PAREN
    )UN-PARENTHESES
    )PAREN

    {LEFT-BRACE
    }RIGHT-BRACE
'''.split())

DEBUG_EXAMPLE_LIMIT = 10


def filter_punctuation_pronunciation(example):
    transcription = example['kaldi_transcription'].split()
    return len(PUNCTUATION_SYMBOLS.intersection(transcription)) == 0
