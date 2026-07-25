"""Backport Python 3.11's inspect.BlockFinder decorator handling to 3.10."""

import inspect
import sys
import tokenize


if (
    sys.version_info[:2] == (3, 10)
    and hasattr(inspect.BlockFinder(), "decoratorhasargs")
):

    def _python311_tokeneater(self, type, token, srowcol, erowcol, line):
        if not self.started and not self.indecorator:
            if token == "@":
                self.indecorator = True
            elif token in ("def", "class", "lambda"):
                if token == "lambda":
                    self.islambda = True
                self.started = True
            self.passline = True
        elif type == tokenize.NEWLINE:
            self.passline = False
            self.last = srowcol[0]
            if self.islambda:
                raise inspect.EndOfBlock
            if self.indecorator:
                self.indecorator = False
        elif self.passline:
            pass
        elif type == tokenize.INDENT:
            if self.body_col0 is None and self.started:
                self.body_col0 = erowcol[1]
            self.indent += 1
            self.passline = True
        elif type == tokenize.DEDENT:
            self.indent -= 1
            if self.indent <= 0:
                raise inspect.EndOfBlock
        elif type == tokenize.COMMENT:
            if self.body_col0 is not None and srowcol[1] >= self.body_col0:
                self.last = srowcol[0]
        elif self.indent == 0 and type not in (tokenize.COMMENT, tokenize.NL):
            raise inspect.EndOfBlock

    inspect.BlockFinder.tokeneater = _python311_tokeneater
