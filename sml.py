import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Tuple, Optional

#########################################################################
# NOTE:
# Implementing a FULL Standard ML (SML) parser is a non-trivial task that
# typically requires a fully-fledged parser generator or a large amount of
# hand-written code. The following code attempts to provide a richer subset
# parser with more SML-like features than the previous example, but is still
# far from a complete SML parser.
#
# The code supports:
#  - Integer and Boolean literals
#  - Variables
#  - Basic arithmetic: +, -, *, div, mod
#  - Relational ops: =, <, <=, >, >=
#  - Boolean ops: andalso, orelse, not
#  - If-then-else expressions
#  - Let-in-end expressions: let val x = expr in expr end
#  - Function definitions: fn <pattern> => expr
#  - Function application: expr expr (left-associative)
#  - Tuples: (expr, expr, ...)
#  - Lists: [expr, expr, ...]
#  - Case expressions: case expr of pattern => expr | pattern => expr ...
#  - Patterns: variable patterns, wildcard (_), tuple patterns, list patterns,
#    integer and boolean constants, and "as" patterns.
#
# This is still a simplification of SML syntax.
#
#########################################################################

#########################################################################
# AST Definitions
#########################################################################

class SmlAst(ABC):
    @abstractmethod
    def eval(self, env: Dict[str, Any]) -> Any:
        pass

# Patterns for pattern matching
class SmlPattern(ABC):
    @abstractmethod
    def match(self, value: Any) -> Optional[Dict[str, Any]]:
        """
        Try to match the given value against this pattern.
        If matches, return a dict of variable bindings.
        If not, return None.
        """
        pass

class PVar(SmlPattern):
    def __init__(self, name: str):
        self.name = name

    def match(self, value: Any) -> Optional[Dict[str, Any]]:
        return {self.name: value}

    def __repr__(self):
        return f"PVar({self.name})"

class PWildcard(SmlPattern):
    def match(self, value: Any) -> Optional[Dict[str, Any]]:
        return {}

    def __repr__(self):
        return f"PWildcard()"

class PInt(SmlPattern):
    def __init__(self, val: int):
        self.val = val

    def match(self, value: Any) -> Optional[Dict[str, Any]]:
        if value == self.val:
            return {}
        return None

    def __repr__(self):
        return f"PInt({self.val})"

class PBool(SmlPattern):
    def __init__(self, val: bool):
        self.val = val

    def match(self, value: Any) -> Optional[Dict[str, Any]]:
        if value == self.val:
            return {}
        return None

    def __repr__(self):
        return f"PBool({self.val})"

class PTuple(SmlPattern):
    def __init__(self, patterns: List[SmlPattern]):
        self.patterns = patterns

    def match(self, value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, tuple):
            return None
        if len(value) != len(self.patterns):
            return None
        bindings = {}
        for p, v in zip(self.patterns, value):
            res = p.match(v)
            if res is None:
                return None
            for k, val in res.items():
                if k in bindings:
                    return None
                bindings[k] = val
        return bindings

    def __repr__(self):
        return f"PTuple({self.patterns})"

class PList(SmlPattern):
    def __init__(self, patterns: List[SmlPattern]):
        self.patterns = patterns

    def match(self, value: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(value, list):
            return None
        if len(value) != len(self.patterns):
            return None
        bindings = {}
        for p, v in zip(self.patterns, value):
            res = p.match(v)
            if res is None:
                return None
            for k, val in res.items():
                if k in bindings:
                    return None
                bindings[k] = val
        return bindings

    def __repr__(self):
        return f"PList({self.patterns})"

class PAs(SmlPattern):
    # pattern: "x as pattern"
    def __init__(self, var_name: str, pattern: SmlPattern):
        self.var_name = var_name
        self.pattern = pattern

    def match(self, value: Any) -> Optional[Dict[str, Any]]:
        res = self.pattern.match(value)
        if res is not None:
            if self.var_name in res:
                return None
            res[self.var_name] = value
            return res
        return None

    def __repr__(self):
        return f"PAs({self.var_name}, {self.pattern})"


# Expressions
class SmlInt(SmlAst):
    def __init__(self, value: int):
        self.value = value

    def eval(self, env: Dict[str, Any]) -> Any:
        return self.value

    def __repr__(self):
        return f"SmlInt({self.value})"

class SmlBool(SmlAst):
    def __init__(self, value: bool):
        self.value = value

    def eval(self, env: Dict[str, Any]) -> Any:
        return self.value

    def __repr__(self):
        return f"SmlBool({self.value})"

class SmlVar(SmlAst):
    def __init__(self, name: str):
        self.name = name

    def eval(self, env: Dict[str, Any]) -> Any:
        return env[self.name]

    def __repr__(self):
        return f"SmlVar({self.name})"

class SmlIf(SmlAst):
    def __init__(self, cond: SmlAst, then_branch: SmlAst, else_branch: SmlAst):
        self.cond = cond
        self.then_branch = then_branch
        self.else_branch = else_branch

    def eval(self, env: Dict[str, Any]) -> Any:
        c = self.cond.eval(env)
        if c:
            return self.then_branch.eval(env)
        else:
            return self.else_branch.eval(env)

    def __repr__(self):
        return f"SmlIf({self.cond}, {self.then_branch}, {self.else_branch})"

class SmlLet(SmlAst):
    # let val x = expr in expr end
    def __init__(self, bindings: List[Tuple[SmlPattern, SmlAst]], body: SmlAst):
        self.bindings = bindings
        self.body = body

    def eval(self, env: Dict[str, Any]) -> Any:
        new_env = env.copy()
        for pattern, expr in self.bindings:
            val = expr.eval(new_env)
            match_res = pattern.match(val)
            if match_res is None:
                raise ValueError("Pattern match failure in let binding.")
            for k, v in match_res.items():
                new_env[k] = v
        return self.body.eval(new_env)

    def __repr__(self):
        return f"SmlLet(bindings={self.bindings}, body={self.body})"

class SmlFn(SmlAst):
    # fn pattern => expr
    def __init__(self, pattern: SmlPattern, body: SmlAst):
        self.pattern = pattern
        self.body = body

    def eval(self, env: Dict[str, Any]) -> Any:
        # return a closure: (tag, env, pattern, body)
        return ("closure", env.copy(), self.pattern, self.body)

    def __repr__(self):
        return f"SmlFn({self.pattern} => {self.body})"

class SmlApp(SmlAst):
    # function application: expr expr
    def __init__(self, func: SmlAst, arg: SmlAst):
        self.func = func
        self.arg = arg

    def eval(self, env: Dict[str, Any]) -> Any:
        closure = self.func.eval(env)
        if not (isinstance(closure, tuple) and closure[0] == "closure"):
            raise ValueError("Attempt to apply a non-function value.")
        _, clos_env, pattern, body = closure
        arg_val = self.arg.eval(env)
        match_res = pattern.match(arg_val)
        if match_res is None:
            raise ValueError("Pattern match failure in function application.")
        new_env = clos_env.copy()
        for k, v in match_res.items():
            new_env[k] = v
        return body.eval(new_env)

    def __repr__(self):
        return f"SmlApp({self.func}, {self.arg})"

class SmlTuple(SmlAst):
    def __init__(self, elements: List[SmlAst]):
        self.elements = elements

    def eval(self, env: Dict[str, Any]) -> Any:
        return tuple(e.eval(env) for e in self.elements)

    def __repr__(self):
        return f"SmlTuple({self.elements})"

class SmlList(SmlAst):
    def __init__(self, elements: List[SmlAst]):
        self.elements = elements

    def eval(self, env: Dict[str, Any]) -> Any:
        return [e.eval(env) for e in self.elements]

    def __repr__(self):
        return f"SmlList({self.elements})"

class SmlCase(SmlAst):
    # case expr of pattern => expr | pattern => expr | ...
    def __init__(self, expr: SmlAst, branches: List[Tuple[SmlPattern, SmlAst]]):
        self.expr = expr
        self.branches = branches

    def eval(self, env: Dict[str, Any]) -> Any:
        val = self.expr.eval(env)
        for (pattern, expr) in self.branches:
            match_res = pattern.match(val)
            if match_res is not None:
                new_env = env.copy()
                for k, v in match_res.items():
                    new_env[k] = v
                return expr.eval(new_env)
        raise ValueError("Non-exhaustive match in case expression.")

    def __repr__(self):
        return f"SmlCase({self.expr}, {self.branches})"

class SmlUnOp(SmlAst):
    # Unary operator: not
    def __init__(self, op: str, expr: SmlAst):
        self.op = op
        self.expr = expr

    def eval(self, env: Dict[str, Any]) -> Any:
        val = self.expr.eval(env)
        if self.op == 'not':
            return not val
        else:
            raise ValueError("Unknown unary operator: " + self.op)

    def __repr__(self):
        return f"SmlUnOp({self.op}, {self.expr})"


class SmlBinOp(SmlAst):
    # binary operations: +, -, *, div, mod, =, <, <=, >, >=, andalso, orelse
    def __init__(self, op: str, left: SmlAst, right: SmlAst):
        self.op = op
        self.left = left
        self.right = right

    def eval(self, env: Dict[str, Any]) -> Any:
        lval = self.left.eval(env)
        rval = self.right.eval(env)
        if self.op == '+':
            return lval + rval
        elif self.op == '-':
            return lval - rval
        elif self.op == '*':
            return lval * rval
        elif self.op == 'div':
            return lval // rval
        elif self.op == 'mod':
            return lval % rval
        elif self.op == '=':
            return lval == rval
        elif self.op == '<':
            return lval < rval
        elif self.op == '<=':
            return lval <= rval
        elif self.op == '>':
            return lval > rval
        elif self.op == '>=':
            return lval >= rval
        elif self.op == 'andalso':
            return lval and rval
        elif self.op == 'orelse':
            return lval or rval
        else:
            raise ValueError("Unknown operator: " + self.op)

    def __repr__(self):
        return f"SmlBinOp({self.op}, {self.left}, {self.right})"


#########################################################################
# Tokenization
#########################################################################

# IMPORTANT: The order of tokens matters: ARROW must come before OP so that '=>'
# is not parsed as '=' and '>'.
token_spec = [
    ("LET", r'let'),
    ("VAL", r'val'),
    ("IN", r'in'),
    ("END", r'end'),
    ("FN", r'fn'),
    ("CASE", r'case'),
    ("OF", r'of'),
    ("IF", r'if'),
    ("THEN", r'then'),
    ("ELSE", r'else'),
    ("DIV", r'div'),
    ("MOD", r'mod'),
    ("ANDALSO", r'andalso'),
    ("ORELSE", r'orelse'),
    ("NOT", r'not'),
    ("AS", r'as'),
    ("ARROW", r'=>'),
    ("BAR", r'\|'),
    ("COMMA", r','),
    ("LPAREN", r'\('),
    ("RPAREN", r'\)'),
    ("LBRACK", r'\['),
    ("RBRACK", r'\]'),
    ("UNDERSCORE", r'_'),
    ("INT", r'\d+'),
    ("BOOL", r'true|false'),
    ("OP", r'[=<>*+\-]'),
    ("ID", r'[a-zA-Z_]\w*'),
    ("EOF", r'$'),
    ("SKIP", r'[ \t\n\r]+'),
]

token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)

class SMLLexer:
    def __init__(self, text: str):
        self.tokens = []
        for m in re.finditer(token_regex, text):
            kind = m.lastgroup
            val = m.group()
            if kind == 'INT':
                val = int(val)
            elif kind == 'BOOL':
                val = (val == 'true')
            elif kind == 'SKIP':
                continue
            self.tokens.append((kind, val))
        self.tokens.append(('EOF','EOF'))
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF','EOF')

    def consume(self, kind=None):
        t = self.peek()
        if kind and t[0] != kind:
            raise ValueError(f"Expected {kind}, got {t}")
        self.pos += 1
        return t


#########################################################################
# Parser
#########################################################################
#
# Grammar (simplified):
#
#   top := expr EOF
#
#   expr := let_expr
#   let_expr := 'let' {val_decl} 'in' expr 'end' | fn_expr
#   val_decl := 'val' pattern '=' expr
#
#   fn_expr := 'fn' pattern '=>' expr | if_expr
#
#   if_expr := 'if' expr 'then' expr 'else' expr | case_expr
#
#   case_expr := 'case' expr 'of' case_branch { '|' case_branch } | infix_expr
#   case_branch := pattern '=>' expr
#
#   infix_expr := cmp_expr { ( 'andalso' | 'orelse' ) cmp_expr }
#
#   cmp_expr := arith_expr {( '=' | '<' | '<=' | '>' | '>=' ) arith_expr}
#
#   arith_expr := term {('+'|'-') term}
#
#   term := factor {('*'|'div'|'mod') factor}
#
#   factor := 'not' factor | application
#
#   application := atomic {atomic}
#
#   atomic := INT | BOOL | ID | '(' expr {',' expr} ')' | '[' [expr {',' expr}] ']' | fn_expr | if_expr | case_expr | let_expr
#
# pattern := p_atomic { 'as' ID }
#
# p_atomic := '_' | INT | BOOL | ID | '(' pattern {',' pattern} ')' | '[' pattern {',' pattern} ']'
#
#########################################################################

class SMLParser:
    def __init__(self, text: str):
        self.lexer = SMLLexer(text)

    def peek(self):
        return self.lexer.peek()

    def consume(self, kind=None):
        return self.lexer.consume(kind)

    def parse(self) -> SmlAst:
        expr = self.parse_expr()
        self.consume('EOF')
        return expr

    # expr
    def parse_expr(self):
        return self.parse_let_expr()

    def parse_let_expr(self):
        t = self.peek()
        if t[0] == 'LET':
            self.consume('LET')
            bindings = []
            while self.peek()[0] == 'VAL':
                self.consume('VAL')
                pat = self.parse_pattern()
                self.consume('OP') # expect '='
                rhs = self.parse_expr()
                bindings.append((pat, rhs))
            self.consume('IN')
            body = self.parse_expr()
            self.consume('END')
            return SmlLet(bindings, body)
        else:
            return self.parse_fn_expr()

    def parse_fn_expr(self):
        t = self.peek()
        if t[0] == 'FN':
            self.consume('FN')
            pat = self.parse_pattern()
            self.consume('ARROW')
            body = self.parse_expr()
            return SmlFn(pat, body)
        else:
            return self.parse_if_expr()

    def parse_if_expr(self):
        t = self.peek()
        if t[0] == 'IF':
            self.consume('IF')
            cond = self.parse_expr()
            self.consume('THEN')
            then_expr = self.parse_expr()
            self.consume('ELSE')
            else_expr = self.parse_expr()
            return SmlIf(cond, then_expr, else_expr)
        else:
            return self.parse_case_expr()

    def parse_case_expr(self):
        t = self.peek()
        if t[0] == 'CASE':
            self.consume('CASE')
            scrutinee = self.parse_expr()
            self.consume('OF')
            branches = []
            while True:
                pat = self.parse_pattern()
                self.consume('ARROW')
                br = self.parse_expr()
                branches.append((pat, br))
                if self.peek()[0] == 'BAR':
                    self.consume('BAR')
                else:
                    break
            return SmlCase(scrutinee, branches)
        else:
            return self.parse_infix_expr()

    def parse_infix_expr(self):
        expr = self.parse_cmp_expr()
        while self.peek()[0] in ('ANDALSO', 'ORELSE'):
            op = self.consume()[1]
            right = self.parse_cmp_expr()
            expr = SmlBinOp(op, expr, right)
        return expr

    def parse_cmp_expr(self):
        expr = self.parse_arith_expr()
        while self.peek()[0] == 'OP' and self.peek()[1] in ('=', '<', '<=', '>', '>='):
            op = self.consume()[1]
            right = self.parse_arith_expr()
            expr = SmlBinOp(op, expr, right)
        return expr

    def parse_arith_expr(self):
        expr = self.parse_term()
        while self.peek()[0] == 'OP' and self.peek()[1] in ('+', '-'):
            op = self.consume()[1]
            right = self.parse_term()
            expr = SmlBinOp(op, expr, right)
        return expr

    def parse_term(self):
        expr = self.parse_factor()
        while True:
            t = self.peek()
            if t[0] == 'OP' and t[1] == '*':
                op = self.consume()[1]
                right = self.parse_factor()
                expr = SmlBinOp(op, expr, right)
            elif t[0] in ('DIV', 'MOD'):
                op = self.consume()[1]
                right = self.parse_factor()
                expr = SmlBinOp(op, expr, right)
            else:
                break
        return expr

    def parse_factor(self):
        t = self.peek()
        if t[0] == 'NOT':
            self.consume('NOT')
            expr = self.parse_factor()
            return SmlUnOp('not', expr)
        else:
            return self.parse_application()

    def parse_application(self):
        expr = self.parse_atomic()
        while True:
            nxt = self.peek()
            # Check if next token can start an atomic expression
            if nxt[0] in ('INT','BOOL','ID','UNDERSCORE','LPAREN','LBRACK','FN','IF','CASE','LET'):
                arg = self.parse_atomic()
                expr = SmlApp(expr, arg)
            else:
                break
        return expr

    def parse_atomic(self):
        t = self.peek()
        if t[0] == 'INT':
            self.consume('INT')
            return SmlInt(t[1])
        elif t[0] == 'BOOL':
            self.consume('BOOL')
            return SmlBool(t[1])
        elif t[0] == 'ID':
            self.consume('ID')
            return SmlVar(t[1])
        elif t[0] == 'UNDERSCORE':
            # underscore not allowed as standalone expr
            # raise error
            raise ValueError("Underscore found in expression context.")
        elif t[0] == 'LPAREN':
            self.consume('LPAREN')
            # could be tuple or single expr
            expr = self.parse_expr()
            elems = [expr]
            while self.peek()[0] == 'COMMA':
                self.consume('COMMA')
                elems.append(self.parse_expr())
            self.consume('RPAREN')
            if len(elems) == 1:
                return elems[0]
            else:
                return SmlTuple(elems)
        elif t[0] == 'LBRACK':
            self.consume('LBRACK')
            elems = []
            if self.peek()[0] != 'RBRACK':
                elems.append(self.parse_expr())
                while self.peek()[0] == 'COMMA':
                    self.consume('COMMA')
                    elems.append(self.parse_expr())
            self.consume('RBRACK')
            return SmlList(elems)
        elif t[0] == 'FN':
            return self.parse_fn_expr()
        elif t[0] == 'IF':
            return self.parse_if_expr()
        elif t[0] == 'CASE':
            return self.parse_case_expr()
        elif t[0] == 'LET':
            return self.parse_let_expr()
        else:
            raise ValueError(f"Unexpected token {t} in atomic expression.")

    def parse_pattern(self):
        # pattern: p_atomic { 'as' ID }
        pat = self.parse_p_atomic()
        while self.peek()[0] == 'AS':
            self.consume('AS')
            v = self.consume('ID')[1]
            pat = PAs(v, pat)
        return pat

    def parse_p_atomic(self):
        t = self.peek()
        if t[0] == 'UNDERSCORE':
            self.consume('UNDERSCORE')
            return PWildcard()
        elif t[0] == 'INT':
            val = t[1]
            self.consume('INT')
            return PInt(val)
        elif t[0] == 'BOOL':
            val = t[1]
            self.consume('BOOL')
            return PBool(val)
        elif t[0] == 'ID':
            name = t[1]
            self.consume('ID')
            return PVar(name)
        elif t[0] == 'LPAREN':
            self.consume('LPAREN')
            pats = [self.parse_pattern()]
            while self.peek()[0] == 'COMMA':
                self.consume('COMMA')
                pats.append(self.parse_pattern())
            self.consume('RPAREN')
            if len(pats) == 1:
                return pats[0]
            return PTuple(pats)
        elif t[0] == 'LBRACK':
            self.consume('LBRACK')
            patlist = []
            if self.peek()[0] != 'RBRACK':
                patlist.append(self.parse_pattern())
                while self.peek()[0] == 'COMMA':
                    self.consume('COMMA')
                    patlist.append(self.parse_pattern())
            self.consume('RBRACK')
            return PList(patlist)
        else:
            raise ValueError(f"Unexpected token {t} in pattern.")


def evaluate_sml_expression(sml_code: str, env: Dict[str, Any]) -> Any:
    parser = SMLParser(sml_code)
    ast = parser.parse()
    return ast.eval(env)


#########################################################################
# Example usage
#########################################################################
if __name__ == "__main__":
    # Example code:
    code = "let val x = 10 val y = 20 in if x < y then [x,y] else [] end"
    print("Code:", code)
    result = evaluate_sml_expression(code, {})
    print("Result:", result)  # Expect [10,20]

    # A function definition and application:
    # (fn (a,b) => a+b) (10,20)
    code = "(fn (a,b) => a+b) (10,20)"
    print("Code:", code)
    result = evaluate_sml_expression(code, {})
    print("Result:", result)  # Expect 30

    # Case expression:
    # case [1,2,3] of [] => 0 | [x,y,z] => x+y+z | _ => 0-1
    code = "case [1,2,3] of [] => 0 | [x,y,z] => x+y+z | _ => 0-1"
    print("Code:", code)
    result = evaluate_sml_expression(code, {})
    print("Result:", result)  # Expect 6

    # Let binding with pattern match:
    # let val (a,b) = (10,30) in a*b end
    code = "let val (a,b) = (10,30) in a*b end"
    print("Code:", code)
    result = evaluate_sml_expression(code, {})
    print("Result:", result)  # Expect 300
