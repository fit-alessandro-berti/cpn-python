from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Union

############################################################
# 1. Core CPN Object Model
############################################################

class ColorSet(ABC):
    """
    Abstract base class for representing color sets.
    Color sets define the type and domain of tokens.
    """
    @abstractmethod
    def is_member(self, value: Any) -> bool:
        """
        Check if a given value is a member of this color set.
        """
        pass

class IntegerColorSet(ColorSet):
    """
    A simple integer color set as an example.
    """
    def is_member(self, value: Any) -> bool:
        return isinstance(value, int)

class StringColorSet(ColorSet):
    """
    A simple string color set as an example.
    """
    def is_member(self, value: Any) -> bool:
        return isinstance(value, str)

class Token:
    """
    Represents a colored token.
    """
    def __init__(self, value: Any, multiplicity: int = 1):
        self.value = value
        self.multiplicity = multiplicity

    def __repr__(self):
        return f"Token(value={self.value}, multiplicity={self.multiplicity})"

class Multiset:
    """
    Represents a multiset of tokens using a Counter internally.
    """
    def __init__(self, initial=None):
        if initial is None:
            initial = []
        self._multiset = Counter(initial)

    def add(self, value: Any, count: int = 1):
        self._multiset[value] += count

    def remove(self, value: Any, count: int = 1):
        if self._multiset[value] < count:
            raise ValueError("Not enough tokens to remove.")
        self._multiset[value] -= count
        if self._multiset[value] <= 0:
            del self._multiset[value]

    def contains(self, other: 'Multiset') -> bool:
        """
        Check if this multiset contains at least all tokens from 'other'.
        """
        for val, cnt in other._multiset.items():
            if self._multiset[val] < cnt:
                return False
        return True

    def __repr__(self):
        return f"Multiset({dict(self._multiset)})"

    def items(self):
        return self._multiset.items()

    def copy(self):
        new_ms = Multiset()
        new_ms._multiset = self._multiset.copy()
        return new_ms

class Place:
    """
    Represents a place in the CPN.
    """
    def __init__(self, name: str, colorset: ColorSet):
        self.name = name
        self.colorset = colorset

    def __repr__(self):
        return f"Place(name={self.name}, colorset={self.colorset.__class__.__name__})"

class Expression(ABC):
    """
    Abstract base class for arc and guard expressions.
    """
    @abstractmethod
    def evaluate(self, binding: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def variables(self) -> List[str]:
        pass

class VariableExpression(Expression):
    def __init__(self, var_name: str):
        self.var_name = var_name

    def evaluate(self, binding: Dict[str, Any]) -> Any:
        return binding[self.var_name]

    def variables(self) -> List[str]:
        return [self.var_name]

    def __repr__(self):
        return f"VariableExpression({self.var_name})"

class ConstantExpression(Expression):
    def __init__(self, value: Any):
        self.value = value

    def evaluate(self, binding: Dict[str, Any]) -> Any:
        return self.value

    def variables(self) -> List[str]:
        return []

    def __repr__(self):
        return f"ConstantExpression({self.value})"

class FunctionExpression(Expression):
    """
    A generic functional expression. E.g., (x + 1).
    """
    def __init__(self, func, args: List[Expression]):
        self.func = func
        self.args = args

    def evaluate(self, binding: Dict[str, Any]) -> Any:
        evaluated_args = [arg.evaluate(binding) for arg in self.args]
        return self.func(*evaluated_args)

    def variables(self) -> List[str]:
        vars_all = []
        for arg in self.args:
            vars_all.extend(arg.variables())
        return vars_all

    def __repr__(self):
        return f"FunctionExpression({self.func.__name__}, args={self.args})"

class Guard(Expression):
    """
    A guard is a boolean expression.
    """
    pass

class Transition:
    """
    Represents a transition with a name, guard, and variables.
    """
    def __init__(self, name: str, guard: Optional[Guard] = None, variables: List[str] = None):
        self.name = name
        self.guard = guard
        self.variables = variables if variables else []

    def __repr__(self):
        return f"Transition(name={self.name}, guard={self.guard}, variables={self.variables})"

class Arc:
    """
    Represents a directed arc.
    """
    def __init__(self, source: Union[Place, Transition], target: Union[Place, Transition], expression: Expression):
        self.source = source
        self.target = target
        self.expression = expression

    def __repr__(self):
        return f"Arc(source={self.source.name}, target={self.target.name}, expression={self.expression})"

class CPN:
    """
    Represents a Coloured Petri Net.
    """
    def __init__(self):
        self.places: List[Place] = []
        self.transitions: List[Transition] = []
        self.arcs: List[Arc] = []
        self.initial_marking: Dict[Place, Multiset] = {}

    def add_place(self, place: Place, initial_tokens: Optional[List[Any]] = None):
        self.places.append(place)
        if initial_tokens is None:
            initial_tokens = []
        for token_val in initial_tokens:
            if not place.colorset.is_member(token_val):
                raise TypeError(f"Token value {token_val} not in colorset for place {place.name}.")
        self.initial_marking[place] = Multiset(initial_tokens)

    def add_transition(self, transition: Transition):
        self.transitions.append(transition)

    def add_arc(self, arc: Arc):
        self.arcs.append(arc)

    def get_input_arcs(self, t: Transition) -> List[Arc]:
        return [a for a in self.arcs if a.target == t and isinstance(a.source, Place)]

    def get_output_arcs(self, t: Transition) -> List[Arc]:
        return [a for a in self.arcs if a.source == t and isinstance(a.target, Place)]

    def is_enabled(self, t: Transition, binding: Dict[str, Any]) -> bool:
        # Check guard
        if t.guard:
            guard_val = t.guard.evaluate(binding)
            if guard_val is not True:
                return False

        # Check input arcs
        for arc in self.get_input_arcs(t):
            req_tokens = arc.expression.evaluate(binding)
            if isinstance(req_tokens, list):
                needed = Multiset(req_tokens)
            else:
                needed = Multiset([req_tokens])
            if not self.initial_marking[arc.source].contains(needed):
                return False
        return True

    def fire_transition(self, t: Transition, binding: Dict[str, Any]):
        # Remove tokens from input places
        for arc in self.get_input_arcs(t):
            req_tokens = arc.expression.evaluate(binding)
            if isinstance(req_tokens, list):
                needed = Multiset(req_tokens)
            else:
                needed = Multiset([req_tokens])

            for val, cnt in needed.items():
                self.initial_marking[arc.source].remove(val, cnt)

        # Add tokens to output places
        for arc in self.get_output_arcs(t):
            prod_tokens = arc.expression.evaluate(binding)
            if isinstance(prod_tokens, list):
                prod = Multiset(prod_tokens)
            else:
                prod = Multiset([prod_tokens])

            for val, cnt in prod.items():
                if not arc.target.colorset.is_member(val):
                    raise TypeError(f"Produced token {val} not in colorset of place {arc.target.name}")
                self.initial_marking[arc.target].add(val, cnt)

    def __repr__(self):
        return (f"CPN(\n  Places={self.places},\n  Transitions={self.transitions},"
                f"\n  Arcs={self.arcs},\n  InitialMarking={self.initial_marking}\n)")


############################################################
# 2. SML Parsing and Evaluation
############################################################

# For simplicity, let's define a very basic SML-like expression syntax.
# In a real scenario, you'd use a proper SML parser. Here we create a mock parser for demonstration.
#
# We will support a small subset of SML expressions:
#   * Integer constants (e.g., 42)
#   * Variables (e.g., x)
#   * Binary operations on integers (+, -, *, =)
#   * If-then-else expressions: if <bool_expr> then <expr> else <expr>
#
# This is just a minimal stand-in. A full SML parser is beyond this demonstration.

# We'll define an AST for our small SML subset:

class SmlAst(ABC):
    @abstractmethod
    def eval(self, env: Dict[str, Any]) -> Any:
        pass

class SmlInt(SmlAst):
    def __init__(self, value: int):
        self.value = value

    def eval(self, env: Dict[str, Any]) -> Any:
        return self.value

    def __repr__(self):
        return f"SmlInt({self.value})"

class SmlVar(SmlAst):
    def __init__(self, name: str):
        self.name = name

    def eval(self, env: Dict[str, Any]) -> Any:
        return env[self.name]

    def __repr__(self):
        return f"SmlVar({self.name})"

class SmlBinOp(SmlAst):
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
        elif self.op == '=':
            return lval == rval
        else:
            raise ValueError(f"Unknown operator: {self.op}")

    def __repr__(self):
        return f"SmlBinOp({self.op}, {self.left}, {self.right})"

class SmlIfThenElse(SmlAst):
    def __init__(self, cond: SmlAst, then_branch: SmlAst, else_branch: SmlAst):
        self.cond = cond
        self.then_branch = then_branch
        self.else_branch = else_branch

    def eval(self, env: Dict[str, Any]) -> Any:
        cond_val = self.cond.eval(env)
        if cond_val:
            return self.then_branch.eval(env)
        else:
            return self.else_branch.eval(env)

    def __repr__(self):
        return f"SmlIfThenElse({self.cond}, {self.then_branch}, {self.else_branch})"

# A simple tokenizer and parser for a very tiny SML-like subset:
# Grammar (informal):
#   Expr := IfExpr | EqExpr
#   IfExpr := 'if' Expr 'then' Expr 'else' Expr
#   EqExpr := AddExpr ('=' AddExpr)?
#   AddExpr := MulExpr (('+'|'-') MulExpr)*
#   MulExpr := Primary ( '*' Primary )*
#   Primary := INT | VAR | '(' Expr ')'
#
# This is just for demonstration. A real SML parser would be much more complex.

import re

class SMLParser:
    def __init__(self, text: str):
        self.tokens = self.tokenize(text)
        self.pos = 0

    def tokenize(self, text: str):
        token_spec = [
            ("IF", r'if'),
            ("THEN", r'then'),
            ("ELSE", r'else'),
            ("INT", r'\d+'),
            ("VAR", r'[a-zA-Z_]\w*'),
            ("OP", r'[=+\-*]'),
            ("LPAREN", r'\('),
            ("RPAREN", r'\)'),
            ("SKIP", r'[ \t\n]+'),
        ]
        token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_spec)
        tokens = []
        for m in re.finditer(token_regex, text):
            kind = m.lastgroup
            value = m.group()
            if kind == "INT":
                value = int(value)
            if kind != "SKIP":
                tokens.append((kind, value))
        return tokens

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, kind=None):
        t = self.peek()
        if t is None:
            raise ValueError("Unexpected end of input")
        if kind and t[0] != kind:
            raise ValueError(f"Expected {kind}, got {t[0]}")
        self.pos += 1
        return t

    def parse_expr(self):
        # Top-level expression is if-expr or equality expr
        return self.parse_if_expr()

    def parse_if_expr(self):
        # if-expr = 'if' expr 'then' expr 'else' expr
        if self.peek() and self.peek()[0] == 'IF':
            self.consume('IF')
            cond = self.parse_expr()
            self.consume('THEN')
            then_expr = self.parse_expr()
            self.consume('ELSE')
            else_expr = self.parse_expr()
            return SmlIfThenElse(cond, then_expr, else_expr)
        else:
            return self.parse_eq_expr()

    def parse_eq_expr(self):
        # eq-expr = add-expr ('=' add-expr)?
        left = self.parse_add_expr()
        if self.peek() and self.peek()[0] == 'OP' and self.peek()[1] == '=':
            self.consume('OP')
            right = self.parse_add_expr()
            return SmlBinOp('=', left, right)
        return left

    def parse_add_expr(self):
        # add-expr = mul-expr (('+'|'-') mul-expr)*
        expr = self.parse_mul_expr()
        while self.peek() and self.peek()[0] == 'OP' and self.peek()[1] in ('+', '-'):
            op = self.consume('OP')[1]
            right = self.parse_mul_expr()
            expr = SmlBinOp(op, expr, right)
        return expr

    def parse_mul_expr(self):
        # mul-expr = primary ('*' primary)*
        expr = self.parse_primary()
        while self.peek() and self.peek()[0] == 'OP' and self.peek()[1] == '*':
            self.consume('OP')
            right = self.parse_primary()
            expr = SmlBinOp('*', expr, right)
        return expr

    def parse_primary(self):
        t = self.peek()
        if t is None:
            raise ValueError("Unexpected end of input in primary")
        if t[0] == 'INT':
            self.consume('INT')
            return SmlInt(t[1])
        elif t[0] == 'VAR':
            self.consume('VAR')
            return SmlVar(t[1])
        elif t[0] == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expr()
            self.consume('RPAREN')
            return expr
        else:
            raise ValueError(f"Unexpected token {t} in primary")

    @staticmethod
    def parse(text: str) -> SmlAst:
        parser = SMLParser(text)
        expr = parser.parse_expr()
        if parser.pos != len(parser.tokens):
            raise ValueError("Extra input after valid expression")
        return expr

# Example integration:
# Given a snippet of SML code, we can parse it into an AST and then evaluate it with a given environment.

def evaluate_sml_expression(sml_code: str, env: Dict[str, Any]) -> Any:
    ast = SMLParser.parse(sml_code)
    return ast.eval(env)


############################################################
# Example usage
############################################################
if __name__ == "__main__":
    # Construct a simple CPN as before
    int_cs = IntegerColorSet()
    p = Place("P", int_cs)
    t = Transition("T")

    var_x = VariableExpression("x")
    inc_x = FunctionExpression(lambda a: a+1, [var_x])

    net = CPN()
    net.add_place(p, initial_tokens=[0])
    t.variables = ["x"]
    net.add_transition(t)

    arc_in = Arc(p, t, var_x)
    arc_out = Arc(t, p, inc_x)
    net.add_arc(arc_in)
    net.add_arc(arc_out)

    binding = {"x": 0}
    if net.is_enabled(t, binding):
        print("Transition T is enabled with x=0.")
        net.fire_transition(t, binding)
        print("After firing:", net.initial_marking)
    else:
        print("Transition T is not enabled with x=0.")

    # SML parsing & evaluation example:
    # SML code: if x = 0 then x+1 else x*2
    sml_code = "if x = 0 then x + 1 else x * 2"
    env = {"x": 0}
    result = evaluate_sml_expression(sml_code, env)
    print(f"SML expression '{sml_code}' with x=0 evaluates to: {result}")

    env = {"x": 5}
    result = evaluate_sml_expression(sml_code, env)
    print(f"SML expression '{sml_code}' with x=5 evaluates to: {result}")
