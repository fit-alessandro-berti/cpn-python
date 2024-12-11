from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Union


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
    A token has a value (of a type from its place's colorset).
    Multiplicity is often represented in the marking itself, but we keep it here if needed.
    """
    def __init__(self, value: Any, multiplicity: int = 1):
        self.value = value
        self.multiplicity = multiplicity

    def __repr__(self):
        return f"Token(value={self.value}, multiplicity={self.multiplicity})"


class Multiset:
    """
    Represents a multiset of tokens. Internally we can use a Counter.
    Keys will be token values, and counts their multiplicity.
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
    Holds a name and a colorset for token type checking.
    """
    def __init__(self, name: str, colorset: ColorSet):
        self.name = name
        self.colorset = colorset

    def __repr__(self):
        return f"Place(name={self.name}, colorset={self.colorset.__class__.__name__})"


class Expression(ABC):
    """
    Abstract base class for arc and guard expressions.
    An expression can be evaluated given a variable binding.
    """
    @abstractmethod
    def evaluate(self, binding: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def variables(self) -> List[str]:
        """
        Return a list of variable names used in this expression.
        """
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
    Represents a function or operation applied to sub-expressions.
    For example, could be (x + 1) or a tuple construction (x, y).
    """
    def __init__(self, func, args: List[Expression]):
        # func: a Python callable that takes a list of evaluated arguments
        # args: list of Expressions
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
    A guard is essentially a boolean expression that must evaluate to True
    for the transition to be enabled.
    We'll inherit from Expression for uniformity, but expect a boolean result.
    """
    pass


class Transition:
    """
    Represents a transition with a name, guard, and a list of variables.
    The variables define the domain of possible bindings that will be tested.
    """
    def __init__(self, name: str, guard: Optional[Guard] = None, variables: List[str] = None):
        self.name = name
        self.guard = guard
        self.variables = variables if variables else []

    def __repr__(self):
        return f"Transition(name={self.name}, guard={self.guard}, variables={self.variables})"


class Arc:
    """
    Represents a directed arc between a place and a transition (or vice versa).
    The expression determines which tokens are moved.
    """
    def __init__(self, source: Union[Place, Transition], target: Union[Place, Transition], expression: Expression):
        self.source = source
        self.target = target
        self.expression = expression

    def __repr__(self):
        return f"Arc(source={self.source.name}, target={self.target.name}, expression={self.expression})"


class CPN:
    """
    The main class representing a Coloured Petri Net.
    It holds places, transitions, arcs, and the initial marking.
    """
    def __init__(self):
        self.places: List[Place] = []
        self.transitions: List[Transition] = []
        self.arcs: List[Arc] = []
        # Marking as a dict: {Place: Multiset_of_values}
        self.initial_marking: Dict[Place, Multiset] = {}

    def add_place(self, place: Place, initial_tokens: Optional[List[Any]] = None):
        self.places.append(place)
        if initial_tokens is None:
            initial_tokens = []
        # Check if tokens conform to the place's colorset
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
        """
        Check if transition t is enabled under a given variable binding.
        This involves:
        1. Evaluating guard.
        2. Checking if all input places have tokens required by input arc expressions.
        """
        # Check guard
        if t.guard:
            guard_val = t.guard.evaluate(binding)
            if guard_val is not True:
                return False

        # Check input arcs
        for arc in self.get_input_arcs(t):
            req_tokens = arc.expression.evaluate(binding)
            # req_tokens could be a single value or a collection.
            # We'll assume a single token value or a list of token values.
            # Convert to a multiset:
            if isinstance(req_tokens, list):
                needed = Multiset(req_tokens)
            else:
                needed = Multiset([req_tokens])
            if not self.initial_marking[arc.source].contains(needed):
                return False

        return True

    def fire_transition(self, t: Transition, binding: Dict[str, Any]):
        """
        Fire the transition t under the given binding.
        This removes tokens from input places and adds tokens to output places.
        """
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
                # Check that val is member of the place's colorset
                if not arc.target.colorset.is_member(val):
                    raise TypeError(f"Produced token {val} not in colorset of place {arc.target.name}")
                self.initial_marking[arc.target].add(val, cnt)

    def __repr__(self):
        return (f"CPN(\n  Places={self.places},\n  Transitions={self.transitions},"
                f"\n  Arcs={self.arcs},\n  InitialMarking={self.initial_marking}\n)")


if __name__ == "__main__":
    # Example usage:

    # Define a simple CPN with one place (holding integers) and one transition.
    int_cs = IntegerColorSet()
    p = Place("P", int_cs)
    t = Transition("T")

    # Arc expressions: from P to T: a variable expression 'x'
    # from T to P: a function expression 'x+1'
    # Let's define variables and guard:
    var_x = VariableExpression("x")
    inc_x = FunctionExpression(lambda a: a+1, [var_x])  # x+1

    # Create a small net
    net = CPN()
    net.add_place(p, initial_tokens=[0])  # Marking: P has one token with value 0
    t.variables = ["x"]  # transition variable
    net.add_transition(t)

    # Input arc: P -> T
    arc_in = Arc(p, t, var_x)
    # Output arc: T -> P
    arc_out = Arc(t, p, inc_x)

    net.add_arc(arc_in)
    net.add_arc(arc_out)

    # Try firing the transition with binding x=0
    binding = {"x": 0}
    if net.is_enabled(t, binding):
        print("Transition T is enabled with x=0.")
        net.fire_transition(t, binding)
        print("After firing:", net.initial_marking)
    else:
        print("Transition T is not enabled with x=0.")
