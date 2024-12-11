from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Union


class ColorSet(ABC):
    """
    Abstract base class for representing color sets.
    Color sets define the type and domain of tokens.
    Each token placed on a place must belong to the place's associated color set.
    """
    @abstractmethod
    def is_member(self, value: Any) -> bool:
        """
        Check if a given value is a member of this color set.
        Must be implemented by subclasses.
        """
        pass


class IntegerColorSet(ColorSet):
    """
    A simple integer color set as an example.
    Only integer values are allowed.
    """
    def is_member(self, value: Any) -> bool:
        return isinstance(value, int)


class StringColorSet(ColorSet):
    """
    A simple string color set as an example.
    Only string values are allowed.
    """
    def is_member(self, value: Any) -> bool:
        return isinstance(value, str)


class Token:
    """
    Represents a colored token.
    A token has a value (of a type from its place's colorset).
    Multiplicity is typically represented in the marking, not in individual tokens,
    so this class is mostly illustrative.
    """
    def __init__(self, value: Any, multiplicity: int = 1):
        self.value = value
        self.multiplicity = multiplicity

    def __repr__(self):
        return f"Token(value={self.value}, multiplicity={self.multiplicity})"


class Multiset:
    """
    Represents a multiset of token values using a Counter internally.
    Each key is a token value, and the value is its multiplicity.
    """
    def __init__(self, initial=None):
        if initial is None:
            initial = []
        self._multiset = Counter(initial)

    def add(self, value: Any, count: int = 1):
        """
        Add 'count' instances of 'value' to the multiset.
        """
        self._multiset[value] += count

    def remove(self, value: Any, count: int = 1):
        """
        Remove 'count' instances of 'value' from the multiset.
        Raises ValueError if there aren't enough instances.
        """
        if self._multiset[value] < count:
            raise ValueError("Not enough tokens to remove.")
        self._multiset[value] -= count
        if self._multiset[value] <= 0:
            del self._multiset[value]

    def contains(self, other: 'Multiset') -> bool:
        """
        Check if this multiset contains at least all tokens from 'other'.
        For each token in 'other', this multiset must have equal or greater multiplicity.
        """
        for val, cnt in other._multiset.items():
            if self._multiset[val] < cnt:
                return False
        return True

    def items(self):
        return self._multiset.items()

    def copy(self):
        new_ms = Multiset()
        new_ms._multiset = self._multiset.copy()
        return new_ms

    def __repr__(self):
        return f"Multiset({dict(self._multiset)})"


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
    An expression can be evaluated given a variable binding (mapping variable names to values).
    """
    @abstractmethod
    def evaluate(self, binding: Dict[str, Any]) -> Any:
        """
        Evaluate this expression under the given variable binding.
        The result is a value (which can be a single token or a collection of tokens).
        """
        pass

    @abstractmethod
    def variables(self) -> List[str]:
        """
        Return a list of variable names used in this expression.
        """
        pass


class VariableExpression(Expression):
    """
    An expression representing a single variable reference.
    The variable's value is taken from the binding.
    """
    def __init__(self, var_name: str):
        self.var_name = var_name

    def evaluate(self, binding: Dict[str, Any]) -> Any:
        if self.var_name not in binding:
            raise KeyError(f"Variable '{self.var_name}' not found in binding.")
        return binding[self.var_name]

    def variables(self) -> List[str]:
        return [self.var_name]

    def __repr__(self):
        return f"VariableExpression({self.var_name})"


class ConstantExpression(Expression):
    """
    An expression representing a constant value (e.g. a number, string, or a token).
    """
    def __init__(self, value: Any):
        self.value = value

    def evaluate(self, binding: Dict[str, Any]) -> Any:
        # Constants do not depend on the binding
        return self.value

    def variables(self) -> List[str]:
        return []

    def __repr__(self):
        return f"ConstantExpression({self.value})"


class FunctionExpression(Expression):
    """
    Represents a function or operation applied to sub-expressions.
    For example, could represent arithmetic (x+1) or more complex operations.
    func: a Python callable that takes the evaluated arguments and returns a result
    args: list of sub-expressions whose values are computed and passed to func
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
        # Just show the func name if possible
        fname = self.func.__name__ if hasattr(self.func, '__name__') else str(self.func)
        return f"FunctionExpression({fname}, args={self.args})"


class Guard(Expression):
    """
    A guard is a boolean expression that must evaluate to True
    for the transition to be enabled.
    In a fully implemented system, guards would be SML expressions.
    Here, it's treated like a normal expression expected to return True or False.
    """
    pass


class Transition:
    """
    Represents a transition with:
    - a name
    - an optional guard (a boolean expression)
    - a list of variables used in associated arc expressions and guards.

    The variables define the domain of possible bindings tested to enable the transition.
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
    The expression determines which tokens are consumed (if from place to transition)
    or produced (if from transition to place).
    """
    def __init__(self, source: Union[Place, Transition], target: Union[Place, Transition], expression: Expression):
        self.source = source
        self.target = target
        self.expression = expression

    def __repr__(self):
        # Show source and target names and the expression
        return f"Arc(source={self.source.name}, target={self.target.name}, expression={self.expression})"


class CPN:
    """
    Represents a Coloured Petri Net (CPN).

    Attributes:
    - places: list of Place objects
    - transitions: list of Transition objects
    - arcs: list of Arc objects
    - initial_marking: a dictionary {Place: Multiset_of_values}

    This class provides methods to add places, transitions, arcs, and to check enabling/firing.
    """

    def __init__(self):
        self.places: List[Place] = []
        self.transitions: List[Transition] = []
        self.arcs: List[Arc] = []
        self.initial_marking: Dict[Place, Multiset] = {}

    def add_place(self, place: Place, initial_tokens: Optional[List[Any]] = None):
        """
        Add a place to the net and set its initial marking.
        Checks that each initial token belongs to the place's color set.
        """
        if initial_tokens is None:
            initial_tokens = []
        # Validate tokens
        for token_val in initial_tokens:
            if not place.colorset.is_member(token_val):
                raise TypeError(f"Token value {token_val} not in colorset for place {place.name}.")
        self.places.append(place)
        self.initial_marking[place] = Multiset(initial_tokens)

    def add_transition(self, transition: Transition):
        """
        Add a transition to the net.
        """
        self.transitions.append(transition)

    def add_arc(self, arc: Arc):
        """
        Add an arc to the net.
        """
        self.arcs.append(arc)

    def get_input_arcs(self, t: Transition) -> List[Arc]:
        """
        Return all arcs that connect places to the given transition (input arcs).
        """
        return [a for a in self.arcs if a.target == t and isinstance(a.source, Place)]

    def get_output_arcs(self, t: Transition) -> List[Arc]:
        """
        Return all arcs that connect the given transition to places (output arcs).
        """
        return [a for a in self.arcs if a.source == t and isinstance(a.target, Place)]

    def is_enabled(self, t: Transition, binding: Dict[str, Any]) -> bool:
        """
        Check if a transition 't' is enabled under the given variable binding.

        Steps:
        1. Evaluate guard. If guard is present and not True, return False.
        2. For each input arc, evaluate the arc expression under 'binding'.
           Check if the required tokens are present in the input place.
           If any required token multiset is not available, return False.
        If all checks pass, return True.
        """
        # Check guard
        if t.guard:
            guard_val = t.guard.evaluate(binding)
            if guard_val is not True:
                return False

        # Check input arcs
        for arc in self.get_input_arcs(t):
            req_tokens = arc.expression.evaluate(binding)

            # Convert to a multiset
            # Arc expression can return a single value or a collection
            if isinstance(req_tokens, list):
                needed = Multiset(req_tokens)
            elif isinstance(req_tokens, tuple):
                # A tuple also can represent multiple tokens
                needed = Multiset(req_tokens)
            else:
                needed = Multiset([req_tokens])

            # Check availability in the marking
            if not self.initial_marking[arc.source].contains(needed):
                return False

        return True

    def fire_transition(self, t: Transition, binding: Dict[str, Any]):
        """
        Fire the transition 't' under the given binding.
        This updates the marking according to the occurrence rule:
        - Remove tokens from input places as defined by input arcs' expressions.
        - Add tokens to output places as defined by output arcs' expressions.

        Raises ValueError if there aren't enough tokens to remove.
        Raises TypeError if produced tokens don't match the target place's color set.
        """
        # Remove tokens from input places
        for arc in self.get_input_arcs(t):
            req_tokens = arc.expression.evaluate(binding)
            if isinstance(req_tokens, list) or isinstance(req_tokens, tuple):
                needed = Multiset(req_tokens)
            else:
                needed = Multiset([req_tokens])

            for val, cnt in needed.items():
                self.initial_marking[arc.source].remove(val, cnt)

        # Add tokens to output places
        for arc in self.get_output_arcs(t):
            prod_tokens = arc.expression.evaluate(binding)
            if isinstance(prod_tokens, list) or isinstance(prod_tokens, tuple):
                prod = Multiset(prod_tokens)
            else:
                prod = Multiset([prod_tokens])

            for val, cnt in prod.items():
                # Check colorset compliance
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
    var_x = VariableExpression("x")
    inc_x = FunctionExpression(lambda a: a+1, [var_x])  # x+1

    # Create a small net
    net = CPN()
    net.add_place(p, initial_tokens=[0])  # P has one token: 0
    t.variables = ["x"]  # The transition T uses variable x
    net.add_transition(t)

    # Add arcs
    arc_in = Arc(p, t, var_x)    # consumes x from P
    arc_out = Arc(t, p, inc_x)   # produces x+1 to P
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

    # This example is straightforward. In more complex scenarios,
    # you would have to integrate a proper SML interpreter and full color set
    # definitions to correctly model the behavior of real-world CPN models.
