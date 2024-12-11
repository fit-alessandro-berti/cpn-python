import itertools
from copy import deepcopy
from cpn import CPN, Transition, Place, Multiset, VariableExpression, ConstantExpression, FunctionExpression, Guard
from sml import evaluate_sml_expression, SMLParser


# Note:
# We assume that:
#   - Each variable in a transition has an associated color set.
#   - The CPN model associates variables with color sets.
#     For simplicity, we assume that we can infer the color set of a variable
#     from the input places or from a provided mapping.
#   - The color sets are small or finite and that we can enumerate their members.
#     In a real implementation, you'd need a method to retrieve the domain of
#     each color set or have it defined explicitly.
#
# Since the user says we don't need to re-implement anything and can rely on the current implementations,
# we will make a simplifying assumption:
# We will try to infer possible values for the variables from the tokens present in the input places' markings.
# This is a heuristic: for each variable, we consider all tokens in input places that match the variable's color set.
# This is not a full solution for arbitrary infinite or large color sets, but will serve as a demonstration.

def get_variable_domains(net: CPN, t: Transition):
    """
    Infer possible domains for the transition variables by looking at the tokens in the input places.
    This is a heuristic approach: gather all token values from input places that match variable color sets.
    """
    # Identify input places
    input_arcs = net.get_input_arcs(t)
    input_places = [arc.source for arc in input_arcs]

    # For simplicity, assume each variable corresponds to a single color set.
    # We'll guess variable-to-colorset mapping by checking the arcs' expressions.
    # In a more complete system, the transition would explicitly provide color set information.

    # Here we assume all variables share a single color set domain or we can guess it from the places.
    # A more advanced implementation would store variable->colorset in the model.
    # We'll just collect all token values from input places as potential candidates.

    tokens_values = set()
    for p in input_places:
        for val, cnt in net.initial_marking[p].items():
            # check if val belongs to a colorset that might be relevant
            # This is a heuristic. We'll just collect them all.
            tokens_values.add(val)

    # If we have multiple variables, we must assume all share the same domain for this demo.
    # A real implementation would differentiate based on known variable color sets.
    domains = {}
    for var in t.variables:
        # In a real scenario, we know var's colorset from the CPN model and could filter `tokens_values`.
        # Let's assume all tokens_values are suitable candidates.
        # If empty, let's just assume integers in a small range as fallback.
        candidate_values = list(tokens_values) if tokens_values else list(range(0, 3))
        domains[var] = candidate_values
    return domains


def evaluate_guard(guard: Guard, binding: dict):
    if guard is None:
        return True
    return guard.evaluate(binding) == True


def evaluate_arc_expression(expr, binding):
    return expr.evaluate(binding)


def is_enabled(net: CPN, t: Transition):
    """
    Determine if there is at least one binding that enables t.
    """
    enabled_bindings = get_enabled_bindings(net, t)
    return len(enabled_bindings) > 0


def get_enabled_bindings(net: CPN, t: Transition):
    """
    The enabling algorithm:
    1. Generate candidate bindings from variable domains.
    2. Check guard.
    3. Check input arcs.
    """
    domains = get_variable_domains(net, t)
    var_names = t.variables
    all_bindings = itertools.product(*(domains[var] for var in var_names))
    enabled = []
    for combo in all_bindings:
        binding = dict(zip(var_names, combo))

        # Check guard
        if not evaluate_guard(t.guard, binding):
            continue

        # Check input arcs
        input_arcs = net.get_input_arcs(t)
        input_ok = True
        for arc in input_arcs:
            req_tokens = evaluate_arc_expression(arc.expression, binding)
            if isinstance(req_tokens, list):
                needed = Multiset(req_tokens)
            else:
                needed = Multiset([req_tokens])
            if not net.initial_marking[arc.source].contains(needed):
                input_ok = False
                break
        if input_ok:
            enabled.append((t, binding))
    return enabled


def fire_transition(net: CPN, t: Transition, binding: dict):
    """
    Occurrence rule:
    For each input arc: remove required tokens.
    For each output arc: add produced tokens.
    """
    input_arcs = net.get_input_arcs(t)
    output_arcs = net.get_output_arcs(t)

    # Remove tokens
    for arc in input_arcs:
        req_tokens = evaluate_arc_expression(arc.expression, binding)
        if isinstance(req_tokens, list):
            needed = Multiset(req_tokens)
        else:
            needed = Multiset([req_tokens])
        for val, cnt in needed.items():
            net.initial_marking[arc.source].remove(val, cnt)

    # Add tokens
    for arc in output_arcs:
        prod_tokens = evaluate_arc_expression(arc.expression, binding)
        if isinstance(prod_tokens, list):
            prod = Multiset(prod_tokens)
        else:
            prod = Multiset([prod_tokens])
        for val, cnt in prod.items():
            if not arc.target.colorset.is_member(val):
                raise TypeError(f"Produced token {val} not in colorset of place {arc.target.name}")
            net.initial_marking[arc.target].add(val, cnt)


def can_fire_step(net: CPN, binding_elements):
    """
    Check if a set of binding elements (transitions with bindings) can fire together.
    This means that the total tokens they require from each place do not exceed that place's marking.
    """
    # Aggregate the required tokens
    requirements = {}  # {Place: Multiset}
    for (t, binding) in binding_elements:
        for arc in net.get_input_arcs(t):
            req_tokens = evaluate_arc_expression(arc.expression, binding)
            if isinstance(req_tokens, list):
                needed = Multiset(req_tokens)
            else:
                needed = Multiset([req_tokens])
            if arc.source not in requirements:
                requirements[arc.source] = Multiset()
            # sum up needed tokens
            for val, cnt in needed.items():
                requirements[arc.source].add(val, cnt)

    # Check if each requirement fits into the current marking
    for p, needed_ms in requirements.items():
        if not net.initial_marking[p].contains(needed_ms):
            return False
    return True


def fire_step(net: CPN, binding_elements):
    """
    Fire all given binding elements concurrently.
    This means removing tokens for all input arcs and then adding tokens for all output arcs.
    """
    # First remove all input tokens
    for (t, binding) in binding_elements:
        for arc in net.get_input_arcs(t):
            req_tokens = evaluate_arc_expression(arc.expression, binding)
            if isinstance(req_tokens, list):
                needed = Multiset(req_tokens)
            else:
                needed = Multiset([req_tokens])
            for val, cnt in needed.items():
                net.initial_marking[arc.source].remove(val, cnt)
    # Then produce output tokens
    for (t, binding) in binding_elements:
        for arc in net.get_output_arcs(t):
            prod_tokens = evaluate_arc_expression(arc.expression, binding)
            if isinstance(prod_tokens, list):
                prod = Multiset(prod_tokens)
            else:
                prod = Multiset([prod_tokens])
            for val, cnt in prod.items():
                if not arc.target.colorset.is_member(val):
                    raise TypeError(f"Produced token {val} not in colorset of place {arc.target.name}")
                net.initial_marking[arc.target].add(val, cnt)


#####################################################################
# TEST CASES
#####################################################################

if __name__ == "__main__":
    # We'll construct a simple net using the classes from cpn.py:
    from cpn import IntegerColorSet

    int_cs = IntegerColorSet()
    p = Place("P", int_cs)
    q = Place("Q", int_cs)

    # We'll create a transition T that increments a token from P and puts it in Q
    t = Transition("T", variables=["x"])

    # x is from int_cs implicitly
    var_x = VariableExpression("x")
    inc_x = FunctionExpression(lambda a: a + 1, [var_x])

    net = CPN()
    net.add_place(p, initial_tokens=[0, 1])  # place P has two tokens: 0 and 1
    net.add_place(q, initial_tokens=[])
    net.add_transition(t)

    from cpn import Arc

    # Arc from P to T consumes a token "x"
    arc_in = Arc(p, t, var_x)
    # Arc from T to Q produces "x+1"
    arc_out = Arc(t, q, inc_x)
    net.add_arc(arc_in)
    net.add_arc(arc_out)

    # Test enabling
    enabled_b = get_enabled_bindings(net, t)
    print("Enabled bindings for T:", enabled_b)
    # Expect binding with x=0 and x=1 both enabled

    # Fire a single binding
    if enabled_b:
        fire_transition(net, t, enabled_b[0][1])
        print("Marking after firing one binding:", net.initial_marking)
        # Should have removed one token from P and added its increment in Q

    # Reset net
    net.initial_marking[p] = Multiset([0, 1])
    net.initial_marking[q] = Multiset([])

    # Test step occurrence with both bindings at once (concurrently)
    # Check if we can fire both x=0 and x=1 at the same time
    enabled_b = get_enabled_bindings(net, t)
    if can_fire_step(net, enabled_b):
        fire_step(net, enabled_b)
        print("Marking after firing step with both bindings:", net.initial_marking)
        # Should have taken both 0 and 1 from P and put 1 and 2 in Q
