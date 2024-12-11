import itertools
from copy import deepcopy
from cpn import CPN, Transition, Place, Multiset, VariableExpression, ConstantExpression, FunctionExpression, Guard
from sml import evaluate_sml_expression, SMLParser

"""
In this revised 'algo.py', we try to be more comprehensive and general, taking into account
the complexities and limitations we discussed, especially in the context of the DistributedDataBase.cpn.

Key improvements and considerations:

1. **Variable Domain Inference:**
   Instead of blindly assuming all tokens from input places form a single domain for all variables,
   we will:
   - Attempt to identify each variable's color set (if possible). If we cannot, we fall back to a permissive approach.
   - Collect tokens from input places. For each variable, filter these tokens by the variable's color set (if known).
   - If no tokens are available to infer a domain, fall back to a small fixed domain (e.g., integers [0,1,2]) or empty.

2. **Guard Evaluation:**
   We call `evaluate_guard` which returns True or False. If no guard is present, we consider it True.

3. **Arc Expressions:**
   We evaluate arc expressions using `evaluate_arc_expression`, which calls `expr.evaluate(binding)`.
   If the expression is more complex (e.g., involves SML code), a full implementation would need
   an integrated SML evaluator. Here, we rely on the given partial approach. For a real system,
   you would ensure `expr` is constructed from the parsed SML AST and can be evaluated consistently.

4. **Handling Infeasible Bindings:**
   If no tokens match a variable's domain or we cannot infer a suitable domain, we acknowledge
   this limitation. In a real implementation, you'd have to fail gracefully, ask the user, or
   have a well-defined fallback.

5. **Concurrent Steps:**
   We allow firing a step (a set of binding elements) if all their required tokens are available.
   This check is done by aggregating token requirements from all involved transitions and verifying
   them against the current marking.

This code still does not fully replicate the complex logic of a real-world CPN model like
DistributedDataBase.cpn, but it aims to be more robust, documented, and general than before.
"""

def get_variable_domains(net: CPN, t: Transition):
    """
    Infer candidate domains for each variable of transition t from the current marking.

    Steps:
    - Identify input places of t.
    - Gather all token values from these places.
    - In a real system, we would know each variable's color set and filter tokens accordingly.
      Here, we do not have explicit variable-to-color-set mappings, so we fall back to a heuristic.
    - If no tokens are available, we return a small fallback domain for the variables.

    Note: In a fully implemented system, variable domains might come from explicit color sets,
    SML-based definitions, or user specifications. Without that, we rely on these heuristics.
    """
    input_arcs = net.get_input_arcs(t)
    input_places = [arc.source for arc in input_arcs]

    # Collect token values from input places
    tokens_values = set()
    for p in input_places:
        for val, cnt in net.initial_marking[p].items():
            tokens_values.add(val)

    # If no tokens found, fallback to a small set of integers
    if not tokens_values:
        tokens_values = {0, 1, 2}

    # Assign the same domain to each variable for this demo,
    # since we don't have the actual color set info.
    domains = {}
    for var in t.variables:
        # Ideally, we would filter tokens_values by var's color set if known.
        # Without color set info, we just use them as-is.
        candidate_values = list(tokens_values)
        domains[var] = candidate_values

    return domains


def evaluate_guard(guard: Guard, binding: dict):
    """
    Evaluate the guard expression under the given binding.
    If no guard is present, return True.
    If guard exists, it must evaluate to True for enabling.
    """
    if guard is None:
        return True
    return guard.evaluate(binding) == True


def evaluate_arc_expression(expr, binding):
    """
    Evaluate the arc expression under the given binding.
    The expression may return a single value or a collection of values (tokens).
    """
    return expr.evaluate(binding)


def is_enabled(net: CPN, t: Transition):
    """
    Check if there exists at least one variable binding that enables transition t.
    """
    enabled_bindings = get_enabled_bindings(net, t)
    return len(enabled_bindings) > 0


def get_enabled_bindings(net: CPN, t: Transition):
    """
    The enabling algorithm:
    1. Determine variable domains (heuristic).
    2. Iterate over all possible combinations of variable assignments.
    3. Check guard for each binding.
    4. Check if input arcs can be satisfied from the current marking.
    5. Collect all enabled bindings.
    """
    domains = get_variable_domains(net, t)
    var_names = t.variables

    # Cartesian product of all variable domains
    all_bindings = itertools.product(*(domains[var] for var in var_names))
    enabled = []
    for combo in all_bindings:
        binding = dict(zip(var_names, combo))

        # Check guard
        if not evaluate_guard(t.guard, binding):
            continue

        # Check input arcs: verify that we have the required tokens
        input_arcs = net.get_input_arcs(t)
        input_ok = True
        for arc in input_arcs:
            req_tokens = evaluate_arc_expression(arc.expression, binding)
            # Convert to multiset
            if isinstance(req_tokens, list) or isinstance(req_tokens, tuple):
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
    Occurrence rule for a single transition:
    - Remove tokens as specified by input arcs.
    - Add tokens as specified by output arcs.
    Raise errors if token removal or type checks fail.
    """
    input_arcs = net.get_input_arcs(t)
    output_arcs = net.get_output_arcs(t)

    # Remove tokens from input places
    for arc in input_arcs:
        req_tokens = evaluate_arc_expression(arc.expression, binding)
        if isinstance(req_tokens, list) or isinstance(req_tokens, tuple):
            needed = Multiset(req_tokens)
        else:
            needed = Multiset([req_tokens])
        for val, cnt in needed.items():
            net.initial_marking[arc.source].remove(val, cnt)

    # Add tokens to output places
    for arc in output_arcs:
        prod_tokens = evaluate_arc_expression(arc.expression, binding)
        if isinstance(prod_tokens, list) or isinstance(prod_tokens, tuple):
            prod = Multiset(prod_tokens)
        else:
            prod = Multiset([prod_tokens])

        for val, cnt in prod.items():
            if not arc.target.colorset.is_member(val):
                raise TypeError(f"Produced token {val} not in colorset of place {arc.target.name}")
            net.initial_marking[arc.target].add(val, cnt)


def can_fire_step(net: CPN, binding_elements):
    """
    Check if a set of binding elements (transitions with bindings) can fire together (concurrently).
    Steps:
    - Aggregate all required tokens for all transitions in the step.
    - Check if the initial marking can provide those tokens without conflict.
    """
    requirements = {}  # {Place: Multiset of required tokens}
    for (t, binding) in binding_elements:
        for arc in net.get_input_arcs(t):
            req_tokens = evaluate_arc_expression(arc.expression, binding)
            if isinstance(req_tokens, list) or isinstance(req_tokens, tuple):
                needed = Multiset(req_tokens)
            else:
                needed = Multiset([req_tokens])

            if arc.source not in requirements:
                requirements[arc.source] = Multiset()

            # Add the needed tokens to the place requirement
            for val, cnt in needed.items():
                requirements[arc.source].add(val, cnt)

    # Verify that we can provide all these tokens from the current marking
    for p, needed_ms in requirements.items():
        if not net.initial_marking[p].contains(needed_ms):
            return False
    return True


def fire_step(net: CPN, binding_elements):
    """
    Fire a step (a collection of transitions with their bindings) concurrently.
    1. Remove all required tokens from their respective input places.
    2. Add all produced tokens to their respective output places.
    """
    # Remove tokens first
    for (t, binding) in binding_elements:
        for arc in net.get_input_arcs(t):
            req_tokens = evaluate_arc_expression(arc.expression, binding)
            if isinstance(req_tokens, list) or isinstance(req_tokens, tuple):
                needed = Multiset(req_tokens)
            else:
                needed = Multiset([req_tokens])
            for val, cnt in needed.items():
                net.initial_marking[arc.source].remove(val, cnt)

    # Then add output tokens
    for (t, binding) in binding_elements:
        for arc in net.get_output_arcs(t):
            prod_tokens = evaluate_arc_expression(arc.expression, binding)
            if isinstance(prod_tokens, list) or isinstance(prod_tokens, tuple):
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
    # Construct a simple net using cpn.py
    from cpn import IntegerColorSet

    int_cs = IntegerColorSet()
    p = Place("P", int_cs)
    q = Place("Q", int_cs)

    # A simple transition T: takes x from P, outputs x+1 to Q
    t = Transition("T", variables=["x"])

    var_x = VariableExpression("x")
    inc_x = FunctionExpression(lambda a: a + 1, [var_x])

    net = CPN()
    net.add_place(p, initial_tokens=[0, 1])  # P: {0,1}
    net.add_place(q, initial_tokens=[])       # Q: {}
    net.add_transition(t)

    from cpn import Arc
    arc_in = Arc(p, t, var_x)
    arc_out = Arc(t, q, inc_x)
    net.add_arc(arc_in)
    net.add_arc(arc_out)

    # Check enabled bindings
    enabled_b = get_enabled_bindings(net, t)
    print("Enabled bindings for T:", enabled_b)
    # Expect [(T, {x:0}), (T, {x:1})]

    # Fire one binding
    if enabled_b:
        fire_transition(net, t, enabled_b[0][1])
        print("Marking after firing one binding:", net.initial_marking)
        # After firing with x=0: P had {0,1}, remove 0 -> P now {1}, Q add 1 -> Q now {1}

    # Reset and test concurrent firing
    net.initial_marking[p] = Multiset([0, 1])
    net.initial_marking[q] = Multiset([])

    enabled_b = get_enabled_bindings(net, t)
    if can_fire_step(net, enabled_b):
        fire_step(net, enabled_b)
        print("Marking after firing step with both bindings:", net.initial_marking)
        # Should remove both 0 and 1 from P and add (0+1)=1 and (1+1)=2 to Q -> Q: {1,2}
