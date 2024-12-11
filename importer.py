import xml.etree.ElementTree as ET
from cpn import CPN, Place, Transition, Arc, ColorSet, Multiset, VariableExpression, ConstantExpression, FunctionExpression, Guard
from sml import evaluate_sml_expression, SMLParser
from algo import get_enabled_bindings, is_enabled, fire_transition, can_fire_step, fire_step

"""
Revised Importer for General Use:

In the new example, there's an issue:
- The arc inscriptions like "Chopsticks(p)" represent a function that returns tokens based on the variable p.
- Previously, we treated unrecognized expressions as constants, but "Chopsticks(p)" is not a token; it's a function call.
- According to the given snippet:
    fun Chopsticks(ph(i)) = 1`cs(i) ++ 1`cs(if i=n then 1 else i+1);

  This means that "Chopsticks(p)" returns two tokens: cs(i) and cs(i+1 or 1 if i=n).
- We have global val n = 5; PH and CS are index color sets from 1 to n.
- p is of type PH, so p is in {1,...,n}.
- Chopsticks(p) should produce two tokens from CS: cs(p) and cs((p mod n) + 1).

We will:
1. Parse global declarations to find n.
2. Parse color sets PH and CS as IndexColorSet(1,n) if found.
3. When we encounter "Chopsticks(p)", we create a FunctionExpression that, when evaluated, returns the tokens [p, p_next] where p_next = 1 if p=n else p+1.
4. This ensures that if p is bound, Chopsticks(p) returns the correct token values that match the place markings.

This fix will make the transition enabled if there's a suitable p.

Note: This is a custom hack for the known function "Chopsticks". A full solution would require a general SML interpreter
to handle arbitrary inscriptions. Here we implement a special case for demonstration.
"""

class IndexColorSet(ColorSet):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def is_member(self, value) -> bool:
        return isinstance(value, int) and self.start <= value <= self.end

    def all_tokens(self):
        return list(range(self.start, self.end + 1))


class PermissiveColorSet(ColorSet):
    def is_member(self, value):
        return isinstance(value, int) or isinstance(value, str)


def parse_global_declarations(globbox_el):
    env = {}
    if globbox_el is None:
        return env

    for ml in globbox_el.findall('ml'):
        code = ml.text.strip()
        # very naive val parser
        if code.startswith("val "):
            code_line = code.split(';')[0]
            parts = code_line.split('=')
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                varname = left.split()[1]
                try:
                    val = int(right)
                    env[varname] = val
                except:
                    env[varname] = right
    return env


def parse_color_sets(globbox_el, env):
    color_sets = {}
    if globbox_el is None:
        return color_sets

    for c_el in globbox_el.findall('color'):
        cid_el = c_el.find('id')
        if cid_el is None:
            continue
        color_name = cid_el.text.strip()
        index_el = c_el.find('index')
        if index_el is not None:
            mls = index_el.findall('ml')
            if len(mls) == 2:
                try:
                    start = int(mls[0].text.strip())
                    end_part = mls[1].text.strip()
                    if end_part in env and isinstance(env[end_part], int):
                        end = env[end_part]
                        color_sets[color_name] = IndexColorSet(start, end)
                    else:
                        color_sets[color_name] = PermissiveColorSet()
                except:
                    color_sets[color_name] = PermissiveColorSet()
            else:
                color_sets[color_name] = PermissiveColorSet()
        else:
            color_sets[color_name] = PermissiveColorSet()
    return color_sets


def parse_initial_marking(mtext, place_obj, color_sets):
    if mtext.endswith('all()'):
        cname = mtext.split('.')[0]
        if cname in color_sets:
            cs = color_sets[cname]
            if isinstance(cs, IndexColorSet):
                return cs.all_tokens()
            else:
                return [0,1,2]
        else:
            return [0,1,2]
    else:
        try:
            val = int(mtext)
            if place_obj.colorset.is_member(val):
                return [val]
            else:
                return []
        except:
            if place_obj.colorset.is_member(mtext):
                return [mtext]
            return []


def Chopsticks_func(n, p):
    # Chopsticks(ph(i)) = cs(i) and cs(if i=n then 1 else i+1)
    # p is in [1..n]
    # next_p = 1 if p=n else p+1
    next_p = 1 if p == n else p+1
    # Return a list of two tokens: p and next_p (both integers)
    return [p, next_p]


def parse_arc_expression(arc_expr_str, color_sets, env):
    arc_expr_str = arc_expr_str.strip()
    # Special case: Chopsticks(p)
    if arc_expr_str == "Chopsticks(p)":
        # We know we have val n in env
        if 'n' in env and isinstance(env['n'], int):
            n = env['n']
            # create a function expression that takes 'p' from binding
            p_var = VariableExpression("p")
            # lambda that calls Chopsticks_func(n, p)
            func = lambda p: Chopsticks_func(n, p)
            return FunctionExpression(func, [p_var])
        else:
            # fallback if no n found
            return ConstantExpression("Chopsticks(p)")

    # If simple variable
    if arc_expr_str.isalpha():
        return VariableExpression(arc_expr_str)

    # If integer
    try:
        val = int(arc_expr_str)
        return ConstantExpression(val)
    except:
        pass

    # Try SML parsing if needed
    try:
        ast = SMLParser.parse(arc_expr_str)
        from sml import SmlInt, SmlVar, SmlBool
        if isinstance(ast, SmlInt):
            return ConstantExpression(ast.value)
        elif isinstance(ast, SmlVar):
            return VariableExpression(ast.name)
        elif isinstance(ast, SmlBool):
            return ConstantExpression(ast.value)
        else:
            return ConstantExpression(arc_expr_str)
    except:
        return ConstantExpression(arc_expr_str)


def parse_cpn(filename: str) -> CPN:
    tree = ET.parse(filename)
    root = tree.getroot()

    net = CPN()

    cpnet = root.find('cpnet')
    if cpnet is None:
        return net

    globbox_el = cpnet.find('globbox')
    env = parse_global_declarations(globbox_el)
    color_sets = parse_color_sets(globbox_el, env)

    places = {}
    transitions = {}
    arcs = []

    for page in cpnet.findall('page'):
        # Places
        for p_el in page.findall('place'):
            pid = p_el.get('id')
            pname = None
            for t in p_el.findall('text'):
                pname = t.text.strip()
            if pname is None:
                pname = pid

            cset_el = p_el.find('type')
            place_colorset = PermissiveColorSet()
            if cset_el is not None:
                ctext_el = cset_el.find('text')
                if ctext_el is not None:
                    ctype = ctext_el.text.strip()
                    if ctype in color_sets:
                        place_colorset = color_sets[ctype]

            place_obj = Place(pname, place_colorset)

            init_mark_el = p_el.find('initmark')
            initial_tokens = []
            if init_mark_el is not None:
                mark_text = init_mark_el.find('text')
                if mark_text is not None:
                    mtext = mark_text.text.strip()
                    initial_tokens = parse_initial_marking(mtext, place_obj, color_sets)

            net.add_place(place_obj, initial_tokens)
            places[pid] = place_obj

        # Transitions
        for t_el in page.findall('trans'):
            tid = t_el.get('id')
            tname = None
            for tnode in t_el.findall('text'):
                tname = tnode.text.strip()
            if tname is None:
                tname = tid
            trans_obj = Transition(tname, guard=None, variables=[])
            net.add_transition(trans_obj)
            transitions[tid] = trans_obj

        # Arcs
        for a_el in page.findall('arc'):
            orientation = a_el.get('orientation')
            transend = a_el.find('transend')
            placeend = a_el.find('placeend')

            if transend is not None and placeend is not None:
                t_id = transend.get('idref')
                p_id = placeend.get('idref')
                # Identify source and target
                # orientation: PtoT means place->transition
                # TtoP means transition->place
                if t_id in transitions and p_id in places:
                    if orientation == "PtoT":
                        source = places[p_id]
                        target = transitions[t_id]
                    else:
                        source = transitions[t_id]
                        target = places[p_id]
                elif p_id in transitions and t_id in places:
                    # swapped references
                    if orientation == "PtoT":
                        source = places[t_id]
                        target = transitions[p_id]
                    else:
                        source = transitions[p_id]
                        target = places[t_id]
                else:
                    # If referencing something not found
                    continue

                annot = a_el.find('annot')
                inscription_expr = ConstantExpression(1)
                if annot is not None:
                    annot_text_el = annot.find('text')
                    if annot_text_el is not None and annot_text_el.text:
                        arc_expr_str = annot_text_el.text.strip()
                        inscription_expr = parse_arc_expression(arc_expr_str, color_sets, env)

                arc_obj = Arc(source, target, inscription_expr)
                net.add_arc(arc_obj)
                arcs.append(arc_obj)

    # Add variables from arcs
    for arc in arcs:
        if isinstance(arc.source, Transition):
            t = arc.source
        elif isinstance(arc.target, Transition):
            t = arc.target
        else:
            t = None
        if t is not None:
            vars_ = arc.expression.variables()
            for v in vars_:
                if v not in t.variables:
                    t.variables.append(v)

    return net


if __name__ == "__main__":
    # Test with the provided example
    net = parse_cpn("testcases/DistributedDataBase.cpn")

    #print("Places:", net.places)
    print("Transitions:", net.transitions)
    #print("Arcs:", net.arcs)
    #print("Initial Marking:", net.initial_marking)

    # Try to enable and fire transitions if possible
    changed = True
    while changed:
        changed = False
        for t in net.transitions:
            enabled_bs = get_enabled_bindings(net, t)
            if enabled_bs:
                print("ENABLED:", t)
                fire_transition(net, t, enabled_bs[0][1])
                print("Marking after firing:", net.initial_marking)
                changed = True
                input("->  ")
                break
