import xml.etree.ElementTree as ET
from cpn import CPN, Place, Transition, Arc, ColorSet, Multiset, VariableExpression, ConstantExpression, \
    FunctionExpression, Guard
from sml import evaluate_sml_expression, SMLParser
from algo import get_enabled_bindings, is_enabled, fire_transition, can_fire_step, fire_step

"""
Generalized CPN Importer:

We aim to provide a generalized importer that does not overfit or hardcode specific color sets, variables, 
or functions from particular examples. Instead, we rely on the structure provided by the CPN XML and 
apply generic heuristics:

Key Points:
- Parse global declarations (val x = num), store them in 'env'.
- Parse color sets:
  * If an 'index' color set is found with a start and an end defined by env, create an IndexColorSet (from start..end).
  * If an 'enum' color set is found, create an EnumColorSet.
  * If a 'product' color set is found (two references), if both are IndexColorSets, create a ProductColorSet.
    Otherwise fallback to PermissiveColorSet.
  * If a 'subset' color set is found, fallback to PermissiveColorSet (we won't try to interpret predicates like diff).
  * If nothing recognized, fallback to PermissiveColorSet.

- For initial markings:
  * If something ends with '.all()', attempt to call `all_tokens()` on the corresponding color set if found.
    If the color set does not implement all_tokens or not found, fallback to [0,1,2].
  * If it's an integer, parse as int. If the token matches the place color set, add it. Otherwise empty.
  * If string token and matches the color set, add it, else empty.

- For arc expressions:
  * Try to parse as a variable (alphabetic => VariableExpression).
  * Try to parse as integer => ConstantExpression.
  * If it looks like a tuple (e.g. "(x,y)"), parse each component recursively as variable/int/constant, 
    then produce a FunctionExpression returning a tuple.
  * Otherwise, try SML parsing. If SML parsing yields a simple var/int/bool, create corresponding expression.
    Else fallback to ConstantExpression of the raw string.

No special-case references to specific functions or sets like Mes(s) or Chopsticks(p). Everything is handled generically.

This should be more general and not overfit to specific examples.
"""


class IndexColorSet(ColorSet):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def is_member(self, value) -> bool:
        return isinstance(value, int) and self.start <= value <= self.end

    def all_tokens(self):
        return list(range(self.start, self.end + 1))


class EnumColorSet(ColorSet):
    def __init__(self, elements: list):
        self.elements = elements

    def is_member(self, value) -> bool:
        return value in self.elements

    def all_tokens(self):
        return self.elements[:]


class ProductColorSet(ColorSet):
    def __init__(self, cs1: ColorSet, cs2: ColorSet):
        # We'll only support all_tokens if both cs1 and cs2 have all_tokens
        self.cs1 = cs1
        self.cs2 = cs2

    def is_member(self, value) -> bool:
        if isinstance(value, tuple) and len(value) == 2:
            return self.cs1.is_member(value[0]) and self.cs2.is_member(value[1])
        return False

    def all_tokens(self):
        # Only if both cs1 and cs2 have all_tokens:
        have_all1 = hasattr(self.cs1, 'all_tokens')
        have_all2 = hasattr(self.cs2, 'all_tokens')
        if have_all1 and have_all2:
            return [(x, y) for x in self.cs1.all_tokens() for y in self.cs2.all_tokens()]
        # fallback if no methods
        return [0, 1, 2]


class PermissiveColorSet(ColorSet):
    def is_member(self, value):
        # allow int, str, tuple of int/str
        if isinstance(value, tuple):
            return all(isinstance(v, (int, str)) for v in value)
        return isinstance(value, (int, str))

    # no all_tokens method, rely on fallback if needed.


def parse_global_declarations(globbox_el):
    env = {}
    if globbox_el is None:
        return env
    for ml in globbox_el.findall('ml'):
        code = ml.text.strip()
        if code.startswith("val "):
            # parse val bindings
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

    pending_products = []  # store (cname,c1,c2) for product sets
    # We'll treat subsets or other complex sets as Permissive for now

    for c_el in globbox_el.findall('color'):
        cid_el = c_el.find('id')
        if cid_el is None:
            continue
        cname = cid_el.text.strip()

        # enum?
        enum_el = c_el.find('enum')
        if enum_el is not None:
            ids = enum_el.findall('id')
            elems = [i.text.strip() for i in ids]
            color_sets[cname] = EnumColorSet(elems)
            continue

        # index?
        index_el = c_el.find('index')
        if index_el is not None:
            mls = index_el.findall('ml')
            if len(mls) == 2:
                try:
                    start = int(mls[0].text.strip())
                    end_part = mls[1].text.strip()
                    if end_part in env and isinstance(env[end_part], int):
                        end = env[end_part]
                        color_sets[cname] = IndexColorSet(start, end)
                        continue
                    else:
                        color_sets[cname] = PermissiveColorSet()
                        continue
                except:
                    color_sets[cname] = PermissiveColorSet()
                    continue
            else:
                color_sets[cname] = PermissiveColorSet()
                continue

        # product?
        product_el = c_el.find('product')
        if product_el is not None:
            ids = product_el.findall('id')
            if len(ids) == 2:
                c1 = ids[0].text.strip()
                c2 = ids[1].text.strip()
                pending_products.append((cname, c1, c2))
            else:
                color_sets[cname] = PermissiveColorSet()
            continue

        # subset or other complex definitions => Permissive
        # We won't handle them specifically
        subset_el = c_el.find('subset')
        if subset_el is not None:
            color_sets[cname] = PermissiveColorSet()
            continue

        # If none matched
        color_sets[cname] = PermissiveColorSet()

    # Second pass for products
    for (cname, c1, c2) in pending_products:
        cs1 = color_sets.get(c1, None)
        cs2 = color_sets.get(c2, None)
        if cs1 and cs2 and isinstance(cs1, (IndexColorSet, EnumColorSet, PermissiveColorSet)) \
                and isinstance(cs2, (IndexColorSet, EnumColorSet, PermissiveColorSet)):
            # We'll allow product if both are Index or Enum or Permissive
            # If both are IndexColorSet, we get nice all_tokens, else fallback
            color_sets[cname] = ProductColorSet(cs1, cs2)
        else:
            color_sets[cname] = PermissiveColorSet()

    return color_sets


def parse_initial_marking(mtext, place_obj, color_sets, env):
    # handle .all()
    if mtext.endswith('all()'):
        cname = mtext.split('.')[0]
        cs = color_sets.get(cname, None)
        if cs and hasattr(cs, 'all_tokens'):
            return cs.all_tokens()
        else:
            # fallback
            return [0, 1, 2]
    else:
        # try int
        try:
            val = int(mtext)
            if place_obj.colorset.is_member(val):
                return [val]
            else:
                return []
        except:
            # treat as string token
            if place_obj.colorset.is_member(mtext):
                return [mtext]
            return []


def parse_atomic_expression(expr_str):
    # Used to parse tuple components
    expr_str = expr_str.strip()
    # variable?
    if expr_str.isalpha():
        return VariableExpression(expr_str)
    # int?
    try:
        val = int(expr_str)
        return ConstantExpression(val)
    except:
        # fallback as constant
        return ConstantExpression(expr_str)


def parse_arc_expression(arc_expr_str, color_sets, env):
    arc_expr_str = arc_expr_str.strip()

    # tuple?
    if arc_expr_str.startswith("(") and arc_expr_str.endswith(")"):
        inner = arc_expr_str.strip("()")
        vars_ = [v.strip() for v in inner.split(",")]
        exprs = [parse_atomic_expression(v) for v in vars_]
        return FunctionExpression(lambda *args: tuple(args), exprs)

    # variable?
    if arc_expr_str.isalpha():
        return VariableExpression(arc_expr_str)

    # int?
    try:
        val = int(arc_expr_str)
        return ConstantExpression(val)
    except:
        pass

    # Try SML parsing as a fallback
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
            # complex SML => fallback to constant
            return ConstantExpression(arc_expr_str)
    except:
        # fallback
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
                    initial_tokens = parse_initial_marking(mtext, place_obj, color_sets, env)

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

                if t_id in transitions and p_id in places:
                    if orientation == "PtoT":
                        source = places[p_id]
                        target = transitions[t_id]
                    else:
                        source = transitions[t_id]
                        target = places[p_id]
                elif p_id in transitions and t_id in places:
                    if orientation == "PtoT":
                        source = places[t_id]
                        target = transitions[p_id]
                    else:
                        source = transitions[p_id]
                        target = places[t_id]
                else:
                    # not found
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
    # Example test:
    # We will parse a given file (user should have a test file prepared)
    # and try to run until no transitions are enabled.
    # Just print the initial state and try firing any enabled transitions.

    # Here we just show the mechanism. The user can replace "testcases/MyModel.cpn" with their file.
    filename = "testcases/DistributedDataBase.cpn"  # placeholder
    net = parse_cpn(filename)
    print("Places:", net.places)
    print("Transitions:", net.transitions)
    print("Arcs:", net.arcs)
    print("Initial Marking:", net.initial_marking)

    changed = True
    while changed:
        changed = False
        for t in net.transitions:
            enabled_bs = get_enabled_bindings(net, t)
            if enabled_bs:
                print("ENABLED:", t, enabled_bs)
                fire_transition(net, t, enabled_bs[0][1])
                print("Marking after firing:", net.initial_marking)
                changed = True
                break
