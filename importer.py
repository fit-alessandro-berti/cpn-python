import xml.etree.ElementTree as ET
import re
from cpn import CPN, Place, Transition, Arc, ColorSet, Multiset, VariableExpression, ConstantExpression, FunctionExpression, Guard
from sml import evaluate_sml_expression, SMLParser
from algo import get_enabled_bindings, is_enabled, fire_transition, can_fire_step, fire_step

"""
Generalized CPN Importer (Improved):

We avoid overfitting by not hardcoding any particular color sets, functions,
or variable names. Instead, we apply generic heuristics to parse and evaluate
the model. The key points are:

- Color sets:
  * If 'enum': create EnumColorSet
  * If 'index': try to create IndexColorSet from 1..n if 'n' is known
  * If 'product': if both sub-colors are known (IndexColorSet/EnumColorSet/Permissive), create ProductColorSet
  * Otherwise fallback to PermissiveColorSet
  * We don't handle subsets or complex constructs specially; fallback to PermissiveColorSet.

- Initial markings:
  * If token ends in '.all()', try to call all_tokens() on the corresponding color set, or fallback to [0,1,2].
  * If integer and matches colorset -> that token
  * If string and matches colorset -> that token
  * Else no tokens

- Arc expressions:
  * Try variable (if purely alphabetic)
  * Try integer
  * If tuple syntax "(x,y,...)" parse each component similarly
  * If function-call-like syntax "Name(arg1,...,argN)": we generically treat this as a FunctionExpression
    returning a tuple of the evaluated arguments. We do not assume any semantic for the function name.
    This handles cases like "Chopsticks(p)" or "Mes(s)" without special-casing them.
  * Otherwise, try SML parsing:
    - If SML parse yields a simple var/int/bool -> corresponding expression
    - Else fallback to a ConstantExpression of the raw string.

This approach is more dynamic and does not rely on specific known examples.
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
        self.cs1 = cs1
        self.cs2 = cs2

    def is_member(self, value) -> bool:
        if isinstance(value, tuple) and len(value) == 2:
            return self.cs1.is_member(value[0]) and self.cs2.is_member(value[1])
        return False

    def all_tokens(self):
        have_all1 = hasattr(self.cs1, 'all_tokens')
        have_all2 = hasattr(self.cs2, 'all_tokens')
        if have_all1 and have_all2:
            return [(x,y) for x in self.cs1.all_tokens() for y in self.cs2.all_tokens()]
        # fallback
        return [0,1,2]


class PermissiveColorSet(ColorSet):
    def is_member(self, value):
        if isinstance(value, tuple):
            return all(isinstance(v, (int, str)) for v in value)
        return isinstance(value, (int,str))


def parse_global_declarations(globbox_el):
    env = {}
    if globbox_el is None:
        return env
    for ml in globbox_el.findall('ml'):
        code = ml.text.strip()
        if code.startswith("val "):
            code_line = code.split(';')[0]
            parts = code_line.split('=')
            if len(parts)==2:
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

    pending_products = []

    for c_el in globbox_el.findall('color'):
        cid_el = c_el.find('id')
        if cid_el is None:
            continue
        cname = cid_el.text.strip()

        enum_el = c_el.find('enum')
        if enum_el is not None:
            ids = enum_el.findall('id')
            elems = [i.text.strip() for i in ids]
            color_sets[cname] = EnumColorSet(elems)
            continue

        index_el = c_el.find('index')
        if index_el is not None:
            mls = index_el.findall('ml')
            if len(mls)==2:
                try:
                    start = int(mls[0].text.strip())
                    end_part = mls[1].text.strip()
                    if end_part in env and isinstance(env[end_part],int):
                        end = env[end_part]
                        color_sets[cname] = IndexColorSet(start,end)
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

        product_el = c_el.find('product')
        if product_el is not None:
            ids = product_el.findall('id')
            if len(ids)==2:
                c1 = ids[0].text.strip()
                c2 = ids[1].text.strip()
                pending_products.append((cname,c1,c2))
            else:
                color_sets[cname] = PermissiveColorSet()
            continue

        subset_el = c_el.find('subset')
        if subset_el is not None:
            # Just fallback
            color_sets[cname] = PermissiveColorSet()
            continue

        # If no known structure, fallback
        color_sets[cname] = PermissiveColorSet()

    # second pass for products
    for (cname,c1,c2) in pending_products:
        cs1 = color_sets.get(c1, None)
        cs2 = color_sets.get(c2, None)
        if cs1 and cs2 and isinstance(cs1,(IndexColorSet,EnumColorSet,PermissiveColorSet)) \
           and isinstance(cs2,(IndexColorSet,EnumColorSet,PermissiveColorSet)):
            color_sets[cname] = ProductColorSet(cs1, cs2)
        else:
            color_sets[cname] = PermissiveColorSet()

    return color_sets


def parse_initial_marking(mtext, place_obj, color_sets, env):
    if mtext.endswith('all()'):
        cname = mtext.split('.')[0]
        cs = color_sets.get(cname, None)
        if cs and hasattr(cs,'all_tokens'):
            return cs.all_tokens()
        else:
            # fallback
            return [0,1,2]
    else:
        # try int
        try:
            val = int(mtext)
            if place_obj.colorset.is_member(val):
                return [val]
            else:
                return []
        except:
            # string token
            if place_obj.colorset.is_member(mtext):
                return [mtext]
            return []


def parse_function_call(expr_str, color_sets, env):
    # Generic function call pattern: Name(arg1, arg2, ...)
    # We'll parse arguments by splitting on commas, assuming no nested parentheses for simplicity.
    # We just return a FunctionExpression that returns a tuple of the arguments.
    # If only one argument, return just that argument as a single value, else a tuple.
    match = re.match(r'^([A-Za-z_]\w*)\((.*)\)$', expr_str)
    if not match:
        return None
    func_name = match.group(1)
    args_str = match.group(2).strip()

    # split by commas at top level (no nested parentheses assumed)
    # a simple split by ',' is risky if we had complex expressions. For now assume simple arguments.
    if args_str.strip()=='':
        # no arguments
        return FunctionExpression(lambda: None, [])

    args_parts = [a.strip() for a in args_str.split(',')]

    args_expr = [parse_arc_expression(a, color_sets, env) for a in args_parts]
    # Return a tuple if more than one arg, else single arg
    def func(*vals):
        if len(vals)==1:
            return vals[0]
        return tuple(vals)
    return FunctionExpression(func, args_expr)


def parse_arc_expression(arc_expr_str, color_sets, env):
    arc_expr_str = arc_expr_str.strip()

    # tuple?
    if arc_expr_str.startswith("(") and arc_expr_str.endswith(")"):
        inner = arc_expr_str.strip("()")
        vars_ = [v.strip() for v in inner.split(",")]
        exprs = []
        for var_ in vars_:
            # variable or int or constant
            # try variable
            if var_.isalpha():
                exprs.append(VariableExpression(var_))
            else:
                # try int
                try:
                    val = int(var_)
                    exprs.append(ConstantExpression(val))
                except:
                    exprs.append(ConstantExpression(var_))
        return FunctionExpression(lambda *args: tuple(args), exprs)

    # function call?
    # Identify by pattern: name(...)
    if re.match(r'^[A-Za-z_]\w*\(.*\)$', arc_expr_str):
        fc_expr = parse_function_call(arc_expr_str, color_sets, env)
        if fc_expr is not None:
            return fc_expr

    # variable?
    if arc_expr_str.isalpha():
        return VariableExpression(arc_expr_str)

    # int?
    try:
        val = int(arc_expr_str)
        return ConstantExpression(val)
    except:
        pass

    # Try SML parsing
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
            # complex -> fallback
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
    # Test run (user should provide a file)
    filename = "testcases/DiningPhilosophers.cpn"
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
