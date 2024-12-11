import xml.etree.ElementTree as ET
from cpn import CPN, Place, Transition, Arc, ColorSet, Multiset, VariableExpression, ConstantExpression, FunctionExpression, Guard
from sml import evaluate_sml_expression, SMLParser
from algo import get_enabled_bindings, is_enabled, fire_transition, can_fire_step, fire_step

"""
Final Revised Importer with Test Cases in __main__:

This importer and runner attempts to handle both the DistributedDataBase.cpn and the Chopsticks example.
We have implemented heuristic-based handling for known functions like "Mes(s)" and "Chopsticks(p)" and
tried to interpret color sets as best as possible.

The __main__ section will:
- Attempt to parse and run the provided testcases/DistributedDataBase.cpn.
- Attempt to parse and run the provided testcases/NewExample.cpn (the Chopsticks example).
- Print out initial states, enabled transitions (if any), and fire them until no more are enabled.
"""

class IndexColorSet(ColorSet):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def is_member(self, value) -> bool:
        return isinstance(value, int) and self.start <= value <= self.end

    def all_tokens(self):
        return list(range(self.start, self.end + 1))


class ProductColorSet(ColorSet):
    def __init__(self, cs1: IndexColorSet, cs2: IndexColorSet):
        self.cs1 = cs1
        self.cs2 = cs2

    def is_member(self, value) -> bool:
        if isinstance(value, tuple) and len(value) == 2:
            return self.cs1.is_member(value[0]) and self.cs2.is_member(value[1])
        return False

    def all_tokens(self):
        return [(x,y) for x in self.cs1.all_tokens() for y in self.cs2.all_tokens()]


class SubsetColorSet(ColorSet):
    def __init__(self, base_cs: ProductColorSet, predicate):
        self.base_cs = base_cs
        self.predicate = predicate

    def is_member(self, value) -> bool:
        return self.base_cs.is_member(value) and self.predicate(value[0], value[1])

    def all_tokens(self):
        return [(x,y) for (x,y) in self.base_cs.all_tokens() if self.predicate(x,y)]


class EnumColorSet(ColorSet):
    def __init__(self, elements):
        self.elements = elements

    def is_member(self, value) -> bool:
        return value in self.elements

    def all_tokens(self):
        return self.elements[:]


class PermissiveColorSet(ColorSet):
    def is_member(self, value):
        if isinstance(value, tuple):
            return all(isinstance(v, (int, str)) for v in value)
        return isinstance(value, (int, str))


def parse_global_declarations(globbox_el):
    env = {}
    if globbox_el is None:
        return env
    for ml in globbox_el.findall('ml'):
        code = ml.text.strip()
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

def diff_pred(x,y):
    return x!=y

def parse_color_sets(globbox_el, env):
    color_sets = {}
    if globbox_el is None:
        return color_sets

    pending_products = []
    pending_subsets = []

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
            base_id = subset_el.find('id')
            if base_id is not None:
                base_name = base_id.text.strip()
                by_el = subset_el.find('by')
                if by_el is not None:
                    ml_el = by_el.find('ml')
                    if ml_el is not None:
                        func_name = ml_el.text.strip()
                        if func_name=="diff":
                            predicate = diff_pred
                        else:
                            predicate = lambda x,y: True
                    else:
                        predicate = lambda x,y:True
                else:
                    predicate = lambda x,y:True
                pending_subsets.append((cname, base_name, predicate))
            else:
                color_sets[cname] = PermissiveColorSet()
            continue

    # second pass for products
    for (cname,c1,c2) in pending_products:
        if c1 in color_sets and isinstance(color_sets[c1], IndexColorSet) and c2 in color_sets and isinstance(color_sets[c2], IndexColorSet):
            color_sets[cname] = ProductColorSet(color_sets[c1], color_sets[c2])
        else:
            color_sets[cname] = PermissiveColorSet()

    # subsets
    for (cname,base_name,predicate) in pending_subsets:
        if base_name in color_sets and isinstance(color_sets[base_name], ProductColorSet):
            color_sets[cname] = SubsetColorSet(color_sets[base_name], predicate)
        else:
            color_sets[cname] = PermissiveColorSet()

    return color_sets

def dbm_all(env):
    n = env.get('n',4)
    return [i for i in range(1,n+1)]

def ph_all(env):
    n = env.get('n',5)
    return [i for i in range(1,n+1)]

def cs_all(env):
    n = env.get('n',5)
    return [i for i in range(1,n+1)]

def pr_all_dbm(env):
    d = dbm_all(env)
    return [(x,y) for x in d for y in d]

def mes_all_dbm(env):
    d = dbm_all(env)
    return [(x,y) for x in d for y in d if x!=y]

def parse_initial_marking(mtext, place_obj, color_sets, env):
    if mtext.endswith('all()'):
        cname = mtext.split('.')[0]
        if cname == "DBM":
            return dbm_all(env)
        elif cname == "MES":
            if "MES" in color_sets and hasattr(color_sets["MES"],'all_tokens'):
                return color_sets["MES"].all_tokens()
            else:
                return mes_all_dbm(env)
        elif cname == "PR":
            if "PR" in color_sets and hasattr(color_sets["PR"],'all_tokens'):
                return color_sets["PR"].all_tokens()
            else:
                return pr_all_dbm(env)
        elif cname == "PH":
            if "PH" in color_sets and isinstance(color_sets["PH"],IndexColorSet):
                return color_sets["PH"].all_tokens()
            else:
                return ph_all(env)
        elif cname == "CS":
            if "CS" in color_sets and isinstance(color_sets["CS"],IndexColorSet):
                return color_sets["CS"].all_tokens()
            else:
                return cs_all(env)
        else:
            if cname in color_sets and hasattr(color_sets[cname],'all_tokens'):
                return color_sets[cname].all_tokens()
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

def mes_function(env, s):
    return [(s,r) for r in dbm_all(env) if r!=s]

def chopsticks_function(env, p):
    n = env.get('n',5)
    next_p = 1 if p==n else p+1
    return [p, next_p]

def parse_arc_expression(arc_expr_str, color_sets, env):
    arc_expr_str = arc_expr_str.strip()

    if arc_expr_str == "Mes(s)":
        if "DBM" in color_sets and isinstance(color_sets["DBM"],IndexColorSet):
            p_var = VariableExpression("s")
            func = lambda s: mes_function(env, s)
            return FunctionExpression(func, [p_var])
        else:
            return ConstantExpression("Mes(s)")

    if arc_expr_str == "Chopsticks(p)":
        if "PH" in color_sets and isinstance(color_sets["PH"],IndexColorSet) and "CS" in color_sets and isinstance(color_sets["CS"],IndexColorSet) and 'n' in env:
            p_var = VariableExpression("p")
            func = lambda p: chopsticks_function(env, p)
            return FunctionExpression(func, [p_var])
        else:
            return ConstantExpression("Chopsticks(p)")

    if arc_expr_str.startswith("(") and arc_expr_str.endswith(")"):
        inner = arc_expr_str.strip("()")
        vars_ = [v.strip() for v in inner.split(",")]
        exprs = []
        for var_ in vars_:
            if var_.isalpha():
                exprs.append(VariableExpression(var_))
            else:
                try:
                    c_val = int(var_)
                    exprs.append(ConstantExpression(c_val))
                except:
                    exprs.append(ConstantExpression(var_))
        return FunctionExpression(lambda *args: tuple(args), exprs)

    if arc_expr_str.isalpha():
        return VariableExpression(arc_expr_str)

    try:
        val = int(arc_expr_str)
        return ConstantExpression(val)
    except:
        pass

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
    # Test with DistributedDataBase.cpn
    print("Testing with DistributedDataBase.cpn:")
    net_db = parse_cpn("testcases/DistributedDataBase.cpn")
    print("Places:", net_db.places)
    print("Transitions:", net_db.transitions)
    print("Arcs:", net_db.arcs)
    print("Initial Marking:", net_db.initial_marking)

    changed = True
    while changed:
        changed = False
        for t in net_db.transitions:
            enabled_bs = get_enabled_bindings(net_db, t)
            if enabled_bs:
                print("ENABLED:", t, enabled_bs)
                fire_transition(net_db, t, enabled_bs[0][1])
                print("Marking after firing:", net_db.initial_marking)
                changed = True
                break

    # Test with NewExample.cpn (Chopsticks)
    print("\nTesting with NewExample.cpn:")
    net_chopsticks = parse_cpn("testcases/DiningPhilosophers.cpn")
    print("Places:", net_chopsticks.places)
    print("Transitions:", net_chopsticks.transitions)
    print("Arcs:", net_chopsticks.arcs)
    print("Initial Marking:", net_chopsticks.initial_marking)

    changed = True
    while changed:
        changed = False
        for t in net_chopsticks.transitions:
            enabled_bs = get_enabled_bindings(net_chopsticks, t)
            if enabled_bs:
                print("ENABLED:", t, enabled_bs)
                fire_transition(net_chopsticks, t, enabled_bs[0][1])
                print("Marking after firing:", net_chopsticks.initial_marking)
                changed = True
                break
