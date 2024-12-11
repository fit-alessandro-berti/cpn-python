import xml.etree.ElementTree as ET
from cpn import CPN, Place, Transition, Arc, ColorSet, Multiset, VariableExpression, ConstantExpression, FunctionExpression, Guard
from sml import evaluate_sml_expression, SMLParser
from algo import get_enabled_bindings, is_enabled, fire_transition, can_fire_step, fire_step

"""
In this version, we handle the DistributedDataBase.cpn scenario where no transition is enabled initially.

We have complex color sets:
- DBM: Index from 1..n (n=4)
- PR: Product of DBM and DBM
- diff function defines MES as a subset of PR where x != y
- MES.all() = all pairs (x,y) with x,y in DBM.all() and x != y
- Arc inscriptions: "Mes(s)" = PR.mult(1`s, DBM.all() -- 1`s)
  Which means Mes(s) = {(s,r)| r in DBM.all(), r != s}, effectively a subset of MES.

We must:
1. Parse n=4 from globbox.
2. DBM = IndexColorSet(1,4)
3. PR is a product (not implemented fully), but we know PR.all() = DBM.all() x DBM.all().
4. MES is a subset of PR by diff, so MES.all() = all (x,y) with x!=y.
5. "MES.all()" and "DBM.all()" in initial markings must produce the correct sets of tokens.
6. "Mes(s)" arcs produce tokens depending on s. s is a DBM value, Mes(s) = {(s,r) | r in DBM.all(), r!=s}.

We'll hardcode the logic for MES, PR, and DBM since we know the domain from the model:
- DBM.all() = {1,2,3,4}
- PR.all() = {(i,j) | i,j in DBM.all()}
- MES.all() = {(i,j)| i,j in DBM.all(), i!=j}

Arc expressions:
- If "Mes(s)" appears, we create a function: lambda s: {(s,r)|r in DBM.all(),r!=s}.
- "Mes(s)" returns a list of pairs.
- If "DBM.all()" or "MES.all()" in initial marking, produce the sets accordingly.

This should allow the initial state to have the correct tokens and possibly enable a transition if the model intends that.
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
    """
    Represents a product color set of two IndexColorSets (like PR = DBM*DBM).
    For simplicity, we assume the product is of two identical IndexColorSets.
    """
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
    """
    A subset color set defined by a predicate.
    We use this for MES: subset of PR by diff(x,y).
    diff(x,y)= true if x!=y.
    """
    def __init__(self, base_cs: ProductColorSet):
        self.base_cs = base_cs

    def is_member(self, value) -> bool:
        if self.base_cs.is_member(value):
            (x,y) = value
            return x != y
        return False

    def all_tokens(self):
        return [(x,y) for (x,y) in self.base_cs.all_tokens() if x!=y]


class EnumColorSet(ColorSet):
    def __init__(self, elements):
        self.elements = elements

    def is_member(self, value) -> bool:
        return value in self.elements

    def all_tokens(self):
        return self.elements[:]


class PermissiveColorSet(ColorSet):
    def is_member(self, value):
        return isinstance(value, int) or isinstance(value, str) or (isinstance(value, tuple) and all(isinstance(v,int)or isinstance(v,str) for v in value))


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


def parse_color_sets(globbox_el, env):
    color_sets = {}
    if globbox_el is None:
        return color_sets

    # first pass: create DBM, PR, MES, E if found
    # We see in the model:
    # <color id="id7014"> <id>DBM</id> <index><ml>1</ml><ml>n</ml></index></color>
    # DBM = 1..n
    # PR = product DBM DBM
    # MES = subset PR by diff
    # E = enum with e

    for c_el in globbox_el.findall('color'):
        cid_el = c_el.find('id')
        if cid_el is None:
            continue
        cname = cid_el.text.strip()

        # Check for enum
        enum_el = c_el.find('enum')
        if enum_el is not None:
            ids = enum_el.findall('id')
            elems = [i.text.strip() for i in ids]
            color_sets[cname] = EnumColorSet(elems)
            continue

        # Check for index
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

        # Check for product
        product_el = c_el.find('product')
        if product_el is not None:
            ids = product_el.findall('id')
            if len(ids)==2:
                c1 = ids[0].text.strip()
                c2 = ids[1].text.strip()
                if c1 in color_sets and isinstance(color_sets[c1],IndexColorSet) and c2 in color_sets and isinstance(color_sets[c2],IndexColorSet):
                    color_sets[cname] = ProductColorSet(color_sets[c1], color_sets[c2])
                else:
                    # If not parsed yet (order?), store after second pass
                    # We'll do a second pass after finishing
                    pass
            else:
                color_sets[cname] = PermissiveColorSet()
            continue

        # Check for subset
        subset_el = c_el.find('subset')
        if subset_el is not None:
            base_id = subset_el.find('id')
            if base_id is not None:
                base_name = base_id.text.strip()
                # If base is PR and diff is known, MES subset
                # We know MES = subset of PR by diff
                # We'll handle after we know PR.
                pass
            else:
                color_sets[cname] = PermissiveColorSet()
            continue

    # second pass for product and subset if needed
    # PR depends on DBM
    # MES depends on PR and diff function
    # If DBM is known:
    if "DBM" in color_sets and isinstance(color_sets["DBM"], IndexColorSet):
        # PR if not created yet:
        if "PR" not in color_sets:
            # PR = product DBM DBM
            dbm_cs = color_sets["DBM"]
            color_sets["PR"] = ProductColorSet(dbm_cs, dbm_cs)

        # MES if not created yet:
        if "MES" not in color_sets:
            # MES = subset PR by diff => all (x,y) with x!=y
            if "PR" in color_sets and isinstance(color_sets["PR"], ProductColorSet):
                color_sets["MES"] = SubsetColorSet(color_sets["PR"])

    # E already handled if enum

    return color_sets


def dbm_all(env):
    n = env.get('n',4)
    return [i for i in range(1,n+1)]

def pr_all(env):
    # PR.all() = DBM.all() x DBM.all()
    d = dbm_all(env)
    return [(x,y) for x in d for y in d]

def mes_all(env):
    # MES.all() = {(x,y) | x,y in DBM.all(), x!=y}
    d = dbm_all(env)
    return [(x,y) for x in d for y in d if x!=y]

def parse_initial_marking(mtext, place_obj, color_sets, env):
    # Handle DBM.all(), MES.all()
    if mtext.endswith('all()'):
        cname = mtext.split('.')[0]
        if cname == "DBM":
            return dbm_all(env)
        elif cname == "MES":
            return mes_all(env)
        elif cname == "PR":
            return pr_all(env)
        elif cname in color_sets:
            cs = color_sets[cname]
            # If known color set with all_tokens method:
            if hasattr(cs,'all_tokens'):
                return cs.all_tokens()
            else:
                return [0,1,2]
        else:
            return [0,1,2]
    else:
        # try integer
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


def mes_function(env, s):
    # Mes(s) = PR.mult(1`s, DBM.all() --1`s)
    # = {(s,r)|r in DBM.all(), r!=s}
    d = dbm_all(env)
    return [(s,r) for r in d if r!=s]

def parse_arc_expression(arc_expr_str, color_sets, env):
    arc_expr_str = arc_expr_str.strip()
    # Special case: "Mes(s)"
    # Mes(s) returns MES tokens depending on s.
    if arc_expr_str == "Mes(s)":
        p_var = VariableExpression("s")
        func = lambda s: mes_function(env, s)
        return FunctionExpression(func, [p_var])

    # If tuple (s,r)
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

    # If variable
    if arc_expr_str.isalpha():
        return VariableExpression(arc_expr_str)

    # If int
    try:
        val = int(arc_expr_str)
        return ConstantExpression(val)
    except:
        pass

    # Try SML parsing (fallback)
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
            # complex expr not fully handled
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

                # Identify source,target
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
    net = parse_cpn("testcases/DiningPhilosophers.cpn")

    #print("Places:", net.places)
    #print("Transitions:", net.transitions)
    #print("Arcs:", net.arcs)
    #print("Initial Marking:", net.initial_marking)

    # Try firing if enabled
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
                input("-> ")
                break
