import xml.etree.ElementTree as ET
from cpn import CPN, Place, Transition, Arc, ColorSet, Multiset, VariableExpression, ConstantExpression, FunctionExpression, Guard
from sml import evaluate_sml_expression, SMLParser
from algo import get_enabled_bindings, is_enabled, fire_transition, can_fire_step, fire_step

"""
We encountered a KeyError for a variable 's' not found in the binding.
This means that the arc expressions contain variables that the transition does not declare.

We must ensure that transitions are aware of all variables used in their arc expressions.

We will:
- After parsing arcs, collect all variables from arc expressions connected to a transition
  and add them to the transition's variable list if they are not already present.

We will also make sure the color sets allow for these variables. We still use the heuristic approach.

If certain variables appear that we cannot bind (like 's','r'), we will try to provide some default domain
for them. For the demonstration, if we can't guess a domain from tokens, let's just assume integers 0,1,2.
"""

class EColorSet(ColorSet):
    def is_member(self, value):
        return value == 'e'

class PermissiveColorSet(ColorSet):
    def is_member(self, value):
        # Allow int or string, as a fallback
        return isinstance(value, int) or isinstance(value, str)


def parse_cpn(filename: str) -> CPN:
    tree = ET.parse(filename)
    root = tree.getroot()

    net = CPN()

    places = {}
    transitions = {}
    arcs = []

    for cpnet in root.findall('cpnet'):
        for page in cpnet.findall('page'):
            # Parse places
            for p_el in page.findall('place'):
                pid = p_el.get('id')
                pname = None
                for t in p_el.findall('text'):
                    pname = t.text.strip()
                if pname is None:
                    pname = pid

                # Determine colorset
                cset = PermissiveColorSet()
                t_el = p_el.find('type')
                if t_el is not None:
                    t_text_el = t_el.find('text')
                    if t_text_el is not None and t_text_el.text:
                        typename = t_text_el.text.strip()
                        if typename == "E":
                            cset = EColorSet()
                        # otherwise permissive

                place_obj = Place(pname, cset)

                # initial marking
                init_mark_el = p_el.find('initmark')
                initial_tokens = []
                if init_mark_el is not None:
                    mark_text = init_mark_el.find('text')
                    if mark_text is not None:
                        mtext = mark_text.text.strip()
                        if mtext == "e":
                            if place_obj.colorset.is_member('e'):
                                initial_tokens.append('e')
                        elif "all()" in mtext:
                            if isinstance(place_obj.colorset, EColorSet):
                                initial_tokens.append('e')
                            else:
                                initial_tokens.extend([0,1,2])
                        else:
                            # try int
                            try:
                                val = int(mtext)
                                if place_obj.colorset.is_member(val):
                                    initial_tokens.append(val)
                            except:
                                if place_obj.colorset.is_member(mtext):
                                    initial_tokens.append(mtext)

                net.add_place(place_obj, initial_tokens)
                places[pid] = place_obj

            # Parse transitions
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

            # Parse arcs
            for a_el in page.findall('arc'):
                orientation = a_el.get('orientation')
                transend = a_el.find('transend')
                placeend = a_el.find('placeend')

                if transend is not None and placeend is not None:
                    t_id = transend.get('idref')
                    p_id = placeend.get('idref')

                    if orientation == "PtoT":
                        source = places[p_id]
                        target = transitions[t_id]
                    else:
                        source = transitions[t_id]
                        target = places[p_id]

                    annot = a_el.find('annot')
                    inscription_expr = ConstantExpression(1)
                    if annot is not None:
                        annot_text_el = annot.find('text')
                        if annot_text_el is not None and annot_text_el.text:
                            arc_expr_str = annot_text_el.text.strip()
                            # Very naive parsing:
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
                                inscription_expr = FunctionExpression(lambda *args: tuple(args), exprs)
                            elif arc_expr_str.isalpha():
                                # single variable?
                                inscription_expr = VariableExpression(arc_expr_str)
                            elif "Mes(s)" in arc_expr_str:
                                inscription_expr = VariableExpression("s")
                            else:
                                # try int
                                try:
                                    c_val = int(arc_expr_str)
                                    inscription_expr = ConstantExpression(c_val)
                                except:
                                    inscription_expr = ConstantExpression(arc_expr_str)

                    arc_obj = Arc(source, target, inscription_expr)
                    net.add_arc(arc_obj)
                    arcs.append(arc_obj)

    # Now, we must ensure transitions know all variables used in arc expressions.
    # Let's go through all arcs and add variables to transitions if needed.
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
    # Try again with the updated code
    net = parse_cpn("testcases/DistributedDataBase.cpn")

    #print("Places:", net.places)
    #print("Transitions:", net.transitions)
    #print("Arcs:", net.arcs)
    #print("Initial Marking:", net.initial_marking)

    # Try to find an enabled transition
    is_break = False
    while True:
        is_break = True
        transitions = list(net.transitions)
        import random
        random.shuffle(transitions)
        for t in transitions:
            enabled_bs = get_enabled_bindings(net, t)
            #print(t)
            if enabled_bs:
                # Fire one
                print("ENABLED:", t)
                fire_transition(net, t, enabled_bs[0][1])
                print("Marking after firing:", net.initial_marking)
                is_break = False
                continue
        print(is_break)
        if is_break:
            break
