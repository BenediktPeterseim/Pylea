"""What a drawn dot diagram must show, whatever the model.

The properties here are the ones a reader relies on: a dot for every variable, a box for
every assignment, a wire for every argument and every result, and nothing overlapping
anything else.  The degenerate models are covered too, since a body with no ops or an output
naming an input has no box to hang the picture on.

The properties of the layout itself hold for any sketch, so they run over the dependency
graph of each model as well; what that picture must show is in `test.draw.test_dependency`.
"""
import random
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree

import pytest

from pylea import DPP, Bool, flip, observe
from pylea.draw import coordinates, diagram_of
from pylea.draw.glyph import GlyphKind, Style, glyph_of
from pylea.draw.layout import _Layered, _inversions, _longest_path_ranks, _order, _ranks, place
from pylea.draw.sketch import _accepted_groups, display_name, segments, sketch_of
from pylea.examples import bioassay, electoral_college, percolation, simple
from pylea.examples.disease import has_disease, test as disease_test
from pylea.ir.body import BodyBuilder
from pylea.ir.prim import Prim
from pylea.ir.program import Program

DECISIVE = electoral_college.decisive("A", {"A": 1, "B": 2, "C": 3, "D": 4}, quota=6)

MODELS = [simple.empty_dpp, simple.identity_dpp, simple.unused_input_dpp,
          simple.unused_output_dpp, simple.two_outputs_dpp, simple.dpp_with_observe,
          simple.tree_structured_fault_tree_larger, simple.color_random_then_match,
          disease_test, has_disease, DECISIVE]


def sketches():
    return [(model.name(), sketch_of(model.program())) for model in MODELS]


def pictures():
    """Every sketch the layout has to place: the dot diagrams and the dependency graphs."""
    return sketches() + [(f"{model.name()} dependencies", model.dependency_graph().sketch)
                         for model in MODELS]


@pytest.mark.parametrize("name,sketch", sketches())
def test_there_is_one_dot_per_wire_and_one_box_per_assignment(name, sketch):
    body = dict((m.name(), m) for m in MODELS)[name].program().main
    dots = [n for n in sketch.nodes if n.wire is not None]
    boxes = [n for n in sketch.nodes if n.op is not None]
    assert sorted(n.wire for n in dots) == list(range(len(body.bits)))
    assert sorted(n.op for n in boxes) == list(range(len(body.ops)))


@pytest.mark.parametrize("name,sketch", sketches())
def test_there_is_one_wire_per_argument_result_and_interface_position(name, sketch):
    body = dict((m.name(), m) for m in MODELS)[name].program().main
    expected = (sum(len(op.args) + len(op.outs) for op in body.ops)
                + len(body.inputs) + len(body.outputs))
    assert len(sketch.edges) == expected


@pytest.mark.parametrize("name,sketch", pictures())
def test_every_wire_runs_the_way_its_picture_reads(name, sketch):
    placement = place(sketch)
    for route in placement.routes:
        ys = [y for _, y in route.points]
        assert ys == sorted(ys, reverse=sketch.downward), f"{name}: a wire doubles back"


@pytest.mark.parametrize("name,sketch", pictures())
def test_nothing_on_one_rank_overlaps_anything_else(name, sketch):
    placement, style = place(sketch), Style()
    rows: dict[float, list[int]] = {}
    for i, node in enumerate(sketch.nodes):
        if node.kind is not GlyphKind.BOUNDARY:
            rows.setdefault(round(placement.y[i], 6), []).append(i)
    for row in rows.values():
        row.sort(key=lambda i: placement.x[i])
        for a, b in zip(row, row[1:]):
            span = (placement.glyphs[a].width + placement.glyphs[b].width) / 2
            gap = placement.x[b] - placement.x[a]
            assert gap >= span + style.node_sep - 1e-6, f"{name}: {a} and {b} collide"


@pytest.mark.parametrize("name,sketch", pictures())
def test_every_node_and_wire_lies_inside_the_measured_extent(name, sketch):
    placement = place(sketch)
    for i in range(len(sketch.nodes)):
        assert -1e-6 <= placement.x[i] <= placement.width + 1e-6, name
        assert -1e-6 <= placement.y[i] <= placement.height + 1e-6, name
    for route in placement.routes:
        for x, y in route.points:
            assert -1e-6 <= x <= placement.width + 1e-6, name
            assert -1e-6 <= y <= placement.height + 1e-6, name


@pytest.mark.parametrize("name,sketch", pictures())
def test_the_layout_of_one_sketch_is_always_the_same(name, sketch):
    first, second = place(sketch), place(sketch)
    assert first.x == second.x and first.y == second.y and first.routes == second.routes


def test_an_output_naming_an_input_is_one_dot_with_a_wire_at_both_ends():
    sketch = sketch_of(simple.identity_dpp.program())
    dots = [i for i, n in enumerate(sketch.nodes) if n.wire is not None]
    assert len(dots) == 1
    assert len([e for e in sketch.edges if dots[0] in (e.src, e.dst)]) == 2


def test_a_wire_read_twice_is_one_dot_with_two_wires_leaving_it():
    @DPP
    def shared() -> Bool:
        x = flip(0.3)
        return x & x

    sketch = sketch_of(shared.program())
    leaving = [e for e in sketch.edges if sketch.nodes[e.src].wire is not None]
    assert sorted(e.dst_port for e in leaving if e.src == leaving[0].src) == [0, 1]


def test_an_argument_reaches_the_port_of_the_position_it_was_passed_at():
    @DPP
    def ordered(a: Bool, b: Bool) -> Bool:
        return a & ~b

    sketch = sketch_of(ordered.program())
    ands = [i for i, n in enumerate(sketch.nodes) if n.kind is GlyphKind.AND]
    ports = sorted(e.dst_port for e in sketch.edges if e.dst == ands[0])
    assert ports == [0, 1]


def test_conditioning_is_drawn_as_an_eye_with_no_result():
    @DPP
    def conditioned() -> Bool:
        x = flip(0.5)
        observe(x)
        return x

    eyes = [n for n in sketch_of(conditioned.program()).nodes
            if n.kind is GlyphKind.OBSERVE]
    assert len(eyes) == 1 and eyes[0].n_out == 0


def test_a_coin_flip_carries_its_probability():
    @DPP
    def rare() -> Bool:
        return flip(1e-4)

    labels = [n.label for n in sketch_of(rare.program()).nodes
              if n.kind is GlyphKind.FLIP]
    assert labels == ["0.0001"]
    assert "0.0001" in diagram_of(rare.program()).svg()


def test_a_one_bit_table_with_no_name_is_drawn_as_the_coin_it_is():
    """Simplification folds the coins of a block into one table over no input."""
    @DPP
    def either() -> Bool:
        return flip(0.5) | flip(0.5)

    sources = [(n.kind, n.label) for n in sketch_of(either.simplified().program()).nodes
               if n.op is not None]
    assert sources == [(GlyphKind.FLIP, "0.75")]


def test_a_table_over_nothing_is_drawn_as_the_factor_it_multiplies_by():
    """Simplification folds the observation of a coin into a weight on every assignment."""
    @DPP
    def weighted() -> Bool:
        observe(flip(0.3))
        return flip(0.5)

    labels = sorted(n.label for n in sketch_of(weighted.simplified().program()).nodes
                    if n.op is not None)
    assert labels == ["0.5", "×0.3"]


def test_a_named_table_keeps_its_name_whatever_its_shape():
    from pylea.examples.sprinkler import wetgrass_given_sprinkler_rain

    boxes = [n.label for n in wetgrass_given_sprinkler_rain.diagram().sketch.nodes
             if n.op is not None]
    assert boxes == ["CPT_GrassWetness"]


def test_the_scale_sets_the_displayed_size_and_leaves_the_drawing_alone():
    plain, doubled = has_disease.diagram(), has_disease.diagram(style=Style(scale=2.0))
    assert doubled.size == plain.size
    one, two = (ElementTree.fromstring(d.svg()).attrib for d in (plain, doubled))
    assert one["viewBox"] == two["viewBox"]
    assert float(two["width"]) == pytest.approx(2 * float(one["width"]))
    assert float(two["height"]) == pytest.approx(2 * float(one["height"]))


@pytest.mark.parametrize("name,sketch", sketches())
def test_the_svg_is_well_formed_and_refers_to_nothing_outside_itself(name, sketch):
    svg = diagram_of(dict((m.name(), m) for m in MODELS)[name].program(),
                     title=name).svg()
    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "http://" not in svg.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in svg


def test_a_title_is_written_under_the_picture():
    assert "burglary" in diagram_of(has_disease.program(), title="burglary").svg()


def test_a_title_wider_than_the_picture_stays_inside_the_view_box():
    from pylea.examples.sprinkler import wetgrass_given_sprinkler_rain

    diagram = wetgrass_given_sprinkler_rain.diagram()
    view_box = diagram.svg().split('viewBox="')[1].split('"')[0].split()
    assert float(view_box[0]) < 0 and float(view_box[2]) > diagram.placement.width
    tikz = diagram.tikz()
    box = tikz.split("useasboundingbox (")[1]
    assert float(box.split(",")[0]) < 0


def test_saving_refuses_a_format_it_cannot_write(tmp_path):
    with pytest.raises(ValueError, match="written as .svg"):
        diagram_of(has_disease.program()).save(tmp_path / "diagram.png")


def test_saving_writes_the_same_markup_the_diagram_reports(tmp_path):
    diagram = diagram_of(has_disease.program())
    written = diagram.save(tmp_path / "diagram.svg")
    assert written.read_text(encoding="utf-8") == diagram.svg()


def test_a_notebook_is_offered_the_picture_and_a_summary():
    diagram = diagram_of(has_disease.program())
    bundle = diagram._repr_mimebundle_()
    assert bundle["image/svg+xml"].startswith("<svg")
    assert bundle["text/plain"] == repr(diagram)


def test_an_unnamed_callee_is_labelled_by_its_index():
    labels = [n.label for n in sketch_of(has_disease.program()).nodes if n.op is not None]
    assert labels.count("body#0") == 2


def test_a_named_callee_is_labelled_by_its_name():
    labels = [n.label for n in
              sketch_of(has_disease.program(), body_names={0: "test"}).nodes
              if n.op is not None]
    assert labels.count("test") == 2


def test_the_underscores_marking_an_internal_callable_are_not_shown():
    assert display_name("_CPT_Rain_") == "CPT_Rain"
    assert display_name("has_disease") == "has_disease"


@pytest.mark.parametrize("values,expected", [
    ([], 0), ([1.0], 0), ([0.0, 1.0, 2.0], 0), ([2.0, 1.0, 0.0], 3), ([1.0, 0.0, 2.0], 1),
])
def test_crossings_are_counted_as_out_of_order_pairs(values, expected):
    assert _inversions(values) == expected


def test_a_source_is_placed_below_the_operation_that_first_reads_it():
    """A coin pinned to the bottom rank would stretch every wire leaving it."""
    late = _late()
    sketch = sketch_of(late.program())
    placement = place(sketch)
    flip_at = next(i for i, n in enumerate(sketch.nodes) if n.kind is GlyphKind.FLIP)
    inputs = [i for i, n in enumerate(sketch.nodes) if n.wire in late.program().main.inputs]
    assert placement.y[flip_at] > max(placement.y[i] for i in inputs)


def _span(sketch, rank) -> int:
    """The total number of ranks the wires of `sketch` rise by under `rank`."""
    return sum(rank[e.dst] - rank[e.src] for e in sketch.edges)


def _least_span_and_lowest_ranking(sketch):
    """The least total span of any ranking of `sketch` and the lowest ranking of that span,
    by two linear programs: the least span, then the least sum of ranks with the span
    bounded by it.  Each output marker has a rank of its own, equal to the top rank, and
    the top rank is above every node whose wires all end at output markers."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, vstack

    n = len(sketch.nodes)
    succs = [[] for _ in range(n)]
    for e in sketch.edges:
        succs[e.src].append(e.dst)
    outs = [i for i, node in enumerate(sketch.nodes)
            if node.kind is GlyphKind.BOUNDARY and not succs[i]]
    ins = [i for i, node in enumerate(sketch.nodes)
           if node.kind is GlyphKind.BOUNDARY and succs[i]]
    top = n
    rises = [(e.src, e.dst) for e in sketch.edges]
    if outs:
        rises += [(v, top) for v in range(n)
                  if v not in outs and all(s in outs for s in succs[v])]
    if not rises:
        return 0, [0] * n

    def matrix(pairs, signs):
        rows = [k for k in range(len(pairs)) for _ in range(2)]
        cols = [v for pair in pairs for v in pair]
        return coo_matrix((signs * len(pairs), (rows, cols)), shape=(len(pairs), n + 1))

    a_ub, b_ub = matrix(rises, [1.0, -1.0]), -np.ones(len(rises))
    a_eq = matrix([(i, top) for i in outs], [1.0, -1.0]) if outs else None
    b_eq = np.zeros(len(outs)) if outs else None
    bounds = [(0, 0) if i in ins else (0, None) for i in range(n + 1)]
    span = np.zeros(n + 1)
    for e in sketch.edges:
        span[e.dst] += 1
        span[e.src] -= 1

    first = linprog(span, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds)
    if first.status != 0:           # HiGHS's presolve fails some of these in scipy 1.13
        first = linprog(span, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds,
                        options={"presolve": False})
    least = round(first.fun)
    second = linprog(np.ones(n + 1), A_ub=vstack([a_ub, coo_matrix(span)]),
                     b_ub=np.append(b_ub, least), A_eq=a_eq, b_eq=b_eq, bounds=bounds)
    return least, np.round(second.x[:n]).astype(int).tolist()


def test_the_ranking_uses_as_few_dummy_nodes_as_any_ranking_can():
    for name, sketch in _layout_corpus():
        least, _ = _least_span_and_lowest_ranking(sketch)
        assert _span(sketch, _ranks(sketch)) == least, name


def test_the_ranking_is_the_lowest_of_those_with_the_fewest_dummy_nodes():
    for name, sketch in _layout_corpus():
        _, lowest = _least_span_and_lowest_ranking(sketch)
        assert _ranks(sketch) == lowest, name


def test_the_ranking_does_not_depend_on_the_order_the_wires_are_listed_in():
    from dataclasses import replace

    for name, sketch in _layout_corpus():
        edges = list(sketch.edges)
        random.Random(name).shuffle(edges)
        assert _ranks(replace(sketch, edges=tuple(edges))) == _ranks(sketch), name


def _early_output():
    """A definition whose output is computed before the observation that ends it."""
    @DPP
    def early() -> Bool:
        x = flip(0.5)
        observe(~x | flip(0.3))
        return x

    return "early", sketch_of(early.program())


def test_an_output_computed_early_ends_above_every_other_node():
    for name, sketch in [_early_output(), *_layout_corpus()]:
        rank = _ranks(sketch)
        leaves = {e.src for e in sketch.edges}
        outs = {i for i, node in enumerate(sketch.nodes)
                if node.kind is GlyphKind.BOUNDARY and i not in leaves}
        assert len({rank[i] for i in outs}) == 1, name
        assert all(rank[i] < rank[min(outs)] for i in range(len(rank)) if i not in outs), name


def _wires_through_boxes(sketch, placement) -> list[tuple[int, int]]:
    """The pairs of a wire and a box it does not end at, where a straight run of the wire
    crosses the middle line of the box inside it."""
    from bisect import bisect_left, bisect_right

    boxes = sorted((placement.y[i], i) for i, node in enumerate(sketch.nodes)
                   if node.kind not in (GlyphKind.DOT, GlyphKind.BOUNDARY))
    heights = [y for y, _ in boxes]
    found = []
    for k, (e, wire) in enumerate(zip(sketch.edges, placement.routes)):
        for (x0, y0), (x1, y1) in zip(wire.points, wire.points[1:]):
            low, high = min(y0, y1), max(y0, y1)
            for y, i in boxes[bisect_right(heights, low):bisect_left(heights, high)]:
                if i in (e.src, e.dst):
                    continue
                x = x0 + (x1 - x0) * (y - y0) / (y1 - y0)
                if abs(x - placement.x[i]) < placement.glyphs[i].width / 2 - 1e-6:
                    found.append((k, i))
    return found


def test_no_wire_passes_through_a_box():
    for name, sketch in [_early_output(), *_layout_corpus()]:
        assert not _wires_through_boxes(sketch, place(sketch)), name


def test_a_running_total_has_dummies_in_proportion_to_its_steps():
    """Each step's comparison is ranked next to the step, rather than at the bottom with a
    wire rising from it through the whole picture."""
    from pylea import Enum

    class Colour(Enum):
        elements = ["R", "G", "B", "Y"]

    @DPP
    def running_total() -> Bool:
        total = flip(0.5)
        for _ in range(50):
            total = total ^ (Colour.random({"R": 0.1, "G": 0.2, "B": 0.3, "Y": 0.4}) == "R")
        return total

    sketch = running_total.diagram().sketch
    assert _span(sketch, _ranks(sketch)) <= 2 * len(sketch.edges)


def _late():
    """A definition whose coin flip is read only by its last gate."""
    @DPP
    def late(a: Bool, b: Bool) -> Bool:
        deep = a & b
        return deep & flip(0.5)

    return late


@pytest.mark.parametrize("answer", ["no solution", "a solution breaking a constraint"])
def test_a_ranking_the_solver_does_not_find_is_the_longest_path_ranking(monkeypatch, answer):
    import numpy as np
    import scipy.optimize

    solve, calls = scipy.optimize.linprog, []

    def failing(*args, **kwargs):
        result = solve(*args, **kwargs)
        calls.append(result)
        if answer == "no solution":
            result.status = 2
        elif len(calls) == 2:
            result.x = np.zeros_like(result.x)
        return result

    sketch = sketch_of(_late().program())
    assert _ranks(sketch) != _longest_path_ranks(sketch)
    monkeypatch.setattr(scipy.optimize, "linprog", failing)
    assert _ranks(sketch) == _longest_path_ranks(sketch)
    # A failed first solve is tried once more without presolve before the fallback.
    assert len(calls) == 2
    assert len(place(sketch).routes) == len(sketch.edges)


def _long_wires(model):
    sketch = sketch_of(model.program())
    return [w for w in place(sketch).routes if len(w.points) > 2]


def _off_the_line(wire):
    """The furthest an inner point of `wire` sits from the line joining its two ends."""
    (x0, y0), (xn, yn) = wire.points[0], wire.points[-1]
    return max(abs(x - (x0 + (xn - x0) * (y - y0) / (yn - y0)))
               for x, y in wire.points[1:-1])


def test_a_wire_crossing_several_ranks_runs_straight_between_its_two_bends():
    """The nodes carrying one wire are placed as a block, so the run between them is one
    line: a wire bends only where it meets a glyph, however many ranks it crosses."""
    @DPP
    def hub_and_spokes(xs, types=(Bool,) * 16) -> Bool:
        hub = xs[0]
        gates = [hub & x for x in xs[1:]]
        combined = gates[0]
        for gate in gates[1:]:
            combined = combined | gate
        return combined

    for model in (disease_test, has_disease, hub_and_spokes):
        wires = _long_wires(model)
        assert wires, f"{model.name()} has no wire crossing more than one rank"
        for wire in wires:
            middle = [x for x, _ in wire.points[1:-1]]
            assert max(middle) - min(middle) < 1e-6, f"{model.name()}: the run is not straight"


def test_a_wire_pushed_aside_by_a_box_keeps_the_bend_instead_of_returning():
    """Aiming back at the original line from the next rank would show as a spike."""
    longest = max(_long_wires(has_disease), key=lambda w: len(w.points))
    xs = [x for x, _ in longest.points]
    assert _off_the_line(longest) > 1.0, "this wire is expected to be pushed aside"
    assert xs == sorted(xs), f"the wire doubles back: {[round(x, 1) for x in xs]}"


# --- labels -------------------------------------------------------------------------

def test_a_call_is_labelled_with_the_name_of_the_model_it_calls():
    labels = [n.label for n in has_disease.diagram().sketch.nodes if n.op is not None]
    assert labels.count("test") == 2


def test_an_opaque_matrix_is_labelled_with_the_callable_that_produced_it():
    """A matrix carries no name of its own, so without this a network draws as blank boxes."""
    from pylea.examples.sprinkler import rain_given_cloudy, sprinkler_network

    assert "RAND_Cloudiness" in sprinkler_network.diagram().svg()
    assert "CPT_Rain" in rain_given_cloudy.diagram().svg()


def test_a_network_labels_each_table_s_box_with_the_model_it_calls():
    from pylea.examples.sprinkler import sprinkler_network

    svg = sprinkler_network.diagram().svg()
    for expected in ("sprinkler_given_cloudy", "rain_given_cloudy",
                     "wetgrass_given_sprinkler_rain"):
        assert expected in svg


def test_an_input_wire_is_labelled_with_the_parameter_it_carries():
    assert "has_it" in disease_test.diagram().svg()


def test_a_parameter_of_several_bits_is_one_named_position():
    from pylea import Enum

    class Door(Enum):
        elements = ["a", "b", "c"]

    @DPP
    def picked(door: Door) -> Bool:
        return door == "a"

    assert picked._input_labels() == {0: "door"}
    assert ">door<" in picked.diagram().svg()


def test_a_model_declaring_its_types_by_argument_labels_no_input():
    """There is one parameter for any number of inputs, so no name belongs to a wire."""
    @DPP
    def all_of(xs, types=(Bool,) * 3) -> Bool:
        return xs[0] & xs[1] & xs[2]

    assert all_of._input_labels() == {}


def test_a_voting_gate_is_labelled_by_what_it_computes():
    """The inner function is called `_gate`, which says nothing to a reader."""
    from pylea import at_least

    @DPP
    def two_of_three() -> Bool:
        return at_least(2, flip(0.5), flip(0.5), flip(0.5))

    assert "at_least(2)" in two_of_three.diagram().svg()


def test_a_retyped_comparison_shows_the_comparison_not_the_circuit_beneath_it():
    from pylea import Enum

    class Colour(Enum):
        elements = ["R", "G", "B"]

    @DPP
    def is_red() -> Bool:
        return Colour.uniform() == "R"

    svg = is_red.diagram().svg()
    assert "EQ_Colour" in svg and "_eq" not in svg


def test_a_labelled_interface_marker_leaves_its_wire_vertical():
    """A marker is drawn past the end of the picture, so it cannot meet what shares its rank.

    Charging it the width of its own label instead pushes it aside and bends the wire.
    """
    diagram = disease_test.diagram()
    markers = {i for i, n in enumerate(diagram.sketch.nodes)
               if n.kind is GlyphKind.BOUNDARY}
    stubs = [w for e, w in zip(diagram.sketch.edges, diagram.placement.routes)
             if e.src in markers or e.dst in markers]
    assert stubs
    for stub in stubs:
        assert abs(stub.points[0][0] - stub.points[-1][0]) < 1e-6, "the stub is not vertical"


# --- the TikZ backend ---------------------------------------------------------------

def test_a_tikz_figure_opens_and_closes_one_picture():
    tikz = has_disease.diagram().tikz()
    assert tikz.startswith(r"\begin{tikzpicture}")
    assert tikz.rstrip().endswith(r"\end{tikzpicture}")
    assert tikz.count(r"\begin{tikzpicture}") == 1


def test_a_standalone_figure_carries_its_own_document():
    tikz = has_disease.diagram().tikz(standalone=True)
    assert r"\documentclass[tikz,border=2pt]{standalone}" in tikz
    assert r"\begin{document}" in tikz and r"\end{document}" in tikz


def test_the_characters_latex_reads_as_markup_are_written_out():
    """A label is an identifier, so underscores and hashes are the common case."""
    from pylea.draw.tikz import _escape

    assert _escape("CPT_Rain") == r"CPT\_Rain"
    assert _escape("body#0") == r"body\#0"
    assert _escape("100%") == r"100\%"
    assert _escape("×0.3") == r"$\times$0.3"


def test_a_figure_leaves_no_underscore_or_hash_unescaped():
    for tikz in (has_disease.diagram().tikz(), diagram_of(has_disease.program()).tikz()):
        for line in tikz.splitlines():
            if r"\node" not in line:
                continue
            text = line[line.index("{", line.index("at ")):]
            for character in "_#":
                assert character not in text.replace("\\" + character, ""), line


def test_both_backends_draw_the_same_number_of_wires():
    diagram = has_disease.diagram()
    assert diagram.tikz().count(r"\draw[pylea wire]") == len(diagram.placement.routes)
    assert diagram.svg().count('class="wire"') == len(diagram.placement.routes)


def test_saving_writes_a_standalone_document_for_a_tex_suffix(tmp_path):
    written = has_disease.diagram().save(tmp_path / "figure.tex")
    assert r"\documentclass" in written.read_text(encoding="utf-8")


def test_saving_refuses_a_suffix_neither_backend_writes(tmp_path):
    with pytest.raises(ValueError, match=r"\.svg or as \.tex"):
        has_disease.diagram().save(tmp_path / "figure.pdf")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX toolchain")
@pytest.mark.parametrize("picture", ["diagram", "dependency_graph"])
def test_a_figure_compiles(tmp_path, picture):
    """The one check that the output is LaTeX rather than merely looking like it."""
    from pylea.examples.sprinkler import sprinkler_network

    model = {"diagram": has_disease, "dependency_graph": sprinkler_network}[picture]
    source = getattr(model, picture)().save(tmp_path / "figure.tex")
    finished = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source.name],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert finished.returncode == 0, finished.stdout[-2000:]
    assert (tmp_path / "figure.pdf").exists()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lualatex") is None, reason="no LuaLaTeX")
def test_a_figure_with_a_label_outside_latin_1_compiles_under_lualatex(tmp_path):
    import os

    source = diagram_of(has_disease.program(), title="名前").save(tmp_path / "figure.tex")
    # A font cache of its own, so that the test needs no writable cache of the user's.
    env = {**os.environ, "TEXMFVAR": str(tmp_path / "texmf-var")}
    finished = subprocess.run(
        ["lualatex", "-interaction=nonstopmode", "-halt-on-error", source.name],
        cwd=tmp_path, capture_output=True, text=True, timeout=300, env=env)
    assert finished.returncode == 0, finished.stdout[-2000:]
    assert (tmp_path / "figure.pdf").exists()


def test_a_wide_character_counts_as_two_columns():
    style = Style()
    assert style.text_width("名前") == style.text_width("ab") * 2
    assert style.text_width("é") == style.text_width("e")


def test_an_output_wire_is_labelled_with_what_the_model_returned():
    from pylea.examples.sprinkler import sprinkler_network

    svg = sprinkler_network.diagram().svg()
    for expected in (">cloudiness<", ">sprinkler_state<", ">rain<", ">grass_wetness<"):
        assert expected in svg


def test_the_label_of_an_output_sits_above_it_and_an_input_below():
    diagram = has_disease.diagram()
    labelled = [(i, g) for i, g in enumerate(diagram.placement.glyphs)
                if diagram.sketch.nodes[i].kind is GlyphKind.BOUNDARY and g.labels]
    assert labelled
    for i, glyph in labelled:
        leaves = glyph.out_ports != ()
        assert (glyph.labels[0].y < 0) is leaves, "a marker labels away from the picture"


# --- variables of several bits ------------------------------------------------------

def test_a_variable_whose_bits_travel_together_is_one_dot():
    from pylea.examples.simple import coin_and_color

    sketch = coin_and_color.diagram().sketch
    assert len([n for n in sketch.nodes if n.wire is not None]) == 2


def test_a_call_taking_variables_of_several_bits_has_one_port_per_variable():
    from pylea.examples.sprinkler import sprinkler_network

    sketch = sprinkler_network.diagram().sketch
    boxes = {n.label: n for n in sketch.nodes if n.op is not None}
    assert boxes["wetgrass_given_sprinkler_rain"].n_in == 2
    assert boxes["wetgrass_given_sprinkler_rain"].n_out == 1


def test_a_variable_whose_bits_are_read_separately_is_drawn_as_bit_dots():
    """A `match` selects on single bits, so one dot would misdraw the selections."""
    from pylea.examples.simple import Color

    @DPP
    def selected() -> Bool:
        c = Color.random({"R": 0.5, "G": 0.25, "B": 0.25})
        return c.match({"R": flip(0.9), "G": flip(0.5), "B": flip(0.1)})

    sketch = selected.diagram().sketch
    body = selected.program().main
    assert len([n for n in sketch.nodes if n.wire is not None]) == len(body.bits)


def test_a_returned_variable_drawn_as_bit_dots_labels_each_bit_by_position():
    from pylea import Enum

    class Colour(Enum):
        elements = ["R", "G", "B"]

    @DPP
    def leaky() -> Colour:
        shade = Colour.uniform()
        observe(shade.bits()[0])
        return shade

    svg = leaky.diagram().svg()
    assert ">shade[0]<" in svg and ">shade[1]<" in svg


def _accepted_by_definition(b, candidates, in_segments, out_segments, joined):
    """The rule of `_accepted_groups`, read directly: every candidate group checked
    against every list of wires on its own."""
    groups, taken = [], set()
    for g in map(tuple, candidates):
        if len(g) < 2 or len(set(g)) != len(g):
            continue
        if any(w in taken for w in g) or not all(0 <= w < len(b.bits) for w in g):
            continue
        groups.append(g)
        taken.update(g)

    apart = {j for ops in joined for j in ops}
    lists = ([b.inputs, b.outputs]
             + [ws for j, op in enumerate(b.ops) if j not in apart for ws in (op.args, op.outs)]
             + [tuple(w for j in ops for w in b.ops[j].outs) for ops in joined])

    def together(g):
        for ws in lists:
            i = 0
            while i < len(ws):
                if ws[i] in g:
                    if tuple(ws[i:i + len(g)]) != g:
                        return False
                    i += len(g)
                else:
                    i += 1
        return True

    accepted = {g for g in groups if together(g)}
    changed = True
    while changed:
        changed = False
        for segment in in_segments + out_segments:
            if len(segment) < 2 or segment in accepted:
                continue
            for g in [g for g in accepted if set(segment) & set(g)]:
                accepted.discard(g)
                changed = True
    return [g for g in groups if g in accepted]


def _random_grouping(seed: int):
    """A definition of tables over runs of earlier results, with candidate groups, interface
    segments and tuples of joined assignments, all drawn at random from seed `seed`."""
    import numpy as np

    rng = random.Random(seed)
    b = BodyBuilder()
    inputs = [b.fresh() for _ in range(rng.randint(0, 5))]
    runs, pool = [tuple(inputs)] if inputs else [], list(inputs)
    for _ in range(rng.randint(2, 20)):
        if pool and rng.random() < 0.5:
            run = rng.choice(runs)
            start = rng.randrange(len(run))
            args = run[start:start + rng.randint(1, 3)]
        else:
            args = tuple(rng.choice(pool) for _ in range(rng.randint(0, min(3, len(pool)))))
        n_out = rng.randint(1, 3)
        table = np.full((2 ** n_out, 2 ** len(args)), 1 / 2 ** n_out)
        outs = b.emit(Prim.from_matrix(table), args)
        runs.append(outs)
        pool += outs
    outputs = [w for run in rng.sample(runs, min(3, len(runs))) for w in run]
    body = b.finish(tuple(inputs), tuple(outputs))

    candidates = [run for run in runs if len(run) > 1 and rng.random() < 0.8]
    candidates += [tuple(rng.sample(pool, 2)) for _ in range(rng.randint(0, 3))]
    candidates += [(pool[0], pool[0]), (len(body.bits), pool[0])]
    rng.shuffle(candidates)

    def widths(wires):
        cut, left = [], len(wires)
        while left:
            cut.append(rng.randint(1, left))
            left -= cut[-1]
        return cut

    ops = list(range(len(body.ops)))
    rng.shuffle(ops)
    joined = [tuple(sorted(ops[k:k + 2])) for k in range(0, min(4, len(ops) - 1), 2)]
    return (body, candidates, segments(body.inputs, widths(body.inputs), "input"),
            segments(body.outputs, widths(body.outputs), "output"), joined)


def test_grouping_accepts_the_groups_the_definition_describes():
    for seed in range(300):
        body, candidates, ins, outs, joined = _random_grouping(seed)
        assert (_accepted_groups(body, candidates, ins, outs, joined)
                == _accepted_by_definition(body, candidates, ins, outs, joined)), seed


class _Counted(tuple):
    """A tuple counting the entries read from it, over every instance."""

    reads = 0

    def __getitem__(self, key):
        found = super().__getitem__(key)
        _Counted.reads += len(found) if isinstance(key, slice) else 1
        return found

    def __iter__(self):
        for value in super().__iter__():
            _Counted.reads += 1
            yield value


def test_grouping_reads_each_list_of_wires_a_bounded_number_of_times():
    """One assignment returning 2,000 pairs and another reading them all: checking each
    pair against both lists on its own would read four million entries."""
    from types import SimpleNamespace

    pairs = [(2 * k, 2 * k + 1) for k in range(2000)]
    wires = _Counted(range(4000))
    body = SimpleNamespace(bits=[1] * 4000, inputs=_Counted(), outputs=wires,
                           ops=[SimpleNamespace(args=_Counted(), outs=wires),
                                SimpleNamespace(args=wires, outs=_Counted())])
    _Counted.reads = 0
    accepted = _accepted_groups(body, pairs, [], [(w,) for w in range(4000)])
    assert accepted == pairs
    assert _Counted.reads <= 2 * 3 * len(wires)


# --- vertical spacing ---------------------------------------------------------------

def _bands(placement):
    """Every step of every wire, grouped by the pair of heights it spans."""
    from collections import defaultdict

    spans = defaultdict(list)
    for wire in placement.routes:
        for (x0, y0), (x1, y1) in zip(wire.points, wire.points[1:]):
            spans[(round(y0, 3), round(y1, 3))].append(abs(x1 - x0))
    return spans


def _furthest_reaching(placement):
    """The rise and the longest run of the band holding the wire that reaches furthest."""
    (low, high), runs = max(_bands(placement).items(), key=lambda band: max(band[1]))
    return high - low, max(runs)


def test_a_rank_whose_wires_reach_far_is_given_more_room_than_one_whose_wires_do_not():
    """A fixed gap makes the angle of a wire depend on the width of the picture."""
    @DPP
    def hub_and_spokes(xs, types=(Bool,) * 16) -> Bool:
        hub = xs[0]
        gates = [hub & x for x in xs[1:]]        # one wire reaching the width of the rank
        combined = gates[0]
        for gate in gates[1:]:
            combined = combined | gate
        return combined

    sketch = sketch_of(hub_and_spokes.program())
    spread, packed = place(sketch), place(sketch, Style(slope=0.0))
    assert spread.width == packed.width, "only the vertical spacing should differ"
    assert spread.height > packed.height

    stretched, run = _furthest_reaching(spread)
    fixed, same_run = _furthest_reaching(packed)
    assert run == same_run, "the two should differ in the rise alone"
    assert stretched > 4 * fixed, "the band carrying the longest wire should be opened up"


def test_a_picture_whose_wires_stay_close_is_spaced_no_more_than_the_least_gap():
    """The rule must not stretch a diagram whose angles are already fine."""
    @DPP
    def chain() -> Bool:
        return ~~~flip(0.5)

    narrow = sketch_of(chain.program())
    assert place(narrow).height == place(narrow, Style(slope=0.0)).height


def test_no_gap_grows_past_what_the_stretch_allows():
    from pylea.examples import simple

    style = Style()
    for model in (simple.tree_structured_fault_tree_larger, has_disease, disease_test):
        placement = place(sketch_of(model.program()), style)
        ys = sorted({round(y, 3) for y in placement.y})
        for below, above in zip(ys, ys[1:]):
            assert above - below <= style.rank_gap * style.max_stretch + style.box_height


def _path_points(wire):
    """Every point the drawn path touches, its control points included."""
    return [(seg[i], seg[i + 1]) for seg in wire.path for i in range(1, len(seg), 2)]


@pytest.mark.parametrize("name,sketch", pictures())
def test_a_wire_never_swings_wide_of_the_line_it_follows(name, sketch):
    """A curve fitted through the points would overshoot a corner whose two sides are of
    very different lengths, which is the corner a wire crossing many ranks turns."""
    placement = place(sketch)
    for wire in placement.routes:
        xs = [x for x, _ in wire.points]
        for x, _ in _path_points(wire):
            assert min(xs) - 1e-6 <= x <= max(xs) + 1e-6, f"{name}: a wire swings wide"


@pytest.mark.parametrize("name,sketch", pictures())
def test_a_drawn_wire_never_turns_back_on_itself(name, sketch):
    placement = place(sketch)
    for wire in placement.routes:
        ys = [y for _, y in _path_points(wire)]
        assert ys == sorted(ys, reverse=sketch.downward), f"{name}: a wire turns back"


def test_a_constant_is_drawn_as_its_value_not_as_its_bit_pattern():
    """The bit pattern only reads as the value for a type whose values are 0, 1, 2, ..."""
    from pylea import Enum

    class Fleet(Enum):
        elements = list(range(1, 9))

    @DPP
    def guessed() -> Fleet:
        return Fleet.const(5)

    assert [n.label for n in guessed.diagram().sketch.nodes if n.op is not None] == ["5"]


@pytest.mark.parametrize("theme,expected", [
    ("auto", True), ("light", False), ("dark", False),
])
def test_the_colours_follow_the_reader_only_where_the_page_does(theme, expected):
    """A file loaded through an img element cannot see the colours of the page holding it."""
    svg = has_disease.diagram().svg(theme)
    assert ("prefers-color-scheme" in svg) is expected


@pytest.mark.parametrize("theme", ["auto", "light", "dark"])
def test_the_picture_carries_its_own_background(theme):
    """Without it the ink of one theme can land on the page of the other."""
    svg = has_disease.diagram().svg(theme)
    root = ElementTree.fromstring(svg)
    drawn = [child for child in root if not child.tag.endswith("style")]
    paper = drawn[0]
    assert paper.tag.endswith("rect") and paper.attrib["class"] == "paper"
    box = [float(v) for v in root.attrib["viewBox"].split()]
    assert [float(paper.attrib[k]) for k in ("x", "y", "width", "height")] == box
    assert ".pylea .paper{fill:var(--paper)" in svg


def test_a_fixed_theme_picks_the_colours_it_names():
    assert "#16181d" in has_disease.diagram().svg("light")
    assert "#e6e8ec" in has_disease.diagram().svg("dark")


def test_a_theme_neither_backend_knows_is_refused():
    with pytest.raises(ValueError, match="'auto', 'light' or 'dark'"):
        has_disease.diagram().svg("sepia")


def _random_gates(seed: int) -> Program:
    """A definition of 5 to 60 coin flips and gates over up to four inputs, returning
    three of its variables."""
    rng = random.Random(seed)
    b = BodyBuilder()
    inputs = [b.fresh() for _ in range(rng.randint(0, 4))]
    pool = inputs + list(b.emit(Prim.flip(0.5), ()))
    for _ in range(rng.randint(5, 60)):
        r = rng.random()
        if r < 0.15:
            pool += b.emit(Prim.flip(round(rng.random(), 2) or 0.5), ())
        elif r < 0.85:
            gate = Prim.and_() if r < 0.5 else Prim.or_()
            pool += b.emit(gate, (rng.choice(pool), rng.choice(pool)))
        else:
            pool += b.emit(Prim.not_(), (rng.choice(pool),))
    return Program((b.finish(tuple(inputs), tuple(rng.sample(pool, 3))),), 0)


def _layered(sketch, style):
    glyphs = tuple(glyph_of(n.kind, n.label, n.n_in, n.n_out, style) for n in sketch.nodes)
    g = _Layered(sketch, _ranks(sketch), glyphs, style)
    _order(g, None)
    return g


def _too_close(g, x, style) -> list[tuple[int, int]]:
    """The neighbours on a rank of `g`, dummies included, closer than their separation."""
    return [(a, b) for nodes in g.layers.values() for a, b in zip(nodes, nodes[1:])
            if x[b] - x[a] < coordinates._separation(g, a, b, style) - 1e-6]


def _layout_corpus():
    yield from ((f"seed {seed}", sketch_of(_random_gates(seed))) for seed in range(200))
    for name, dpp in (("decisive", DECISIVE),
                      ("pivotal Wyoming", electoral_college.pivotal("Wyoming")),
                      ("wet stones", percolation.wet_stones(10, 40, 0.5))):
        yield name, dpp.diagram().sketch


def test_neighbours_on_a_rank_are_never_closer_than_their_separation():
    style = Style()
    for name, sketch in _layout_corpus():
        g = _layered(sketch, style)
        assert not _too_close(g, coordinates.coordinates(g, style), style), name


def test_each_compaction_pass_keeps_the_separation():
    style = Style()
    for name, sketch in _layout_corpus():
        g = _layered(sketch, style)
        conflicted = coordinates._crossed_by_a_short_wire(g)
        for upward in (False, True):
            for rightward in (False, True):
                x = coordinates._one_pass(g, style, conflicted, upward, rightward)
                assert not _too_close(g, x, style), (name, upward, rightward)


def test_the_electoral_college_diagram_has_no_overlapping_boxes():
    diagram = DECISIVE.diagram()
    p, nodes = diagram.placement, diagram.sketch.nodes
    boxes = [i for i, n in enumerate(nodes)
             if n.kind not in (GlyphKind.DOT, GlyphKind.BOUNDARY)]
    for a in boxes:
        for b in boxes:
            ga, gb = p.glyphs[a], p.glyphs[b]
            if a < b:
                apart_x = abs(p.x[a] - p.x[b]) >= (ga.width + gb.width) / 2
                apart_y = abs(p.y[a] - p.y[b]) >= (ga.height + gb.height) / 2
                assert apart_x or apart_y, (nodes[a].label, nodes[b].label)


def _passes(sketch, style):
    g = _layered(sketch, style)
    conflicted = coordinates._crossed_by_a_short_wire(g)
    return g, [(rightward, coordinates._one_pass(g, style, conflicted, upward, rightward))
               for upward in (False, True) for rightward in (False, True)]


def test_the_median_layout_is_kept_where_the_passes_are_about_equally_wide():
    """Each box then sits centred over what it reads, rather than to one side of it."""
    style = Style()
    g, passes = _passes(bioassay.lethal_dose.diagram().sketch, style)
    widths = [max(x) - min(x) for _, x in passes]
    narrowest = passes[widths.index(min(widths))][1]
    x = coordinates.coordinates(g, style)
    assert x != narrowest and max(x) - min(x) <= 3 * min(widths)


def test_one_narrow_pass_among_wide_ones_is_used_alone():
    """A long running total is a column in one pass and a staircase in the others."""
    style = Style()
    sketch = electoral_college.pivotal("Wyoming").diagram().sketch
    g, passes = _passes(sketch, style)
    widths = sorted(max(x) - min(x) for _, x in passes)
    assert widths[1] > 3 * widths[0]
    assert place(sketch, style).width < 1000


def test_showing_without_a_browser_says_where_the_file_is(monkeypatch):
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match=r"written to .*\.svg") as raised:
        has_disease.diagram().show()
    path = str(raised.value).split("written to ")[1]
    assert open(path, encoding="utf-8").read().startswith("<svg")


def test_a_call_is_labelled_with_the_name_it_was_called_by_at_its_site():
    """Two callees with equal bodies are one body, which one name cannot label."""
    prog = has_disease.program()
    calls = [j for j, op in enumerate(prog.main.ops) if not isinstance(op.callee, Prim)]
    labels = [n.label for n in sketch_of(prog, body_names={0: "sensor_a"},
                                         op_names={calls[1]: "sensor_b"}).nodes
              if n.op in calls]
    assert labels == ["sensor_a", "sensor_b"]


def _bit_by_bit(value: int, bits: int) -> Program:
    """A definition returning `value` as `bits` one-bit constants, most significant first."""
    b = BodyBuilder()
    outs = [b.emit(Prim.const((value >> (bits - 1 - i)) & 1, 1), ())[0] for i in range(bits)]
    return Program((b.finish((), tuple(outs)),), 0)


def test_a_constant_recorded_bit_by_bit_by_one_call_is_one_box_with_its_value():
    prog = _bit_by_bit(1000, 12)
    everything = tuple(range(12))
    sketch = sketch_of(prog, wire_groups=[everything], output_widths=(12,),
                       op_groups=[everything])
    boxes = [n for n in sketch.nodes if n.op is not None]
    assert [(n.label, n.n_out) for n in boxes] == [("1000", 1)]
    assert len([n for n in sketch.nodes if n.kind is GlyphKind.DOT]) == 1


def test_constants_not_recorded_by_one_call_stay_separate_boxes():
    """A pair of two separate one-bit constants is the same body as a two-bit constant
    recorded bit by bit."""
    sketch = sketch_of(_bit_by_bit(3, 2), wire_groups=[(0, 1)], output_widths=(2,))
    assert [n.label for n in sketch.nodes if n.op is not None] == ["1", "1"]


def test_a_wide_constant_is_drawn_as_one_box():
    from pylea import UInt

    class Narrow(UInt):
        width = 8

    class Wide(UInt):
        width = 12

    def boxes(data_type):
        @DPP
        def constant() -> data_type:
            return data_type.const(200)

        return [n.label for n in constant.diagram().sketch.nodes if n.op is not None]

    assert boxes(Wide) == boxes(Narrow) == ["200"]
