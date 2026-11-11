"""What a drawn dot diagram must show, whatever the model.

The properties here are the ones a reader relies on: a dot for every variable, a box for
every assignment, a wire for every argument and every result, and nothing overlapping
anything else.  The degenerate models are covered too, since a body with no ops or an output
naming an input has no box to hang the picture on.

The properties of the layout itself hold for any sketch, so they run over the dependency
graph of each model as well; what that picture must show is in `test.draw.test_dependency`.
"""
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree

import pytest

from pylea import DPP, Bool, flip, observe
from pylea.draw import diagram_of
from pylea.draw.glyph import GlyphKind, Style
from pylea.draw.layout import _inversions, place
from pylea.draw.sketch import display_name, sketch_of
from pylea.examples import simple
from pylea.examples.disease import has_disease, test as disease_test

MODELS = [simple.empty_dpp, simple.identity_dpp, simple.unused_input_dpp,
          simple.unused_output_dpp, simple.two_outputs_dpp, simple.dpp_with_observe,
          simple.tree_structured_fault_tree_larger, simple.color_random_then_match,
          disease_test, has_disease]


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
    bundle = diagram_of(has_disease.program())._repr_mimebundle_()
    assert bundle["image/svg+xml"].startswith("<svg")
    assert bundle["text/plain"].startswith("<diagram")


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
    @DPP
    def late(a: Bool, b: Bool) -> Bool:
        deep = a & b
        return deep & flip(0.5)

    sketch = sketch_of(late.program())
    placement = place(sketch)
    flip_at = next(i for i, n in enumerate(sketch.nodes) if n.kind is GlyphKind.FLIP)
    inputs = [i for i, n in enumerate(sketch.nodes) if n.wire in late.program().main.inputs]
    assert placement.y[flip_at] > max(placement.y[i] for i in inputs)


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
    from pylea.examples.simple import color_random_then_match

    sketch = color_random_then_match.diagram().sketch
    body = color_random_then_match.program().main
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


def test_a_fixed_theme_picks_the_colours_it_names():
    assert "#16181d" in has_disease.diagram().svg("light")
    assert "#e6e8ec" in has_disease.diagram().svg("dark")


def test_a_theme_neither_backend_knows_is_refused():
    with pytest.raises(ValueError, match="'auto', 'light' or 'dark'"):
        has_disease.diagram().svg("sepia")
