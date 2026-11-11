"""What a drawn dependency graph must show.

One box per variable the model is given or returns, one arrow per dependency between two of
them, and a name on every box.  The geometry these pictures share with the dot diagram is
checked in `test.draw.test_draw`, which places both kinds.
"""
import xml.etree.ElementTree as ElementTree
from dataclasses import replace

import pytest

from pylea import DPP, Bool, Enum, flip, observe
from pylea.draw import dependency_graph_of
from pylea.draw.dependency import dependency_sketch_of
from pylea.draw.glyph import GlyphKind, Style, _within
from pylea.draw.layout import place
from pylea.examples import simple
from pylea.examples.sprinkler import (rain_given_wet_or_damp_grass, sprinkler_given_cloudy,
                                      sprinkler_network)


def wires_between(sketch):
    """Every wire of `sketch`, as the pair of labels it joins."""
    return sorted((sketch.nodes[e.src].label, sketch.nodes[e.dst].label)
                  for e in sketch.edges)


def test_a_network_draws_as_the_network_it_was_written_from():
    sketch = sprinkler_network.dependency_graph().sketch
    assert [n.label for n in sketch.nodes] == ["cloudiness", "sprinkler_state", "rain",
                                              "grass_wetness"]
    assert wires_between(sketch) == [("cloudiness", "rain"),
                                     ("cloudiness", "sprinkler_state"),
                                     ("rain", "grass_wetness"),
                                     ("sprinkler_state", "grass_wetness")]


def test_a_variable_reached_only_through_another_has_no_wire_of_its_own():
    """The wetness is computed from the rain, which is computed from the cloudiness."""
    joined = wires_between(sprinkler_network.dependency_graph().sketch)
    assert ("cloudiness", "grass_wetness") not in joined


def test_returning_what_a_sub_model_returned_draws_the_same_graph():
    """The relation between two results of one call lies inside the model it calls."""
    @DPP
    def wrapper():
        return sprinkler_network()

    def shape(sketch):
        return sorted((e.src, e.dst) for e in sketch.edges)

    assert (shape(wrapper.dependency_graph().sketch)
            == shape(sprinkler_network.dependency_graph().sketch))
    assert [n.label for n in wrapper.dependency_graph().sketch.nodes] == [
        "Cloudiness", "SprinklerState", "Rain", "GrassWetness"]


def test_a_variable_the_model_is_given_is_drawn_with_a_second_outline():
    graph = sprinkler_given_cloudy.dependency_graph()
    kinds = [n.kind for n in graph.sketch.nodes]
    assert kinds == [GlyphKind.PARAMETER, GlyphKind.VARIABLE]
    given, computed = graph.placement.glyphs
    assert len(given.shapes) == 2 and len(computed.shapes) == 1


def test_a_box_carries_the_name_the_model_returned_the_variable_as():
    svg = sprinkler_network.dependency_graph().svg()
    assert all(f">{name}<" in svg
               for name in ("cloudiness", "sprinkler_state", "rain", "grass_wetness"))


def test_a_variable_the_source_gives_no_name_is_labelled_by_its_type():
    """The model returns an expression, which has no name to show."""
    labels = [n.label for n in sprinkler_given_cloudy.dependency_graph().sketch.nodes]
    assert labels == ["c", "SprinklerState"]        # `c` is the model's own parameter


def test_two_variables_that_would_share_a_label_are_told_apart_by_their_index():
    labels = [n.label for n in simple.two_input_output_dpp.dependency_graph().sketch.nodes]
    assert labels == ["x", "y", "Bool 2", "Bool 3"]


def test_a_variable_returned_twice_is_one_box():
    @DPP
    def twice() -> tuple[Bool, Bool]:
        x = flip(0.5)
        return x, x

    assert [n.label for n in twice.dependency_graph().sketch.nodes] == ["x"]


def test_an_input_returned_unchanged_is_one_box_and_is_drawn_as_given():
    sketch = simple.identity_dpp.dependency_graph().sketch
    assert [(n.kind, n.label) for n in sketch.nodes] == [(GlyphKind.PARAMETER, "x")]


def test_a_variable_of_several_bits_is_one_box():
    body = sprinkler_network.program().main
    sketch = sprinkler_network.dependency_graph().sketch
    assert len(body.outputs) == 6 and len(sketch.nodes) == 4


def test_simplifying_a_model_draws_the_same_graph():
    """The passes rewrite the assignments, not what each returned variable comes from."""
    simplified = sprinkler_network.simplified().dependency_graph().sketch
    original = sprinkler_network.dependency_graph().sketch
    assert wires_between(simplified) == wires_between(original)


def test_conditioning_draws_no_wire():
    """An `observe` reads no variable of the interface, so it joins none of them."""
    @DPP
    def conditioned() -> tuple[Bool, Bool]:
        a = flip(0.5)
        b = flip(0.5)
        observe(a | b)
        return a, b

    sketch = conditioned.dependency_graph().sketch
    assert len(sketch.nodes) == 2 and not sketch.edges


def test_an_argument_reaches_the_port_of_the_position_it_was_passed_at():
    sketch = sprinkler_network.dependency_graph().sketch
    boxes = {n.label: i for i, n in enumerate(sketch.nodes)}
    into_grass = sorted(e.dst_port for e in sketch.edges
                        if e.dst == boxes["grass_wetness"])
    out_of_cloud = sorted(e.src_port for e in sketch.edges if e.src == boxes["cloudiness"])
    assert into_grass == [0, 1] and out_of_cloud == [0, 1]
    assert sketch.nodes[boxes["grass_wetness"]].n_in == 2
    assert sketch.nodes[boxes["cloudiness"]].n_out == 2


def test_a_model_returning_nothing_draws_an_empty_picture():
    graph = simple.empty_dpp.dependency_graph()
    assert not graph.sketch.nodes and not graph.sketch.edges
    assert graph.svg().startswith("<svg")


def test_a_model_whose_variables_have_no_order_is_refused():
    """Two variables whose bits are computed in turn from each other form a cycle."""
    class Colour(Enum):
        elements = ["R", "G", "B"]

    @DPP
    def woven() -> tuple[Colour, Colour]:
        first = flip(0.5)
        second = first & flip(0.5)
        third = flip(0.5)
        fourth = third & flip(0.5)
        return Colour(first, fourth), Colour(second, third)

    with pytest.raises(ValueError, match="no order to draw"):
        woven.dependency_graph()


def test_a_query_over_a_network_draws_only_what_it_returns():
    """The variables of the network are internal to it, so the graph is the one answer."""
    graph = rain_given_wet_or_damp_grass.dependency_graph()
    assert len(graph.sketch.nodes) == 1 and not graph.sketch.edges


# --- the arrows -----------------------------------------------------------------------

def test_a_variable_is_drawn_above_the_ones_computed_from_it():
    """Which is how a network is usually drawn, and the other way round from a diagram."""
    graph = sprinkler_network.dependency_graph()
    height = {n.label: y for n, y in zip(graph.sketch.nodes, graph.placement.y)}
    assert height["cloudiness"] > height["rain"] > height["grass_wetness"]
    assert height["cloudiness"] > height["sprinkler_state"] > height["grass_wetness"]


def test_a_wire_stops_on_the_outline_of_the_box_it_points_at():
    """A wire aimed at a port would stop short of a box it arrives at from the side."""
    graph = sprinkler_network.dependency_graph()
    for edge, wire in zip(graph.sketch.edges, graph.placement.routes):
        glyph = graph.placement.glyphs[edge.dst]
        centre = (graph.placement.x[edge.dst], graph.placement.y[edge.dst])
        end, before = wire.points[-1], wire.points[-2]
        along = _unit(before, end)

        def inside(step, end=end, along=along, centre=centre, glyph=glyph):
            return _within(end[0] + along[0] * step - centre[0],
                           end[1] + along[1] * step - centre[1],
                           glyph.width / 2, glyph.height / 2, glyph.corner)

        assert not inside(-0.05) and inside(0.05), "the end is not on the outline"


def test_a_wire_arriving_from_the_side_lands_on_the_side_it_comes_from():
    """The head of a wire ending on the edge it runs along would lie flat against it."""
    @DPP
    def wide() -> tuple[Bool, Bool, Bool]:
        a_source_with_a_long_name = flip(0.5)
        one_reading = a_source_with_a_long_name & flip(0.5)
        another_reading = a_source_with_a_long_name & flip(0.5)
        return a_source_with_a_long_name, one_reading, another_reading

    graph = wide.dependency_graph()
    for edge, wire in zip(graph.sketch.edges, graph.placement.routes):
        glyph = graph.placement.glyphs[edge.dst]
        across = abs(wire.points[-1][0] - graph.placement.x[edge.dst])
        down = abs(wire.points[-1][1] - graph.placement.y[edge.dst])
        assert across > glyph.width / 2 - glyph.corner or down > glyph.height / 2 - 1e-6, (
            "the wire ends in the middle of an edge, not on the corner it comes round")



def test_every_wire_ends_in_an_arrow_head_at_the_variable_it_points_into():
    placement = sprinkler_network.dependency_graph().placement
    assert placement.routes
    for wire in placement.routes:
        assert wire.head[0][1:] == wire.points[-1], "the tip is not at the end of the wire"
        assert wire.head[-1] == ("Z",), "the head is not a closed outline"


def test_an_arrow_head_points_the_way_its_wire_arrives():
    """A head across the wire, or pointing back down it, would say the wrong thing."""
    for wire in sprinkler_network.dependency_graph().placement.routes:
        tip, before = wire.points[-1], wire.points[-2]
        base = [(seg[1], seg[2]) for seg in wire.head[1:3]]
        middle = ((base[0][0] + base[1][0]) / 2, (base[0][1] + base[1][1]) / 2)
        arriving = _unit(before, tip)
        assert abs(_unit(middle, tip)[0] - arriving[0]) < 1e-9
        assert abs(_unit(middle, tip)[1] - arriving[1]) < 1e-9


def _unit(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    return dx / length, dy / length


def test_a_dot_diagram_carries_no_arrow_heads():
    """The two pictures share the layout, so the dot diagram must be left as it was."""
    diagram = simple.tree_structured_fault_tree_larger.diagram()
    assert not any(wire.head for wire in diagram.placement.routes)


def test_the_ranks_of_a_directed_picture_leave_room_for_the_heads():
    """A head as long as the gap between two ranks leaves them reading as joined.

    The names are short, so the wires reach nowhere and the gap is the least one; where
    they reach far the gap already follows how far, and needs nothing added.
    """
    @DPP
    def chain() -> tuple[Bool, Bool]:
        a = flip(0.5)
        b = a & flip(0.5)
        return a, b

    style = Style()
    sketch = chain.dependency_graph().sketch
    spread = place(sketch, style)
    packed = place(replace(sketch, directed=False), style)
    assert spread.height >= packed.height + style.arrow_head - 1e-6


def test_both_backends_draw_one_head_per_wire():
    graph = sprinkler_network.dependency_graph()
    assert graph.svg().count('class="solid"') == len(graph.placement.routes)
    assert graph.tikz().count(r"\path[pylea solid]") == len(graph.placement.routes)


# --- the picture itself ---------------------------------------------------------------

def test_the_svg_is_well_formed_and_carries_the_title():
    svg = sprinkler_network.dependency_graph().svg()
    assert ElementTree.fromstring(svg).tag.endswith("svg")
    assert ">sprinkler_network<" in svg


def test_both_backends_draw_the_same_number_of_wires():
    graph = sprinkler_network.dependency_graph()
    assert graph.tikz().count(r"\draw[pylea wire]") == len(graph.placement.routes)
    assert graph.svg().count('class="wire"') == len(graph.placement.routes)


def test_saving_and_showing_work_as_they_do_for_a_dot_diagram(tmp_path):
    graph = sprinkler_network.dependency_graph()
    written = graph.save(tmp_path / "graph.svg")
    assert written.read_text(encoding="utf-8") == graph.svg()
    assert graph._repr_mimebundle_()["image/svg+xml"].startswith("<svg")


def test_a_position_with_no_name_is_labelled_by_the_interface_it_is_on():
    """Drawing a program on its own offers no names, only the shape of the interface."""
    sketch = dependency_sketch_of(sprinkler_network.program(), output_widths=(1, 1, 2, 2))
    assert [n.label for n in sketch.nodes] == ["out 0", "out 1", "out 2", "out 3"]


def test_widths_that_do_not_cover_the_interface_are_refused():
    with pytest.raises(ValueError, match="output widths"):
        dependency_graph_of(sprinkler_network.program(), output_widths=(1, 1))
