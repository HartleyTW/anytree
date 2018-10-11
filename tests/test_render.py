# -*- coding: utf-8 -*-
import six

from nose.tools import eq_

import anytreePyt

from helper import eq_str


def test_render_str():
    """Render string cast."""
    root = anytreePyt.Node("root")
    s0 = anytreePyt.Node("sub0", parent=root)
    anytreePyt.Node("sub0B", parent=s0)
    anytreePyt.Node("sub0A", parent=s0)
    anytreePyt.Node("sub1", parent=root)
    r = anytreePyt.RenderTree(root)

    expected = u"\n".join([
        u"Node('/root')",
        u"├── Node('/root/sub0')",
        u"│   ├── Node('/root/sub0/sub0B')",
        u"│   └── Node('/root/sub0/sub0A')",
        u"└── Node('/root/sub1')",
    ])
    eq_str(str(r), expected)


def test_render_repr():
    """Render representation."""
    root = anytreePyt.Node("root")
    anytreePyt.Node("sub", parent=root)
    r = anytreePyt.RenderTree(root)

    if six.PY2:
        expected = ("RenderTree(Node('/root'), style=ContStyle(), "
                    "childiter=<type 'list'>)")
    else:
        expected = ("RenderTree(Node('/root'), style=ContStyle(), "
                    "childiter=<class 'list'>)")
    eq_(repr(r), expected)


def test_render():
    """Rendering."""
    root = anytreePyt.Node("root", lines=["c0fe", "c0de"])
    s0 = anytreePyt.Node("sub0", parent=root, lines=["ha", "ba"])
    s0b = anytreePyt.Node("sub0B", parent=s0, lines=["1", "2", "3"])
    s0a = anytreePyt.Node("sub0A", parent=s0, lines=["a", "b"])
    s1 = anytreePyt.Node("sub1", parent=root, lines=["Z"])

    r = anytreePyt.RenderTree(root, style=anytreePyt.DoubleStyle)
    result = [(pre, node) for pre, _, node in r]
    expected = [
        (u'', root),
        (u'╠══ ', s0),
        (u'║   ╠══ ', s0b),
        (u'║   ╚══ ', s0a),
        (u'╚══ ', s1),
    ]
    eq_(result, expected)

    def multi(root):
        for pre, fill, node in anytreePyt.RenderTree(root):
            yield "%s%s" % (pre, node.lines[0]), node
            for line in node.lines[1:]:
                yield "%s%s" % (fill, line), node
    result = list(multi(root))
    expected = [
        (u'c0fe', root),
        (u'c0de', root),
        (u'├── ha', s0),
        (u'│   ba', s0),
        (u'│   ├── 1', s0b),
        (u'│   │   2', s0b),
        (u'│   │   3', s0b),
        (u'│   └── a', s0a),
        (u'│       b', s0a),
        (u'└── Z', s1),
    ]
    eq_(result, expected)


def test_asciistyle():
    style = anytreePyt.AsciiStyle()
    eq_(style.vertical, u'|   ')
    eq_(style.cont, '|-- ')
    eq_(style.end, u'+-- ')


def test_contstyle():
    style = anytreePyt.ContStyle()
    eq_(style.vertical, u'\u2502   ')
    eq_(style.cont, u'\u251c\u2500\u2500 ')
    eq_(style.end, u'\u2514\u2500\u2500 ')


def test_controundstyle():
    style = anytreePyt.ContRoundStyle()
    eq_(style.vertical, u'\u2502   ')
    eq_(style.cont, u'\u251c\u2500\u2500 ')
    eq_(style.end, u'\u2570\u2500\u2500 ')


def test_doublestyle():
    style = anytreePyt.DoubleStyle()
    eq_(style.vertical, u'\u2551   ')
    eq_(style.cont, u'\u2560\u2550\u2550 ')
    eq_(style.end, u'\u255a\u2550\u2550 ')


def test_by_attr():
    """by attr."""
    root = anytreePyt.Node("root", lines=["root"])
    s0 = anytreePyt.Node("sub0", parent=root, lines=["su", "b0"])
    anytreePyt.Node("sub0B", parent=s0, lines=["sub", "0B"])
    anytreePyt.Node("sub0A", parent=s0)
    anytreePyt.Node("sub1", parent=root, lines=["sub1"])
    eq_(anytreePyt.RenderTree(root).by_attr(),
        u"root\n├── sub0\n│   ├── sub0B\n│   └── sub0A\n└── sub1")
    eq_(anytreePyt.RenderTree(root).by_attr("lines"),
        u"root\n├── su\n│   b0\n│   ├── sub\n│   │   0B\n│   └── \n└── sub1")
