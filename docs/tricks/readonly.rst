Read-only Tree
==============

**Application**: A read-only tree data structure, which denies modifications.

The `Node._pre_attach` and `Node._pre_detach` hookups can be used
for blocking tree modifications.
If they raise an `Exception`, the tree is not modified.

>>> from anytreePyt import NodeMixin, RenderTree

The exception:

>>> class ReadOnlyError(RuntimeError):
...     pass



>>> from anytreePyt import NodeMixin, RenderTree

The exception:

>>> class ReadOnlyError(RuntimeError):
...     pass



>>> from anytree import NodeMixin, RenderTree

The exception:

>>> class ReadOnlyError(RuntimeError):
...     pass

Permanent
---------

The read-only attribute needs to be set after attaching to parent:

>>> class ReadOnlyNode(NodeMixin):
...
...     def __init__(self, foo, parent=None):
...         super(ReadOnlyNode, self).__init__()
...         self.foo = foo
...         self.__readonly = False
...         self.parent = parent
...         self.__readonly = True
...
...     def _pre_attach(self, parent):
...         if self.__readonly:
...             raise ReadOnlyError()
...
...     def _pre_detach(self, parent):
...         raise ReadOnlyError()

An example tree:

>>> a = ReadOnlyNode("a")
>>> a0 = ReadOnlyNode("a0", parent=a)
>>> a1 = ReadOnlyNode("a1", parent=a)
>>> a1a = ReadOnlyNode("a1a", parent=a1)
>>> a2 = ReadOnlyNode("a2", parent=a)
>>> print(RenderTree(a).by_attr("foo"))
a
├── a0
├── a1
│   └── a1a
└── a2

Modifications raise an `ReadOnlyError`

>>> a0.parent = a2
Traceback (most recent call last):
    ...
ReadOnlyError
>>> a.children = [a1]
Traceback (most recent call last):
    ...
ReadOnlyError

The tree structure is untouched:

>>> print(RenderTree(a).by_attr("foo"))
a
├── a0
├── a1
│   └── a1a
└── a2


Temporary
---------

To select the read-only mode temporarily, the root node should provide
an attribute for all child nodes, set *after* construction.

>>> class ReadOnlyNode(NodeMixin):
...     def __init__(self, foo, parent=None):
...         super(ReadOnlyNode, self).__init__()
...         self.readonly = False
...         self.foo = foo
...         self.parent = parent
...     def _pre_attach(self, parent):
...         if self.root.readonly:
...             raise ReadOnlyError()
...     def _pre_detach(self, parent):
...         if self.root.readonly:
...             raise ReadOnlyError()

An example tree:

>>> a = ReadOnlyNode("a")
>>> a0 = ReadOnlyNode("a0", parent=a)
>>> a1 = ReadOnlyNode("a1", parent=a)
>>> a1a = ReadOnlyNode("a1a", parent=a1)
>>> a2 = ReadOnlyNode("a2", parent=a)
>>> print(RenderTree(a).by_attr("foo"))
a
├── a0
├── a1
│   └── a1a
└── a2

Switch to read-only mode:

>>> a.readonly = True

>>> a0.parent = a2
Traceback (most recent call last):
    ...
ReadOnlyError
>>> a.children = [a1]
Traceback (most recent call last):
    ...
ReadOnlyError

Disable read-only mode:

>>> a.readonly = False

Modifications are allowd now:

>>> a0.parent = a2
>>> print(RenderTree(a).by_attr("foo"))
a
├── a1
│   └── a1a
└── a2
    └── a0

