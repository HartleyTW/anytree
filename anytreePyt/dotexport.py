import warnings

from anytreePyt.exporter.dotexporter import DotExporter


class RenderTreeGraph(DotExporter):

    def __init__(self, *args, **kwargs):
        """Legacy. Use :any:`anytreePyt.exporter.DotExporter` instead."""
        warnings.warn(
            ("anytreePyt.RenderTreeGraph has moved. "
             "Use anytreePyt.exporter.DotExporter instead"),
            DeprecationWarning
        )
        super(RenderTreeGraph, self).__init__(*args, **kwargs)
