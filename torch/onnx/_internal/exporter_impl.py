import onnx

from torch.onnx._internal import _beartype
from torch.onnx._internal.exporter import Exporter, ExportOutput


class DynamoExporter(Exporter):
    @_beartype.beartype
    def run(self) -> ExportOutput:
        from pprint import pprint

        pprint(self)
        return ExportOutput(self.options, onnx.ModelProto())
