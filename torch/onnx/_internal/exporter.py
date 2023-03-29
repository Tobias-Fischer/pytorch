# necessary to surface onnx.ModelProto through ExportOutput:
from __future__ import annotations

import abc
import io
import logging
from typing import (
    Any,
    Callable,
    Final,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import torch

from torch.onnx._internal import _beartype

# We can only import onnx from this module in a type-checking context to ensure that
# 'import torch.onnx' continues to work without having 'onnx' installed. We fully
# 'import onnx' inside of dynamo_export (by way of _assert_dependencies).
if TYPE_CHECKING:
    import onnx


_DEFAULT_OPSET_VERSION: Final[int] = 18
"""The default ONNX opset version the exporter will use if one is not specified explicitly
through ``ExportOptions``. This should NEVER be accessed outside of this module! Users
should reference ``ExportOptions.opset_version``."""


@_beartype.beartype
class ExportOptions:
    """Options to influence the TorchDynamo ONNX exporter."""

    opset_version: Optional[int] = None
    """The ONNX opset version the exporter should target. Defaults to the latest
    supported ONNX opset version. The default version will increment over time as
    ONNX continues to evolve."""

    dynamic_shapes: Optional[bool] = None
    """Shape information hint for input/output tensors.

    - ``None``: the exporter determines the most compatible setting.
    - ``True``: all input shapes are considered dynamic.
    - ``False``: all input shapes are considered static."""

    logger: Optional[logging.Logger] = None
    """The logger for the ONNX exporter to use. Defaults to creating a child
    logger named "torch.onnx" under the current logger (as returned by
    :py:meth:`logging.getLogger`)."""

    def __init__(
        self,
        *,
        opset_version: Optional[int] = None,
        dynamic_shapes: Optional[bool] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.opset_version = opset_version
        self.dynamic_shapes = dynamic_shapes
        self.logger = logger


@runtime_checkable
class ExportOutputSerializer(Protocol):
    """Protocol for serializing an ONNX graph into a specific format (e.g. Protobuf)."""

    def serialize(
        self, export_output: ExportOutput, destination: io.BufferedIOBase
    ) -> None:
        ...


class ProtobufExportOutputSerializer:
    """Serializes ONNX graph as Protobuf."""

    @_beartype.beartype
    def serialize(
        self, export_output: ExportOutput, destination: io.BufferedIOBase
    ) -> None:
        import onnx

        if not isinstance(export_output.model_proto, onnx.ModelProto):
            raise ValueError("export_output.ModelProto is not an onnx.ModelProto")
        destination.write(export_output.model_proto.SerializeToString())


class ExportOutput:
    """An in-memory representation of a PyTorch model that has been exported to ONNX."""

    _export_options: Final[ExportOptions]
    _model_proto: Final[onnx.ModelProto]

    @_beartype.beartype
    def __init__(self, export_options: ExportOptions, model_proto: onnx.ModelProto):
        self._export_options = export_options
        self._model_proto = model_proto

    @property
    def export_options(self) -> ExportOptions:
        """The options used to produce the exported ONNX model."""

        return self._export_options

    @property
    def model_proto(self) -> onnx.ModelProto:
        """The exported ONNX model as an ``onnx.ModelProto``."""

        return self._model_proto

    @_beartype.beartype
    def save(
        self,
        destination: Union[str, io.BufferedIOBase],
        *,
        serializer: Optional[ExportOutputSerializer] = None,
    ) -> None:
        """Saves the in-memory ONNX model to ``destination`` using specified ``serializer``.
        If no ``serializer`` is specified, the model will be serialized as Protobuf."""

        if serializer is None:
            serializer = ProtobufExportOutputSerializer()
        if isinstance(destination, str):
            with open(destination, "wb") as f:
                serializer.serialize(self, f)
        else:
            serializer.serialize(self, destination)


class Exporter(abc.ABC):
    @_beartype.beartype
    def __init__(
        self,
        options: ExportOptions,
        model: torch.nn.Module,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ):
        self.options = options
        self.model = model
        self.model_args = model_args
        self.model_kwargs = model_kwargs

    @abc.abstractmethod
    def run(self) -> ExportOutput:
        pass

    @property
    def logger(self) -> logging.Logger:
        # options.logger will always be resolved to an instance when constructing
        assert isinstance(self.options.logger, logging.Logger)
        return self.options.logger


@_beartype.beartype
def _resolve_export_options(options: Optional[ExportOptions]) -> ExportOptions:
    if options is None:
        options = ExportOptions()

    T = TypeVar("T")

    def resolve(value: Optional[T], fallback: Callable[[], T]):
        return fallback() if value is None else value

    return ExportOptions(
        opset_version=resolve(options.opset_version, lambda: _DEFAULT_OPSET_VERSION),
        dynamic_shapes=options.dynamic_shapes,
        logger=resolve(
            options.logger, lambda: logging.getLogger().getChild("torch.onnx")
        ),
    )


@_beartype.beartype
def dynamo_export(
    model: torch.nn.Module,
    /,
    *model_args,
    export_options: Optional[ExportOptions] = None,
    **model_kwargs,
) -> ExportOutput:
    """Export a torch.nn.Module to an ONNX graph.

    Args:
        model: The PyTorch model to be exported to ONNX.
        model_args: Positional inputs to ``model``.
        model_kwargs: Keyword inputs to ``model``.
        export_options: Options to influence the export to ONNX.

    Returns:
        An in-memory representation of the exported ONNX model.

    Example:
    ::

        import torch.onnx
        torch.onnx.dynamo_export(
            my_nn_module,
            torch.randn(2, 2, 2), # positional input 1
            torch.randn(2, 2, 2), # positional input 2
            my_nn_module_attribute="hello", # keword input
            export_options=ExportOptions(
                opset_version=17,
            )
        ).save("my_model.onnx")
    """

    resolved_export_options = _resolve_export_options(export_options)

    def assert_dependencies():
        # this has been resolved above, so will not be 'None'
        assert (opset_version := resolved_export_options.opset_version) is not None

        # TODO: each of these imports/assertions should be turned into
        # an actionable diagnostic, e.g.:
        # - Please `pip install onnxscript_torchlib`
        # - You have specified an opset version X that is newer than the
        #   supported opset version Y
        # - etc.
        import onnx

        # mypy is unhappy with 'import onnxscript' right now:
        # - https://github.com/microsoft/onnx-script/issues/572
        import onnxscript  # type: ignore[import]

        assert onnx.defs.onnx_opset_version() >= opset_version
        assert isinstance(
            onnxscript.onnx_opset.all_opsets[("", opset_version)],
            onnxscript.values.Opset,
        )

    assert_dependencies()

    from .exporter_impl import DynamoExporter

    return DynamoExporter(
        options=resolved_export_options,
        model=model,
        model_args=model_args,
        model_kwargs=model_kwargs,
    ).run()
