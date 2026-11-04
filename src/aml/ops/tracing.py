"""OpenTelemetry tracing.

Every case produces one trace: workflow step -> agent node -> tool / LLM call. LLM spans carry
the GenAI semantic-convention attributes (gen_ai.*), so any OTLP backend works: Grafana Tempo,
Jaeger, or Langfuse (which ingests OTLP and shows token cost per case).

Off by default (no exporter = no-op spans, near-zero cost). Turn on with:
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318      # or your Langfuse OTLP endpoint
    OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20<key>  # Langfuse auth, if used
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

tracer = trace.get_tracer("aml")
F = TypeVar("F", bound=Callable[..., Any])
_configured: list[TracerProvider] = []


def init_tracing(service: str, exporter: SpanExporter | None = None) -> TracerProvider | None:
    """Configure once per process. Returns None when tracing is not configured."""
    if _configured:
        return _configured[0]
    if exporter is None:
        if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            return None
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured.append(provider)
    return provider


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[trace.Span]:
    with tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(k, v if isinstance(v, str | int | float | bool) else str(v))
        yield s


def traced(name: str) -> Callable[[F], F]:
    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(name):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return deco
