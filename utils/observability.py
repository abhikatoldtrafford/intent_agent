"""
Observability Configuration Module

Provides centralized configuration for:
- Python logging
- LangSmith tracing
- OpenTelemetry instrumentation

Usage:
    from utils import configure_observability
    configure_observability(enable_all=True)
"""

import os
import sys
import logging
from typing import Optional
from pathlib import Path


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"

        # Format the message
        result = super().format(record)

        # Reset levelname for other handlers
        record.levelname = levelname

        return result


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Configure Python logging with console and optional file output.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        enable_colors: Enable colored console output (default: True)

    Returns:
        Configured root logger
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if enable_colors and sys.stdout.isatty():
        console_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        console_handler.setFormatter(ColoredFormatter(
            console_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
    else:
        console_format = '%(asctime)s - %(levelname)-8s - %(name)-30s - %(message)s'
        console_handler.setFormatter(logging.Formatter(
            console_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        file_format = '%(asctime)s - %(levelname)-8s - %(name)-30s - [%(filename)s:%(lineno)d] - %(message)s'
        file_handler.setFormatter(logging.Formatter(
            file_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    logger.info("Logging configured successfully")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# ============================================================================
# LANGSMITH CONFIGURATION
# ============================================================================

def setup_langsmith(
    project: str = "intent-agent-poc",
    api_key: Optional[str] = None,
    enable: bool = True
) -> bool:
    """
    Configure LangSmith tracing for LangChain/LangGraph.

    Args:
        project: LangSmith project name
        api_key: LangSmith API key (or use LANGCHAIN_API_KEY env var)
        enable: Enable tracing (default: True)

    Returns:
        True if successfully configured, False otherwise
    """
    logger = get_logger(__name__)

    if not enable:
        logger.info("LangSmith tracing disabled")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    # Set tracing environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project

    # Set API key if provided
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key
        logger.info(f"LangSmith API key set (***{api_key[-8:]})")
    elif "LANGCHAIN_API_KEY" in os.environ:
        api_key = os.getenv("LANGCHAIN_API_KEY", "")
        logger.info(f"LangSmith API key found in environment (***{api_key[-8:] if api_key else 'missing'})")
    else:
        logger.warning("LangSmith API key not set - tracing may not work properly")
        logger.warning("Set LANGCHAIN_API_KEY environment variable or pass api_key parameter")

    # Set endpoint (optional)
    if "LANGCHAIN_ENDPOINT" not in os.environ:
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    logger.info(f"LangSmith tracing enabled for project: {project}")
    logger.info("Traces will be available at: https://smith.langchain.com/")

    return True


# ============================================================================
# OPENTELEMETRY CONFIGURATION
# ============================================================================

def setup_opentelemetry(
    service_name: str = "intent-agent",
    exporter: str = "console",
    enable: bool = True
) -> bool:
    """
    Configure OpenTelemetry instrumentation for distributed tracing.

    Args:
        service_name: Service name for traces
        exporter: Exporter type ('console', 'otlp', 'jaeger')
        enable: Enable OpenTelemetry (default: True)

    Returns:
        True if successfully configured, False otherwise
    """
    logger = get_logger(__name__)

    if not enable:
        logger.info("OpenTelemetry instrumentation disabled")
        return False

    try:
        # Import OpenTelemetry components (optional dependency)
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter
        )
        from opentelemetry.sdk.resources import Resource

        # Create resource with service name
        resource = Resource.create({"service.name": service_name})

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Configure exporter
        if exporter == "console":
            exporter_instance = ConsoleSpanExporter()
            logger.info("OpenTelemetry console exporter configured")

        elif exporter == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
            exporter_instance = OTLPSpanExporter(endpoint=endpoint)
            logger.info(f"OpenTelemetry OTLP exporter configured (endpoint: {endpoint})")

        elif exporter == "jaeger":
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter

            agent_host = os.getenv("JAEGER_AGENT_HOST", "localhost")
            agent_port = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
            exporter_instance = JaegerExporter(
                agent_host_name=agent_host,
                agent_port=agent_port
            )
            logger.info(f"OpenTelemetry Jaeger exporter configured ({agent_host}:{agent_port})")

        else:
            logger.error(f"Unknown exporter type: {exporter}")
            return False

        # Add span processor
        provider.add_span_processor(BatchSpanProcessor(exporter_instance))

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Set environment variables
        os.environ["OTEL_SERVICE_NAME"] = service_name
        os.environ["OTEL_EXPORTER"] = exporter

        logger.info(f"OpenTelemetry instrumentation enabled for service: {service_name}")

        return True

    except ImportError as e:
        logger.warning(f"OpenTelemetry not installed: {e}")
        logger.warning("Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
        return False

    except Exception as e:
        logger.error(f"Failed to configure OpenTelemetry: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# UNIFIED CONFIGURATION
# ============================================================================

def configure_observability(
    enable_logging: bool = True,
    enable_langsmith: bool = True,
    enable_opentelemetry: bool = False,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    langsmith_project: str = "intent-agent-poc",
    otel_service_name: str = "intent-agent",
    otel_exporter: str = "console",
    enable_all: bool = False
) -> dict:
    """
    Configure all observability components in one call.

    Args:
        enable_logging: Enable Python logging
        enable_langsmith: Enable LangSmith tracing
        enable_opentelemetry: Enable OpenTelemetry instrumentation
        log_level: Logging level (default: INFO)
        log_file: Optional log file path
        langsmith_project: LangSmith project name
        otel_service_name: OpenTelemetry service name
        otel_exporter: OpenTelemetry exporter type
        enable_all: Enable all observability features

    Returns:
        Dictionary with configuration status
    """
    logger = get_logger(__name__)

    if enable_all:
        enable_logging = True
        enable_langsmith = True
        enable_opentelemetry = True

    status = {
        "logging": False,
        "langsmith": False,
        "opentelemetry": False
    }

    # Configure logging
    if enable_logging:
        try:
            setup_logging(level=log_level, log_file=log_file)
            status["logging"] = True
            logger.info("✓ Logging configured")
        except Exception as e:
            logger.error(f"✗ Logging configuration failed: {e}")

    # Configure LangSmith
    if enable_langsmith:
        try:
            status["langsmith"] = setup_langsmith(project=langsmith_project)
            if status["langsmith"]:
                logger.info("✓ LangSmith configured")
            else:
                logger.warning("✗ LangSmith configuration failed")
        except Exception as e:
            logger.error(f"✗ LangSmith configuration failed: {e}")

    # Configure OpenTelemetry
    if enable_opentelemetry:
        try:
            status["opentelemetry"] = setup_opentelemetry(
                service_name=otel_service_name,
                exporter=otel_exporter
            )
            if status["opentelemetry"]:
                logger.info("✓ OpenTelemetry configured")
            else:
                logger.warning("✗ OpenTelemetry configuration failed")
        except Exception as e:
            logger.error(f"✗ OpenTelemetry configuration failed: {e}")

    # Summary
    enabled_count = sum(status.values())
    total_count = len(status)

    logger.info(f"\n{'='*60}")
    logger.info(f"Observability Configuration Summary:")
    logger.info(f"  • Logging:        {'✓ Enabled' if status['logging'] else '✗ Disabled'}")
    logger.info(f"  • LangSmith:      {'✓ Enabled' if status['langsmith'] else '✗ Disabled'}")
    logger.info(f"  • OpenTelemetry:  {'✓ Enabled' if status['opentelemetry'] else '✗ Disabled'}")
    logger.info(f"  • Total:          {enabled_count}/{total_count} features enabled")
    logger.info(f"{'='*60}\n")

    return status


# ============================================================================
# AUTO-CONFIGURATION
# ============================================================================

def auto_configure():
    """
    Automatically configure observability based on environment variables.

    This function is called when the module is imported.
    """
    # Check if we should enable observability
    auto_enable = os.getenv("ENABLE_OBSERVABILITY", "true").lower() == "true"

    if not auto_enable:
        return

    # Determine what to enable based on environment
    enable_langsmith = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    enable_otel = os.getenv("OTEL_EXPORTER") is not None

    # Only auto-configure if at least one is explicitly enabled
    if enable_langsmith or enable_otel:
        configure_observability(
            enable_logging=True,
            enable_langsmith=enable_langsmith,
            enable_opentelemetry=enable_otel,
            log_level=logging.INFO
        )
