"""HTTP server for Prometheus metrics scraping.

This module provides an aiohttp-based HTTP server that exposes
Prometheus metrics at the /metrics endpoint for scraping.

Default port: 9090
Endpoints:
  - /metrics: Prometheus metrics in text format
  - /health: Health check endpoint
"""

import asyncio
import logging
from typing import Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

from nethical.monitoring.prometheus_exporter import PrometheusMetrics

logger = logging.getLogger(__name__)


class MetricsServer:
    """HTTP server for Prometheus metrics scraping.
    
    Provides a simple HTTP server that exposes Prometheus metrics
    for scraping. Includes health check endpoint for monitoring.
    
    Default port: 9090
    Endpoint: /metrics
    """

    def __init__(self, metrics: PrometheusMetrics, port: int = 9090, host: str = '0.0.0.0'):
        """Initialize metrics server.
        
        Args:
            metrics: PrometheusMetrics instance to export
            port: Port to listen on (default: 9090)
            host: Host to bind to (default: 0.0.0.0)
        """
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is required for MetricsServer. "
                "Install with: pip install aiohttp>=3.9.0"
            )
        
        self.metrics = metrics
        self.port = port
        self.host = host
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Setup HTTP routes for the metrics server."""
        self.app = web.Application()
        self.app.router.add_get('/metrics', self.handle_metrics)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/', self.handle_root)
    
    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Serve Prometheus metrics.
        
        Args:
            request: aiohttp request object
            
        Returns:
            Response with Prometheus metrics in text format
        """
        try:
            metrics_output = self.metrics.export_metrics()
            return web.Response(
                body=metrics_output,
                content_type=self.metrics.get_content_type()
            )
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return web.Response(
                text=f"Error exporting metrics: {e}",
                status=500
            )
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint.
        
        Args:
            request: aiohttp request object
            
        Returns:
            JSON response with health status
        """
        health_data = {
            'status': 'healthy',
            'metrics_enabled': self.metrics.enabled,
            'server': 'nethical-metrics'
        }
        return web.json_response(health_data)
    
    async def handle_root(self, request: web.Request) -> web.Response:
        """Root endpoint with server information.
        
        Args:
            request: aiohttp request object
            
        Returns:
            HTML response with links to available endpoints
        """
        html = """
        <html>
            <head><title>Nethical Metrics Server</title></head>
            <body>
                <h1>Nethical Metrics Server</h1>
                <p>Prometheus metrics server for Nethical threat detection system.</p>
                <h2>Available Endpoints:</h2>
                <ul>
                    <li><a href="/metrics">/metrics</a> - Prometheus metrics</li>
                    <li><a href="/health">/health</a> - Health check</li>
                </ul>
            </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def start_async(self) -> None:
        """Start the metrics server asynchronously."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        logger.info(f"Metrics server started on {self.host}:{self.port}")
        logger.info(f"Metrics available at http://{self.host}:{self.port}/metrics")
    
    async def stop_async(self) -> None:
        """Stop the metrics server asynchronously."""
        if self.runner:
            await self.runner.cleanup()
            logger.info("Metrics server stopped")
    
    def start(self) -> None:
        """Start metrics server (blocking).
        
        This method starts the server and blocks until interrupted.
        Use start_async() for non-blocking operation.
        """
        try:
            web.run_app(self.app, host=self.host, port=self.port, print=lambda x: None)
        except KeyboardInterrupt:
            logger.info("Metrics server stopped by user")
    
    async def run_forever(self) -> None:
        """Run the metrics server forever.
        
        This is a non-blocking version that can be used with asyncio.
        """
        await self.start_async()
        try:
            # Keep running until cancelled
            while True:
                await asyncio.sleep(3600)  # Sleep for an hour
        except asyncio.CancelledError:
            logger.info("Metrics server cancelled")
        finally:
            await self.stop_async()


def start_metrics_server(
    metrics: Optional[PrometheusMetrics] = None,
    port: int = 9090,
    host: str = '0.0.0.0'
) -> MetricsServer:
    """Start a metrics server (convenience function).
    
    Args:
        metrics: Optional PrometheusMetrics instance. If None, creates a new one.
        port: Port to listen on (default: 9090)
        host: Host to bind to (default: 0.0.0.0)
        
    Returns:
        MetricsServer instance
        
    Example:
        >>> from nethical.monitoring import start_metrics_server
        >>> server = start_metrics_server(port=9090)
        >>> # Server is now running on http://0.0.0.0:9090/metrics
    """
    if metrics is None:
        from nethical.monitoring.prometheus_exporter import get_prometheus_metrics
        metrics = get_prometheus_metrics()
    
    server = MetricsServer(metrics, port=port, host=host)
    
    # Start in a background thread/task
    import threading
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    
    logger.info(f"Metrics server starting in background on {host}:{port}")
    return server


async def start_metrics_server_async(
    metrics: Optional[PrometheusMetrics] = None,
    port: int = 9090,
    host: str = '0.0.0.0'
) -> MetricsServer:
    """Start a metrics server asynchronously.
    
    Args:
        metrics: Optional PrometheusMetrics instance. If None, creates a new one.
        port: Port to listen on (default: 9090)
        host: Host to bind to (default: 0.0.0.0)
        
    Returns:
        MetricsServer instance (already started)
        
    Example:
        >>> from nethical.monitoring import start_metrics_server_async
        >>> server = await start_metrics_server_async(port=9090)
        >>> # Server is now running
    """
    if metrics is None:
        from nethical.monitoring.prometheus_exporter import get_prometheus_metrics
        metrics = get_prometheus_metrics()
    
    server = MetricsServer(metrics, port=port, host=host)
    await server.start_async()
    
    return server
