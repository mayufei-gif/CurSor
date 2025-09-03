#!/usr/bin/env python3
"""
PDF-MCP Server Command Line Interface

Provides command-line tools for PDF processing operations.

Usage:
    pdf-mcp-cli process --file document.pdf --mode full_pipeline
    pdf-mcp-cli server --host 0.0.0.0 --port 8000
    pdf-mcp-cli config --create
    pdf-mcp-cli health --url http://localhost:8000

Author: PDF-MCP Team
License: MIT
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.json import JSON
from rich.panel import Panel
from rich.text import Text

# Add src to Python path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from pdf_mcp_server.core import PDFProcessor, ProcessingRequest
from pdf_mcp_server.utils import Config, setup_logging, get_logger
from pdf_mcp_server.main import main as server_main

# Initialize console for rich output
console = Console()
logger = get_logger(__name__)


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool, quiet: bool):
    """PDF-MCP Server Command Line Interface."""
    ctx.ensure_object(dict)
    
    # Set up logging
    if verbose:
        log_level = 'DEBUG'
    elif quiet:
        log_level = 'ERROR'
    else:
        log_level = 'INFO'
    
    setup_logging(level=log_level)
    
    # Load configuration
    try:
        if config:
            ctx.obj['config'] = Config.load_config(config)
        else:
            ctx.obj['config'] = Config.load_config()
    except Exception as e:
        if not quiet:
            console.print(f"[red]Error loading configuration: {e}[/red]")
        ctx.obj['config'] = None
    
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


@cli.command()
@click.option('--file', '-f', required=True, help='PDF file path')
@click.option('--mode', '-m', 
              type=click.Choice(['read_text', 'extract_tables', 'extract_formulas', 'process_ocr', 'full_pipeline']),
              default='full_pipeline', help='Processing mode')
@click.option('--pages', '-p', help='Page range (e.g., "1-5" or "1,3,5")')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', 'output_format', 
              type=click.Choice(['json', 'csv', 'xlsx', 'xml', 'html']),
              default='json', help='Output format')
@click.option('--include-ocr/--no-ocr', default=True, help='Include OCR processing')
@click.option('--include-formulas/--no-formulas', default=False, help='Include formula extraction')
@click.option('--include-grobid/--no-grobid', default=False, help='Include GROBID processing')
@click.option('--table-engine', 
              type=click.Choice(['camelot', 'tabula', 'pdfplumber', 'auto']),
              default='auto', help='Table extraction engine')
@click.option('--formula-model',
              type=click.Choice(['pix2tex', 'latex-ocr', 'texify', 'auto']),
              default='auto', help='Formula recognition model')
@click.pass_context
def process(ctx, file: str, mode: str, pages: Optional[str], output: Optional[str],
           output_format: str, include_ocr: bool, include_formulas: bool, 
           include_grobid: bool, table_engine: str, formula_model: str):
    """Process a PDF file with specified options."""
    config = ctx.obj.get('config')
    verbose = ctx.obj.get('verbose', False)
    quiet = ctx.obj.get('quiet', False)
    
    if not config:
        console.print("[red]Configuration not loaded. Cannot process file.[/red]")
        sys.exit(1)
    
    # Validate file exists
    file_path = Path(file)
    if not file_path.exists():
        console.print(f"[red]File not found: {file}[/red]")
        sys.exit(1)
    
    if not file_path.suffix.lower() == '.pdf':
        console.print(f"[red]File must be a PDF: {file}[/red]")
        sys.exit(1)
    
    async def process_file():
        try:
            # Initialize processor
            if not quiet:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Initializing processor...", total=None)
                    
                    processor = PDFProcessor(config)
                    await processor.initialize()
                    
                    progress.update(task, description="Processing PDF...")
                    
                    # Parse pages
                    parsed_pages = None
                    if pages:
                        if '-' in pages:
                            start, end = map(int, pages.split('-'))
                            parsed_pages = list(range(start, end + 1))
                        elif ',' in pages:
                            parsed_pages = [int(p.strip()) for p in pages.split(',')]
                        else:
                            parsed_pages = [int(pages)]
                    
                    # Create processing request
                    options = {
                        'include_ocr': include_ocr,
                        'include_formulas': include_formulas,
                        'include_grobid': include_grobid,
                        'table_engine': table_engine,
                        'formula_model': formula_model
                    }
                    
                    request = ProcessingRequest(
                        file_path=str(file_path),
                        mode=mode,
                        pages=parsed_pages,
                        output_format=output_format,
                        options=options
                    )
                    
                    # Process file
                    result = await processor.process(request)
                    
                    progress.update(task, description="Processing complete!")
            else:
                # Silent processing
                processor = PDFProcessor(config)
                await processor.initialize()
                
                # Parse pages
                parsed_pages = None
                if pages:
                    if '-' in pages:
                        start, end = map(int, pages.split('-'))
                        parsed_pages = list(range(start, end + 1))
                    elif ',' in pages:
                        parsed_pages = [int(p.strip()) for p in pages.split(',')]
                    else:
                        parsed_pages = [int(pages)]
                
                # Create processing request
                options = {
                    'include_ocr': include_ocr,
                    'include_formulas': include_formulas,
                    'include_grobid': include_grobid,
                    'table_engine': table_engine,
                    'formula_model': formula_model
                }
                
                request = ProcessingRequest(
                    file_path=str(file_path),
                    mode=mode,
                    pages=parsed_pages,
                    output_format=output_format,
                    options=options
                )
                
                result = await processor.process(request)
            
            # Output results
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if output_format == 'json':
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result.dict(), f, indent=2, ensure_ascii=False)
                else:
                    # Handle other formats based on result content
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(str(result))
                
                if not quiet:
                    console.print(f"[green]Results saved to: {output_path}[/green]")
            else:
                # Print to console
                if output_format == 'json':
                    if not quiet:
                        console.print(Panel(JSON(json.dumps(result.dict(), indent=2)), title="Processing Results"))
                    else:
                        print(json.dumps(result.dict(), indent=2))
                else:
                    if not quiet:
                        console.print(Panel(str(result), title="Processing Results"))
                    else:
                        print(str(result))
            
            # Cleanup
            await processor.cleanup()
            
        except Exception as e:
            if not quiet:
                console.print(f"[red]Processing failed: {e}[/red]")
                if verbose:
                    console.print_exception()
            else:
                print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Run async processing
    asyncio.run(process_file())


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', type=int, default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--workers', type=int, default=1, help='Number of worker processes')
@click.pass_context
def server(ctx, host: str, port: int, reload: bool, workers: int):
    """Start the PDF-MCP server."""
    quiet = ctx.obj.get('quiet', False)
    
    if not quiet:
        console.print(f"[green]Starting PDF-MCP Server on {host}:{port}[/green]")
        console.print(f"[blue]Workers: {workers}, Reload: {reload}[/blue]")
    
    # Set up arguments for server
    sys.argv = [
        'pdf-mcp-server',
        '--host', host,
        '--port', str(port),
        '--workers', str(workers)
    ]
    
    if reload:
        sys.argv.append('--reload')
    
    # Start server
    server_main()


@cli.command()
@click.option('--create', is_flag=True, help='Create default configuration file')
@click.option('--validate', is_flag=True, help='Validate configuration file')
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--path', help='Configuration file path')
@click.pass_context
def config(ctx, create: bool, validate: bool, show: bool, path: Optional[str]):
    """Manage configuration files."""
    quiet = ctx.obj.get('quiet', False)
    
    if create:
        config_path = Path(path) if path else Config.get_default_config_path()
        
        try:
            Config.create_default_config(config_path)
            if not quiet:
                console.print(f"[green]Default configuration created: {config_path}[/green]")
        except Exception as e:
            if not quiet:
                console.print(f"[red]Failed to create configuration: {e}[/red]")
            sys.exit(1)
    
    elif validate:
        try:
            config_obj = Config.load_config(path)
            config_obj.validate()
            if not quiet:
                console.print("[green]Configuration is valid[/green]")
        except Exception as e:
            if not quiet:
                console.print(f"[red]Configuration validation failed: {e}[/red]")
            sys.exit(1)
    
    elif show:
        try:
            config_obj = Config.load_config(path)
            config_dict = config_obj.to_dict()
            
            if not quiet:
                console.print(Panel(JSON(json.dumps(config_dict, indent=2)), title="Current Configuration"))
            else:
                print(json.dumps(config_dict, indent=2))
        except Exception as e:
            if not quiet:
                console.print(f"[red]Failed to load configuration: {e}[/red]")
            sys.exit(1)
    
    else:
        console.print("[yellow]Please specify an action: --create, --validate, or --show[/yellow]")


@cli.command()
@click.option('--url', default='http://localhost:8000', help='Server URL')
@click.option('--timeout', type=int, default=10, help='Request timeout in seconds')
@click.pass_context
def health(ctx, url: str, timeout: int):
    """Check server health status."""
    quiet = ctx.obj.get('quiet', False)
    
    async def check_health():
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{url}/health")
                response.raise_for_status()
                
                health_data = response.json()
                
                if not quiet:
                    # Create health status table
                    table = Table(title="Server Health Status")
                    table.add_column("Component", style="cyan")
                    table.add_column("Status", style="green")
                    table.add_column("Details")
                    
                    # Overall status
                    overall_status = "Healthy" if health_data.get('status') == 'healthy' else "Unhealthy"
                    table.add_row("Overall", overall_status, f"Uptime: {health_data.get('uptime', 0):.2f}s")
                    
                    # Dependencies
                    dependencies = health_data.get('dependencies', {})
                    for dep, status in dependencies.items():
                        status_text = "✓ OK" if status else "✗ Failed"
                        table.add_row(dep, status_text, "")
                    
                    console.print(table)
                    
                    # System info
                    system_info = health_data.get('system_info', {})
                    if system_info:
                        console.print("\n[bold]System Information:[/bold]")
                        for key, value in system_info.items():
                            console.print(f"  {key}: {value}")
                else:
                    print(json.dumps(health_data, indent=2))
                
        except httpx.TimeoutException:
            if not quiet:
                console.print(f"[red]Health check timed out after {timeout} seconds[/red]")
            else:
                print("Error: Timeout", file=sys.stderr)
            sys.exit(1)
        except httpx.HTTPStatusError as e:
            if not quiet:
                console.print(f"[red]Health check failed with status {e.response.status_code}[/red]")
            else:
                print(f"Error: HTTP {e.response.status_code}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            if not quiet:
                console.print(f"[red]Health check failed: {e}[/red]")
            else:
                print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    asyncio.run(check_health())


@cli.command()
@click.option('--files', '-f', multiple=True, required=True, help='PDF files to process (can be specified multiple times)')
@click.option('--mode', '-m',
              type=click.Choice(['read_text', 'extract_tables', 'extract_formulas', 'process_ocr', 'full_pipeline']),
              default='full_pipeline', help='Processing mode')
@click.option('--output-dir', '-o', help='Output directory for results')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'csv', 'xlsx', 'xml', 'html']),
              default='json', help='Output format')
@click.option('--parallel/--sequential', default=True, help='Process files in parallel')
@click.option('--max-workers', type=int, default=4, help='Maximum number of parallel workers')
@click.pass_context
def batch(ctx, files: List[str], mode: str, output_dir: Optional[str],
         output_format: str, parallel: bool, max_workers: int):
    """Process multiple PDF files in batch."""
    config = ctx.obj.get('config')
    quiet = ctx.obj.get('quiet', False)
    
    if not config:
        console.print("[red]Configuration not loaded. Cannot process files.[/red]")
        sys.exit(1)
    
    # Validate files
    valid_files = []
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            if not quiet:
                console.print(f"[yellow]Warning: File not found: {file_path}[/yellow]")
            continue
        if not path.suffix.lower() == '.pdf':
            if not quiet:
                console.print(f"[yellow]Warning: Not a PDF file: {file_path}[/yellow]")
            continue
        valid_files.append(str(path))
    
    if not valid_files:
        console.print("[red]No valid PDF files found[/red]")
        sys.exit(1)
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    async def process_batch():
        try:
            processor = PDFProcessor(config)
            await processor.initialize()
            
            if not quiet:
                console.print(f"[green]Processing {len(valid_files)} files...[/green]")
            
            results = []
            
            if parallel:
                # Process files in parallel
                semaphore = asyncio.Semaphore(max_workers)
                
                async def process_single_file(file_path: str):
                    async with semaphore:
                        try:
                            request = ProcessingRequest(
                                file_path=file_path,
                                mode=mode,
                                output_format=output_format
                            )
                            result = await processor.process(request)
                            return {'file': file_path, 'result': result, 'success': True}
                        except Exception as e:
                            return {'file': file_path, 'error': str(e), 'success': False}
                
                tasks = [process_single_file(f) for f in valid_files]
                results = await asyncio.gather(*tasks)
            else:
                # Process files sequentially
                for file_path in valid_files:
                    try:
                        if not quiet:
                            console.print(f"Processing: {Path(file_path).name}")
                        
                        request = ProcessingRequest(
                            file_path=file_path,
                            mode=mode,
                            output_format=output_format
                        )
                        result = await processor.process(request)
                        results.append({'file': file_path, 'result': result, 'success': True})
                    except Exception as e:
                        results.append({'file': file_path, 'error': str(e), 'success': False})
            
            # Save results
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            
            if output_dir:
                for i, result in enumerate(results):
                    file_name = Path(result['file']).stem
                    if result['success']:
                        output_file = output_path / f"{file_name}_result.{output_format}"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            if output_format == 'json':
                                json.dump(result['result'].dict(), f, indent=2, ensure_ascii=False)
                            else:
                                f.write(str(result['result']))
                    else:
                        error_file = output_path / f"{file_name}_error.txt"
                        with open(error_file, 'w', encoding='utf-8') as f:
                            f.write(result['error'])
            
            if not quiet:
                console.print(f"\n[green]Batch processing complete![/green]")
                console.print(f"Successful: {successful}/{len(results)}")
                console.print(f"Failed: {failed}/{len(results)}")
                
                if output_dir:
                    console.print(f"Results saved to: {output_path}")
            else:
                # Output summary in JSON format
                summary = {
                    'total': len(results),
                    'successful': successful,
                    'failed': failed,
                    'results': results
                }
                print(json.dumps(summary, indent=2))
            
            await processor.cleanup()
            
        except Exception as e:
            if not quiet:
                console.print(f"[red]Batch processing failed: {e}[/red]")
            else:
                print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    asyncio.run(process_batch())


@cli.command()
@click.option('--url', default='http://localhost:8000', help='Server URL')
@click.option('--timeout', type=int, default=30, help='Request timeout in seconds')
@click.pass_context
def models(ctx, url: str, timeout: int):
    """Get information about loaded models."""
    quiet = ctx.obj.get('quiet', False)
    
    async def get_models_info():
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{url}/models")
                response.raise_for_status()
                
                models_data = response.json()
                
                if not quiet:
                    console.print(Panel(JSON(json.dumps(models_data, indent=2)), title="Loaded Models"))
                else:
                    print(json.dumps(models_data, indent=2))
                
        except Exception as e:
            if not quiet:
                console.print(f"[red]Failed to get models info: {e}[/red]")
            else:
                print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    asyncio.run(get_models_info())


def main():
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()