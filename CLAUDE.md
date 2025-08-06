# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

- Setup: `chmod +x setup.sh && ./setup.sh`
- Activate environment: `source qtm_env/bin/activate`
- Run QTM streaming: `python qtm_stream.py [--qtm-host HOST] [--qtm-port PORT] [--insole-ip IP] [--insole-port PORT]`
- Run Moticon receiver: `python moticon_receiver.py [--ip IP] [--port PORT]`
- Process data: `python process_data.py`
- Process force data: `python process_data_force.py`
- Lint code: `python -m ruff check [filename].py`

## Code Style Guidelines

- **Formatting**: Line length 88 characters (Ruff)
- **Python version**: 3.8+
- **Imports**: Group standard library, third-party, and local imports (handled by isort)
- **Naming**: Use snake_case for variables/functions, PascalCase for classes
- **Documentation**: Use docstrings for functions, classes, and modules
- **Error handling**: Use try/except blocks with specific exceptions
- **Comments**: Add comments for complex code sections
- **Whitespace**: Use 4 spaces for indentation

## Project Structure
- `data/`: Contains QTM, insole, and OpenCap data files
- `plots/`: Output directory for generated plots
- `comparison_plots/`: Output directory for comparison plots
- `src/`: Source code directory for package modules