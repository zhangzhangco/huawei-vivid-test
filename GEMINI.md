# GEMINI.md

## Project Overview

This project is a visualization tool for an HDR (High Dynamic Range) tone mapping patent based on the Phoenix curve algorithm. It provides a Gradio-based web interface for real-time parameter adjustment, quality assessment, and image processing.

The core of the project is the Phoenix curve algorithm, a patented tone mapping technique. The application allows users to upload HDR images, apply the tone mapping with adjustable parameters, and visualize the results in real-time. It also provides quality metrics like perceptual distortion and local contrast.

The project is built with Python and uses a number of libraries for scientific computing, image processing, and visualization, including:

*   **Gradio:** For the web UI
*   **NumPy:** For numerical operations
*   **OpenCV and Pillow:** For image processing
*   **Matplotlib and Plotly:** For plotting and visualization
*   **Numba:** For GPU acceleration

The architecture is divided into a Gradio frontend, a core logic layer containing the main algorithms, a data processing layer, and an infrastructure layer for state management, error handling, and performance monitoring.

## Building and Running

1.  **Clone the repository:**
    ```bash
    git clone https://huggingface.co/spaces/zhangzhangco/HuaweiHDR
    cd HuaweiHDR
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python launch_gradio.py
    ```
    The application will be available at `http://localhost:7860`.

### GPU Acceleration

The application supports NVIDIA GPU acceleration for improved performance. To enable it, install the required libraries:

```bash
pip install cupy-cuda12x numba
```

## Development Conventions

The codebase is structured into a `src` directory containing the main application logic and a `tests` directory for tests.

*   `src/gradio_app.py`: Contains the Gradio UI code.
*   `src/core/`: Contains the core business logic, including the Phoenix curve algorithm, PQ converter, quality metrics calculator, and other components.
*   `launch_gradio.py`: The main entry point for running the Gradio application.
*   `app.py`: An entry point for deploying the application on platforms like Hugging Face Spaces.
*   `tests/`: Contains tests for the project. The testing framework seems to be pytest.

The project uses a `setup.py` file, which indicates that it can be installed as a package.

## Key Files

*   `README.md`: Provides a high-level overview of the project, its features, and how to run it.
*   `HDR色调映射专利可视化技术说明书.md`: A detailed technical document explaining the project's architecture, algorithms, and implementation.
*   `requirements.txt`: Lists the Python dependencies.
*   `launch_gradio.py`: The main entry point for running the application.
*   `src/gradio_app.py`: The main file for the Gradio UI.
*   `src/core/`: The directory containing the core logic of the application.
*   `setup.py`: The setup script for the project.
*   `app.py`: The entry point for Hugging Face Spaces deployment.
*   `tests/`: The directory containing the tests.
*   `config/`: The directory containing configuration files.
