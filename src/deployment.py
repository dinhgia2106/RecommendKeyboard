"""
CRF Model Deployment for Vietnamese Word Segmentation

This module provides deployment options for the trained CRF model including:
- Gradio web interface for interactive demos
- FastAPI REST API for production use
- Batch processing capabilities
- Model monitoring and logging

Features:
- User-friendly web interface
- RESTful API endpoints
- Real-time performance monitoring
- Error handling and logging
- Scalable deployment configuration
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .inference import CRFInference, SegmentationResult, PerformanceMonitor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationRequest(BaseModel):
    """Request model for segmentation API."""
    text: str
    preprocess: bool = True


class SegmentationResponse(BaseModel):
    """Response model for segmentation API."""
    input_text: str
    segmented_text: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class BatchSegmentationRequest(BaseModel):
    """Request model for batch segmentation."""
    texts: List[str]
    preprocess: bool = True


class BatchSegmentationResponse(BaseModel):
    """Response model for batch segmentation."""
    results: List[SegmentationResponse]
    total_processing_time: float
    success_count: int
    error_count: int


class CRFDeploymentManager:
    """
    Deployment manager for CRF-based Vietnamese word segmentation.
    
    This class handles model loading, API creation, and deployment
    configuration for both development and production environments.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize deployment manager.
        
        Args:
            model_path: Path to trained CRF model
        """
        self.model_path = model_path
        self.inference_engine = None
        self.performance_monitor = PerformanceMonitor()
        self.load_model()
    
    def load_model(self):
        """Load the CRF model for deployment."""
        try:
            logger.info(f"Loading CRF model from {self.model_path}")
            self.inference_engine = CRFInference(self.model_path)
            logger.info("✅ CRF model loaded successfully for deployment")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def segment_text(self, text: str, preprocess: bool = True) -> SegmentationResponse:
        """
        Segment single text with error handling.
        
        Args:
            text: Input text to segment
            preprocess: Whether to preprocess the text
            
        Returns:
            SegmentationResponse with results
        """
        try:
            if not self.inference_engine:
                raise RuntimeError("Model not loaded")
            
            result = self.inference_engine.segment(text)
            
            return SegmentationResponse(
                input_text=result.input_text,
                segmented_text=result.segmented_text,
                processing_time=result.processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Segmentation error for text '{text}': {e}")
            return SegmentationResponse(
                input_text=text,
                segmented_text="",
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def segment_batch(self, texts: List[str], preprocess: bool = True) -> BatchSegmentationResponse:
        """
        Segment multiple texts with error handling.
        
        Args:
            texts: List of input texts
            preprocess: Whether to preprocess texts
            
        Returns:
            BatchSegmentationResponse with results
        """
        start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        
        for text in texts:
            response = self.segment_text(text, preprocess)
            results.append(response)
            
            if response.success:
                success_count += 1
            else:
                error_count += 1
        
        total_time = time.time() - start_time
        
        return BatchSegmentationResponse(
            results=results,
            total_processing_time=total_time,
            success_count=success_count,
            error_count=error_count
        )


# Global deployment manager
deployment_manager = None


def initialize_deployment(model_path: str = "models/crf/best_model.pkl"):
    """Initialize global deployment manager."""
    global deployment_manager
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    deployment_manager = CRFDeploymentManager(model_path)


# Gradio Interface
def create_gradio_interface() -> gr.Interface:
    """
    Create Gradio web interface for Vietnamese word segmentation.
    
    Returns:
        Configured Gradio interface
    """
    
    def segment_interface(text: str) -> tuple:
        """Interface function for Gradio."""
        if not deployment_manager:
            return "❌ Model not loaded", 0.0, "Error: Model not initialized"
        
        if not text.strip():
            return "", 0.0, "Please enter some text to segment"
        
        response = deployment_manager.segment_text(text)
        
        if response.success:
            status = f"✅ Segmented successfully in {response.processing_time:.4f}s"
            return response.segmented_text, response.processing_time, status
        else:
            return "", 0.0, f"❌ Error: {response.error_message}"
    
    def batch_segment_interface(texts: str) -> str:
        """Batch segmentation interface for Gradio."""
        if not deployment_manager:
            return "❌ Model not loaded"
        
        if not texts.strip():
            return "Please enter texts to segment (one per line)"
        
        text_list = [line.strip() for line in texts.split('\n') if line.strip()]
        
        if not text_list:
            return "No valid texts found"
        
        batch_response = deployment_manager.segment_batch(text_list)
        
        # Format results
        results = []
        for i, result in enumerate(batch_response.results):
            if result.success:
                results.append(f"{i+1}. '{result.input_text}' → '{result.segmented_text}'")
            else:
                results.append(f"{i+1}. '{result.input_text}' → ERROR: {result.error_message}")
        
        summary = f"\n📊 Summary: {batch_response.success_count} successful, {batch_response.error_count} errors"
        summary += f"\n⏱️  Total time: {batch_response.total_processing_time:.2f}s"
        
        return "\n".join(results) + "\n" + summary
    
    def show_examples():
        """Return example texts for demonstration."""
        examples = [
            "xinchao",
            "toilasinhhvien",
            "moibandenquannuocvietnam",
            "chungtoicunglamviec",
            "homnaylaicuoituan"
        ]
        return "\n".join(examples)
    
    # Create interface with tabs
    with gr.Blocks(title="Vietnamese Word Segmentation - CRF", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🇻🇳 Vietnamese Word Segmentation using CRF")
        gr.Markdown("Enter Vietnamese text without spaces and diacritics to get properly segmented text.")
        
        with gr.Tab("Single Text Segmentation"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="Input Text (no spaces, no diacritics)",
                        placeholder="Example: xinchao",
                        lines=2
                    )
                    segment_btn = gr.Button("Segment Text", variant="primary")
                
                with gr.Column(scale=2):
                    output_text = gr.Textbox(
                        label="Segmented Text",
                        lines=2,
                        interactive=False
                    )
                    processing_time = gr.Number(
                        label="Processing Time (seconds)",
                        interactive=False
                    )
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
            
            # Examples
            gr.Markdown("### Example Inputs:")
            example_texts = [
                ["xinchao"],
                ["toilasinhhvien"],
                ["moibandenquannuocvietnam"],
                ["chungtoicunglamviec"]
            ]
            
            gr.Examples(
                examples=example_texts,
                inputs=input_text,
                outputs=[output_text, processing_time, status_text],
                fn=segment_interface,
                cache_examples=False
            )
        
        with gr.Tab("Batch Segmentation"):
            with gr.Row():
                with gr.Column():
                    batch_input = gr.Textbox(
                        label="Input Texts (one per line)",
                        placeholder="Enter multiple texts, one per line",
                        lines=8
                    )
                    batch_segment_btn = gr.Button("Segment All", variant="primary")
                    show_examples_btn = gr.Button("Show Examples", variant="secondary")
                
                with gr.Column():
                    batch_output = gr.Textbox(
                        label="Segmentation Results",
                        lines=10,
                        interactive=False
                    )
        
        with gr.Tab("Model Information"):
            if deployment_manager and deployment_manager.inference_engine:
                model_info = deployment_manager.inference_engine.get_model_info()
                
                gr.Markdown("### Model Details")
                gr.JSON(model_info, label="Model Information")
            else:
                gr.Markdown("❌ Model not loaded")
        
        # Event handlers
        segment_btn.click(
            fn=segment_interface,
            inputs=input_text,
            outputs=[output_text, processing_time, status_text]
        )
        
        batch_segment_btn.click(
            fn=batch_segment_interface,
            inputs=batch_input,
            outputs=batch_output
        )
        
        show_examples_btn.click(
            fn=show_examples,
            outputs=batch_input
        )
    
    return demo


# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    # Startup
    logger.info("🚀 Starting Vietnamese Word Segmentation API")
    yield
    # Shutdown
    logger.info("👋 Shutting down API")


app = FastAPI(
    title="Vietnamese Word Segmentation API",
    description="CRF-based Vietnamese word segmentation service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Vietnamese Word Segmentation API",
        "model": "CRF-based",
        "endpoints": ["/segment", "/batch_segment", "/health", "/model_info"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if deployment_manager and deployment_manager.inference_engine:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}


@app.get("/model_info")
async def get_model_info():
    """Get model information."""
    if not deployment_manager:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return deployment_manager.inference_engine.get_model_info()


@app.post("/segment", response_model=SegmentationResponse)
async def segment_text(request: SegmentationRequest):
    """Segment single text."""
    if not deployment_manager:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return deployment_manager.segment_text(request.text, request.preprocess)


@app.post("/batch_segment", response_model=BatchSegmentationResponse)
async def batch_segment_texts(request: BatchSegmentationRequest):
    """Segment multiple texts."""
    if not deployment_manager:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.texts) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    return deployment_manager.segment_batch(request.texts, request.preprocess)


def main():
    """Main deployment script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Vietnamese Word Segmentation CRF Model")
    parser.add_argument("--mode", choices=["gradio", "api", "both"], default="gradio",
                       help="Deployment mode")
    parser.add_argument("--model-path", default="models/crf/best_model.pkl",
                       help="Path to trained model")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--api-port", type=int, default=8000, help="API port number")
    
    args = parser.parse_args()
    
    # Initialize deployment
    try:
        initialize_deployment(args.model_path)
        logger.info("✅ Deployment manager initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize deployment: {e}")
        sys.exit(1)
    
    # Deploy based on mode
    if args.mode == "gradio":
        logger.info("🌐 Launching Gradio interface...")
        demo = create_gradio_interface()
        demo.launch(server_name=args.host, server_port=args.port, share=False)
        
    elif args.mode == "api":
        logger.info("🚀 Launching FastAPI server...")
        uvicorn.run(app, host=args.host, port=args.api_port)
        
    elif args.mode == "both":
        logger.info("🚀 Launching both Gradio and FastAPI...")
        # In production, you'd want to run these in separate processes
        # For demo, we'll just show how to configure both
        print(f"Gradio would run on: http://{args.host}:{args.port}")
        print(f"FastAPI would run on: http://{args.host}:{args.api_port}")
        print("Run them separately for production deployment")


if __name__ == "__main__":
    main() 