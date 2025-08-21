#!/bin/bash

# Voice Conversion Docker Management Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} $1 ${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if required directories exist
check_directories() {
    print_status "Checking directory structure..."
    
    if [ ! -d "data" ]; then
        print_warning "Creating data directory structure..."
        mkdir -p data/source_voice data/target_voice
    fi
    
    if [ ! -d "processed" ]; then
        mkdir -p processed/source processed/target
    fi
    
    if [ ! -d "checkpoints" ]; then
        mkdir -p checkpoints
    fi
    
    if [ ! -d "models" ]; then
        mkdir -p models
    fi
    
    if [ ! -d "output" ]; then
        mkdir -p output
    fi
    
    print_status "Directory structure ready!"
}

# Build the Docker image
build_image() {
    print_header "Building Docker Image"
    docker-compose build voice-vc
    print_status "Docker image built successfully!"
}

# Run preprocessing
run_preprocess() {
    print_header "Running Audio Preprocessing"
    check_directories
    
    # Check if source data exists
    if [ -z "$(ls -A data/source_voice 2>/dev/null)" ]; then
        print_error "No files found in data/source_voice/"
        print_status "Please add your source audio files to data/source_voice/"
        exit 1
    fi
    
    if [ -z "$(ls -A data/target_voice 2>/dev/null)" ]; then
        print_error "No files found in data/target_voice/"
        print_status "Please add your target audio files to data/target_voice/"
        exit 1
    fi
    
    docker-compose run --rm voice-vc
    print_status "Preprocessing completed!"
}

# Run training
run_training() {
    print_header "Starting Training"
    
    # Check if preprocessed data exists
    if [ -z "$(ls -A processed/source 2>/dev/null)" ] || [ -z "$(ls -A processed/target 2>/dev/null)" ]; then
        print_error "No preprocessed data found!"
        print_status "Please run preprocessing first: ./run.sh preprocess"
        exit 1
    fi
    
    docker-compose --profile train run --rm voice-vc-train
    print_status "Training completed!"
}

# Run inference
run_inference() {
    print_header "Running Voice Conversion"
    
    # Check if model exists
    if [ ! -f "starGAN_G.pth" ] && [ -z "$(ls -A checkpoints 2>/dev/null)" ]; then
        print_error "No trained model found!"
        print_status "Please run training first: ./run.sh train"
        exit 1
    fi
    
    docker-compose --profile inference run --rm voice-vc-inference
    print_status "Voice conversion completed! Check the output directory."
}

# Interactive mode
interactive_mode() {
    print_header "Interactive Docker Shell"
    docker-compose run --rm -it voice-vc bash
}

# Clean up containers and volumes
cleanup() {
    print_header "Cleaning Up"
    docker-compose down -v
    docker system prune -f
    print_status "Cleanup completed!"
}

# Show logs
show_logs() {
    docker-compose logs -f voice-vc
}

# Usage function
usage() {
    echo "Usage: $0 {build|preprocess|train|inference|interactive|cleanup|logs|help}"
    echo ""
    echo "Commands:"
    echo "  build        Build the Docker image"
    echo "  preprocess   Run audio preprocessing"
    echo "  train        Start model training"
    echo "  inference    Run voice conversion"
    echo "  interactive  Open interactive shell in container"
    echo "  cleanup      Clean up containers and volumes"
    echo "  logs         Show container logs"
    echo "  help         Show this help message"
    echo ""
    echo "Workflow:"
    echo "  1. ./run.sh build"
    echo "  2. Add audio files to data/source_voice and data/target_voice"
    echo "  3. ./run.sh preprocess"
    echo "  4. ./run.sh train"
    echo "  5. ./run.sh inference"
}

# Main script logic
case "$1" in
    build)
        build_image
        ;;
    preprocess)
        run_preprocess
        ;;
    train)
        run_training
        ;;
    inference)
        run_inference
        ;;
    interactive)
        interactive_mode
        ;;
    cleanup)
        cleanup
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        print_error "Invalid command: $1"
        usage
        exit 1
        ;;
esac