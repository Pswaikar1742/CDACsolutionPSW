#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up EEG Attention Classification Solution...${NC}"

# Check Python version
python_version=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo -e "${RED}Error: Python version must be >= 3.8${NC}"
    echo -e "Current version: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate || source venv/Scripts/activate

# Install requirements
echo -e "${YELLOW}Installing requirements...${NC}"
pip install -r requirements.txt

# Install package in development mode
echo -e "${YELLOW}Installing package in development mode...${NC}"
pip install -e .

# Run tests
echo -e "${YELLOW}Running tests...${NC}"
python -m pytest test_solution.py -v

# Check if output directory exists
if [ ! -d "output" ]; then
    echo -e "${YELLOW}Creating output directory...${NC}"
    mkdir output
fi

# Run the solution
echo -e "${YELLOW}Running the solution...${NC}"
python main.py --validate --save_model --output_dir output

echo -e "${GREEN}Setup and execution completed!${NC}"
echo -e "Results can be found in the 'output' directory"
echo -e "\nTo deactivate the virtual environment, run: deactivate"
