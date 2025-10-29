#!/bin/bash

# Test all model and format permutations
IMAGE_TYPE="candidate"
OUTPUT_FILE="sagemaker_test_results.txt"

# Clear previous results
> "$OUTPUT_FILE"

echo "Starting SageMaker ML endpoint tests at $(date)" | tee -a "$OUTPUT_FILE"
echo "Image type: $IMAGE_TYPE" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"

# All models to test
MODELS=(
    "sklearn-sagemaker-formatters"
    "sklearn-djl-formatters" 
    "sklearn-skops-basic"
    "xgboost-sagemaker-formatters"
    "xgboost-djl-formatters"
    "xgboost-basic"
)

# Multi-model tests
MULTI_MODELS=(
    "sklearn-multi"
)

for model in "${MODELS[@]}"; do
    echo "" | tee -a "$OUTPUT_FILE"
    echo "Testing: $model with JSON and CSV" | tee -a "$OUTPUT_FILE"
    echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
    
    if python3 sagemaker-ml-endpoint-tests.py "$model" "$IMAGE_TYPE" --test-both 2>&1 | tee -a "$OUTPUT_FILE"; then
        if grep -q "Successfully tested" "$OUTPUT_FILE"; then
            echo "SUCCESS: $model (JSON + CSV)" | tee -a "$OUTPUT_FILE"
        else
            echo "FAILED: $model (JSON + CSV) - No success message found" | tee -a "$OUTPUT_FILE"
        fi
    else
        echo "FAILED: $model (JSON + CSV) - Script returned error" | tee -a "$OUTPUT_FILE"
    fi
    
    echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
done

# Test batch predictions
for model in "${MODELS[@]}"; do
    echo "" | tee -a "$OUTPUT_FILE"
    echo "Testing: $model with batch predictions" | tee -a "$OUTPUT_FILE"
    echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
    
    if python3 sagemaker-ml-endpoint-tests.py "$model" "$IMAGE_TYPE" --test-batch --test-json 2>&1 | tee -a "$OUTPUT_FILE"; then
        if grep -q "Successfully tested" "$OUTPUT_FILE"; then
            echo "SUCCESS: $model (batch)" | tee -a "$OUTPUT_FILE"
        else
            echo "FAILED: $model (batch) - No success message found" | tee -a "$OUTPUT_FILE"
        fi
    else
        echo "FAILED: $model (batch) - Script returned error" | tee -a "$OUTPUT_FILE"
    fi
    
    echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
done

# Test multi-model endpoints
for model in "${MULTI_MODELS[@]}"; do
    echo "" | tee -a "$OUTPUT_FILE"
    echo "Testing: $model multi-model endpoint" | tee -a "$OUTPUT_FILE"
    echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
    
    if python3 sagemaker-ml-endpoint-tests.py "$model" "$IMAGE_TYPE" --test-multi-model --test-json 2>&1 | tee -a "$OUTPUT_FILE"; then
        if grep -q "Successfully tested" "$OUTPUT_FILE"; then
            echo "SUCCESS: $model (multi-model)" | tee -a "$OUTPUT_FILE"
        else
            echo "FAILED: $model (multi-model) - No success message found" | tee -a "$OUTPUT_FILE"
        fi
    else
        echo "FAILED: $model (multi-model) - Script returned error" | tee -a "$OUTPUT_FILE"
    fi
    
    echo "----------------------------------------" | tee -a "$OUTPUT_FILE"
done

echo "" | tee -a "$OUTPUT_FILE"
echo "All tests completed at $(date)" | tee -a "$OUTPUT_FILE"
echo "Results saved to: $OUTPUT_FILE"